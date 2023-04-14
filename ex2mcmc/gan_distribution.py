from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import torch


class Distribution(ABC):
    """Abstract class for distribution"""

    @abstractmethod
    def log_prob(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Computes log probability of input z"""
        raise NotImplementedError


class DistributionRegistry:
    registry: Dict = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        def inner_wrapper(wrapped_class: Distribution) -> Callable:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Distribution:
        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor


@DistributionRegistry.register()
class PriorTarget(Distribution):
    def __init__(
        self,
        gan,
    ):
        self.gan = gan
        self.proposal = gan.prior

    def log_prob(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.proposal.log_prob(z)

    def project(self, z):
        return self.proposal.project(z)


@DistributionRegistry.register()
class DiscriminatorTarget(Distribution):
    def __init__(self, gan, batch_size: Optional[int] = None):
        self.gan = gan
        self.proposal = gan.prior
        self.batch_size = batch_size
        self.device = next(self.gan.gen.parameters()).device

    def log_prob(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        init_shape = z.shape
        z = z.reshape(-1, init_shape[-1])
        batch_size = kwargs.get("batch_size", self.batch_size or len(z))
        log_prob = torch.empty((0,), device=self.device)
        for chunk_id, chunk in enumerate(torch.split(z, batch_size)):
            if "x" in kwargs:
                x = kwargs["x"][chunk_id * batch_size : (chunk_id + 1) * batch_size].to(
                    self.device
                )
            else:
                x = self.gan.gen(chunk.to(self.device))
            dgz = self.gan.dis(x).squeeze()
            logp_z = self.proposal.log_prob(chunk)
            log_prob = torch.cat([log_prob, (logp_z + dgz) / 1.0])
        return log_prob.reshape(init_shape[:-1])

    def project(self, z):
        return self.proposal.project(z)


@DistributionRegistry.register()
class CondTarget(Distribution):
    def __init__(
        self,
        gan,
        data_batch: Optional[torch.FloatTensor] = None,
        batch_size: Optional[int] = None,
    ):
        self.gan = gan
        self.proposal = gan.prior
        self.data_batch = data_batch
        self.batch_size = batch_size

    def log_prob(
        self,
        z: torch.FloatTensor,
        data_batch: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size = kwargs.get("batch_size", self.batch_size or len(z))
        data_batch = data_batch if data_batch is not None else self.data_batch
        log_prob = torch.empty((0,), device=z.device)
        for chunk, data_chunk in zip(
            torch.split(z, batch_size), torch.split(data_batch, batch_size)
        ):
            logp_xz = (
                -torch.norm(
                    (self.gan.gen(chunk) - data_chunk.to(chunk.device)).reshape(
                        len(chunk), -1
                    ),
                    dim=1,
                )
                ** 2
                / 2
            )
            logp_z = self.proposal.log_prob(chunk)
            if logp_xz.shape != logp_z.shape:
                raise Exception
            log_prob = torch.cat([log_prob, (logp_z + logp_xz) / 1.0])

        return log_prob

    def project(self, z):
        return self.proposal.project(z)
