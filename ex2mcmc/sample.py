import copy
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributions import Distribution as torchDist
from tqdm import trange

from .gan_distribution import Distribution
from .samplers import MCMCRegistry
from .utils import time_comp_cls
from .utils.callbacks import Callback


class Sampler:
    def __init__(
        self,
        gen: nn.Module,
        ref_dist: Union[Distribution, torchDist],
        *,
        burn_in_steps: int = 0,
        start_sample: int = 0,
        n_sampling_steps: int = 1,
        batch_size: Optional[int] = None,
        save_every: int = 1,
        verbose: bool = True,
        n_steps: Optional[int] = None,
        collect_imgs: bool = True,
        sampling: str = "ula",
        mcmc_args: Optional[Dict] = None,
        callbacks: Optional[Iterable[Callback]] = None,
        keep_graph: bool = False,
    ):
        self.gen = gen
        self._ref_dist = ref_dist
        self.burn_in_steps = burn_in_steps
        self.start_sample = start_sample
        self.n_sampling_steps = n_sampling_steps
        self.save_every = save_every
        self.verbose = verbose
        self.trange = trange if self.verbose else range
        self.n_steps = n_steps
        self.collect_imgs = collect_imgs
        self.callbacks = callbacks or []
        self.keep_graph = keep_graph

        self.sampling = sampling
        self.init_mcmc_args: Dict = copy.deepcopy(mcmc_args or dict())
        self.mcmc_args = copy.deepcopy(self.init_mcmc_args)

        self.mcmc = MCMCRegistry()
        self.target = self.ref_dist

    def reset(self):
        self.mcmc_args = copy.deepcopy(self.init_mcmc_args)
        for callback in self.callbacks:
            callback.reset()

    @property
    def ref_dist(self) -> Distribution:
        return self._ref_dist

    def step(
        self,
        z: torch.Tensor,
        it: int = 1,
        meta: Optional[Dict] = None,
        keep_graph: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        pts, meta = self.mcmc(
            self.sampling,
            z,
            self.target,
            proposal=self.gen.proposal,
            # n_samples=self.n_sampling_steps,
            # burn_in=self.n_sampling_steps - 1,
            project=self.ref_dist.project,
            **self.mcmc_args,
            meta=meta,
            keep_graph=keep_graph,
        )

        self.mcmc_args.update(
            {key: meta[key][-1] for key in self.mcmc_args.keys() & meta.keys()}
        )

        return pts[-1], meta

    @time_comp_cls
    def __call__(
        self,
        z: torch.Tensor,
        n_steps: Optional[int] = None,
        collect_imgs: bool = False,
        keep_graph: bool = False,
    ) -> Tuple[List, List, List, List]:
        n_steps = n_steps if n_steps is not None else self.n_steps
        collect_imgs = collect_imgs or self.collect_imgs
        keep_graph = keep_graph or self.keep_graph
        zs = [z.cpu()]
        xs = []
        meta = dict()
        if collect_imgs:
            xs.append(self.gen.inverse_transform(self.gen(z)).detach().cpu())

        it = 0
        for it in self.trange(1, n_steps + 1):
            new_z, meta = self.step(z, it, meta=meta, keep_graph=keep_graph)
            if it > self.start_sample:
                z = new_z.to(z.device)

            if it > self.burn_in_steps and it % self.save_every == 0:
                if keep_graph:
                    zs.append(z.cpu())
                else:
                    zs.append(z.detach().cpu())
                if collect_imgs:
                    xs.append(
                        self.gen.inverse_transform(self.gen(z.detach())).detach().cpu()
                    )

            for callback in self.callbacks:
                callback.invoke(self.mcmc_args)

        return zs, xs
