from typing import Optional, Tuple

import torch
from torch import nn
from torchvision import transforms


class ModelRegistry:
    registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> nn.Module:
        def inner_wrapper(wrapped_class: nn.Module) -> nn.Module:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        model = cls.registry[name]
        model = model(**kwargs)
        return model


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-10)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class BaseDiscriminator(nn.Module):
    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        output_layer: str,
    ):
        super().__init__()
        self.transform = transforms.Normalize(mean, std)
        if output_layer == "sigmoid":
            self.output_layer = nn.Sigmoid()
        else:
            self.output_layer = nn.Identity()
        # self.input = None
        # self.output = None
        # self.register_forward_hook(forward_memory_hook)
        # self.register_backward_hook(backward_hook)

    def get_label(self) -> torch.LongTensor:
        return self.label.data.long()


class BaseGenerator(nn.Module):
    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ):
        super().__init__()
        self.inverse_transform = transforms.Compose(
            [
                NormalizeInverse(mean, std),
                transforms.Lambda(lambda x: torch.clip(x, 0, 1)),
            ]
        )
        # self.input = None
        # self.output = None
        # self.register_forward_hook(forward_memory_hook)

    def sample_label(self, *args, **kwargs):
        return None

    def get_label(self) -> torch.LongTensor:
        return self.label.data.long()


class MemoryModel(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.input = None
        self.output = None

    def forward(self, input):
        if self.input is not None and torch.equal(self.input, input):
            output = self.output
        else:
            output = self.module.forward(input)
            self.output = output
            self.input = input
        return output
