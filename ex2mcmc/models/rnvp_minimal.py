from typing import Union

import torch
from torch import nn
from torch.distributions import MultivariateNormal as MNormal


class MinimalRNVP(nn.Module):
    def __init__(
        self,
        dim: int,
        device: Union[str, int, torch.device],
        hidden: int = 32,
        num_blocks: int = 4,
        init_weight_scale: float = 1e-4,
        scale: float = 1.0,
    ):
        self.init_weight_scale = init_weight_scale
        device = torch.device(device)
        super().__init__()

        self.prior = MNormal(
            torch.zeros(dim, requires_grad=False).to(device),
            scale**2 * torch.eye(dim, requires_grad=False).to(device),
        )

        masks = num_blocks * [
            [i % 2 for i in range(dim)],
            [(i + 1) % 2 for i in range(dim)],
        ]
        masks = torch.FloatTensor(masks)
        self.masks = nn.Parameter(masks, requires_grad=False)

        self.t = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.LeakyReLU(),
                    nn.Linear(hidden, hidden),
                    nn.LeakyReLU(),
                    nn.Linear(hidden, dim),
                )
                for _ in range(2 * num_blocks)
            ]
        )
        self.s = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.LeakyReLU(),
                    nn.Linear(hidden, hidden),
                    nn.LeakyReLU(),
                    nn.Linear(hidden, dim),
                    nn.Tanh(),
                )
                for _ in range(2 * num_blocks)
            ]
        )

        self.to(device)
        self.init_params([p for n, p in self.named_parameters() if "weight" in n])

    def init_params(self, params):
        # torch.nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain('relu')) #45
        for p in params:
            torch.nn.init.sparse_(p, sparsity=0.3, std=self.init_weight_scale)
            # torch.nn.init.normal_(p, 0, self.init_weight_scale)

    def inverse_flatten(self, z):
        log_det_J_inv, x = z.new_zeros(z.shape[0]), z
        for i in range(len(self.t)):
            x_ = x * self.masks[i]
            s = self.s[i](x_) * (1 - self.masks[i])
            t = self.t[i](x_) * (1 - self.masks[i])
            x = x_ + (1 - self.masks[i]) * (x * torch.exp(s) + t)

            log_det_J_inv += s.sum(dim=1)
        return x, log_det_J_inv

    def inverse(self, z):
        dim = z.shape[-1]
        first_dims = z.shape[:-1]

        x, log_det_J_inv = self.inverse_flatten(z.reshape(-1, dim))
        x = x.reshape(*z.shape)
        log_det_J_inv = log_det_J_inv.reshape(*first_dims)

        return x, log_det_J_inv

    def forward_flatten(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.masks[i] * z
            s = self.s[i](z_) * (1 - self.masks[i])
            t = self.t[i](z_) * (1 - self.masks[i])
            z = (1 - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def forward(self, x):
        dim = x.shape[-1]
        first_dims = x.shape[:-1]

        z, log_det_J = self.forward_flatten(x.reshape(-1, dim))
        z = z.reshape(*x.shape)
        log_det_J = log_det_J.reshape(*first_dims)

        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.forward(x)
        return self.prior.log_prob(z) + logp

    def sample(self, shape):
        z = self.prior.sample(shape).detach()
        x, _ = self.inverse(z)
        # self.log_det_J_inv = log_det_J_inv
        return x
