from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
from pyro.distributions.transforms import AffineCoupling
from torch.distributions import MultivariateNormal as MNormal


class ConditionalDenseNN(nn.Module):
    """
    An implementation of a simple dense feedforward network taking a context variable, for use
      in, e.g.,
    some conditional flows such as :class:`pyro.distributions.transforms.ConditionalAffineCoupling`.

    Example usage:

    >>> input_dim = 10
    >>> context_dim = 5
    >>> x = torch.rand(100, input_dim)
    >>> z = torch.rand(100, context_dim)
    >>> nn = ConditionalDenseNN(input_dim, context_dim, [50], param_dims=[1, input_dim, input_dim])
    >>> a, b, c = nn(x, context=z)  # parameters of size (100, 1), (100, 10), (100, 10)

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param context_dim: the dimensionality of the context variable
    :type context_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n,) for p_n in param_dims
        when p_n > 1 and dimension () when p_n == 1. The default is [1, 1], i.e. output two parameters of dimension ().
    :type param_dims: list[int]
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.Module

    """

    def __init__(
        self,
        input_dim,
        context_dim,
        hidden_dims,
        param_dims=[1, 1],
        nonlinearity=torch.nn.ReLU(),
        init_weight_scale=1e-3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)
        self.init_weight_scale = init_weight_scale

        # Calculate the indices on the output corresponding to each parameter
        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        # Create masked layers
        layers = [torch.nn.Linear(input_dim + context_dim, hidden_dims[0])]
        self.init_params(layers[0].weight)

        for i in range(1, len(hidden_dims)):
            layers.append(torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            self.init_params(layers[-1].weight)
        layers.append(torch.nn.Linear(hidden_dims[-1], self.output_multiplier))
        self.init_params(layers[-1].weight)
        self.layers = torch.nn.ModuleList(layers)

        # Save the nonlinearity
        self.f = nonlinearity

    def forward(self, x, context):
        # We must be able to broadcast the size of the context over the input
        context = context.expand(x.size()[:-1] + (context.size(-1),))

        x = torch.cat([context, x], dim=-1)
        return self._forward(x)

    def init_params(self, params):
        # torch.nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain('relu')) #45
        torch.nn.init.sparse_(params, sparsity=0.3, std=self.init_weight_scale)
        # torch.nn.init.normal_(params, 0, self.init_weight_scale) 46

    def _forward(self, x):
        """
        The forward method
        """
        h = x
        for layer in self.layers[:-1]:
            h = self.f(layer(h))
        h = self.layers[-1](h)

        # Shape the output, squeezing the parameter dimension if all ones
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier])

            if self.count_params == 1:
                return h

            else:
                return tuple(h[..., s] for s in self.param_slices)


class DenseNN(ConditionalDenseNN):
    """
    An implementation of a simple dense feedforward network, for use in, e.g., some conditional
      flows such as
    :class:`pyro.distributions.transforms.ConditionalPlanarFlow` and other unconditional flows
      such as
    :class:`pyro.distributions.transforms.AffineCoupling` that do not require an autoregressive
      network.

    Example usage:

    >>> input_dim = 10
    >>> context_dim = 5
    >>> z = torch.rand(100, context_dim)
    >>> nn = DenseNN(context_dim, [50], param_dims=[1, input_dim, input_dim])
    >>> a, b, c = nn(z)  # parameters of size (100, 1), (100, 10), (100, 10)

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n,) for p_n in param_dims
        when p_n > 1 and dimension () when p_n == 1. The default is [1, 1], i.e. output two
      parameters of dimension ().
    :type param_dims: list[int]
    :param nonlinearity: The nonlinearity to use in the feedforward network such as
      torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded
        real number.
    :type nonlinearity: torch.nn.module

    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        param_dims=[1, 1],
        nonlinearity=torch.nn.ReLU(),
        init_weight_scale=1e-3,
    ):
        super().__init__(
            input_dim,
            0,
            hidden_dims,
            param_dims=param_dims,
            nonlinearity=nonlinearity,
            init_weight_scale=init_weight_scale,
        )

    def forward(self, x):
        return self._forward(x)


class RNVP(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        dim: int,
        flows: Optional[Iterable] = None,
        init_weight_scale: float = 1e-4,
        device: Union[str, int, torch.device] = 0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.init_weight_scale = init_weight_scale
        self.x = None
        split_dim = dim // 2
        param_dims = [dim - split_dim, dim - split_dim]
        if flows is not None:
            self.flow = nn.ModuleList(flows)
        else:
            # hypernet = DenseNN(split_dim, [2 * dim], param_dims)
            self.flow = nn.ModuleList(
                [
                    AffineCoupling(
                        split_dim,
                        DenseNN(
                            split_dim,
                            [2 * dim],
                            param_dims,
                        ),
                    )
                    for _ in range(num_blocks)
                ],
            )
            self.init_params(self.parameters())

        even = [i for i in range(0, dim, 2)]
        odd = [i for i in range(1, dim, 2)]
        reverse_eo = [
            i // 2 if i % 2 == 0 else (i // 2 + len(even)) for i in range(dim)
        ]
        reverse_oe = [(i // 2 + len(odd)) if i % 2 == 0 else i // 2 for i in range(dim)]
        self.register_buffer("eo", torch.tensor(even + odd, dtype=torch.int64))
        self.register_buffer("oe", torch.tensor(odd + even, dtype=torch.int64))
        self.register_buffer(
            "reverse_eo",
            torch.tensor(reverse_eo, dtype=torch.int64),
        )
        self.register_buffer(
            "reverse_oe",
            torch.tensor(reverse_oe, dtype=torch.int64),
        )

        self.prior = MNormal(
            torch.zeros(dim).to(device), scale**2 * torch.eye(dim).to(device)
        )
        self.to(device)

    def init_params(self, params):
        # torch.nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain('relu'))
        for p in params:
            if p.ndim == 2:
                torch.nn.init.sparse_(p, sparsity=0.3, std=self.init_weight_scale)
        # torch.nn.init.normal_(params, 0, self.init_weight_scale)

    def to(self, *args, **kwargs):
        """
        overloads to method to make sure the manually registered buffers are sent to device
        """
        self = super().to(*args, **kwargs)
        self.eo = self.eo.to(*args, **kwargs)
        self.oo = self.oe.to(*args, **kwargs)
        self.reverse_eo = self.reverse_eo.to(*args, **kwargs)
        self.reverse_oe = self.reverse_oe.to(*args, **kwargs)
        return self

    def permute(self, z, i, reverse=False):
        if not reverse:
            if i % 2 == 0:
                z = torch.index_select(z, -1, self.eo)
            else:
                z = torch.index_select(z, -1, self.oe)
        else:
            if i % 2 == 0:
                z = torch.index_select(z, -1, self.reverse_eo)
            else:
                z = torch.index_select(z, -1, self.reverse_oe)
        return z

    def forward(self, x):
        log_jacob = torch.zeros_like(x[..., 0], dtype=torch.float32)
        for i, current_flow in enumerate(self.flow):
            x = self.permute(x, i)
            z = current_flow(x)
            log_jacob += current_flow.log_abs_det_jacobian(x, z)  # z, z_new)
            z = self.permute(z, i, reverse=True)
            x = z
        return z, log_jacob

    def inverse(self, z):
        log_jacob_inv = torch.zeros_like(z[..., 0], dtype=torch.float32)
        n = len(self.flow) - 1
        for i, current_flow in enumerate(self.flow[::-1]):
            z = self.permute(z, n - i)
            x = current_flow._inverse(z)
            log_jacob_inv -= current_flow.log_abs_det_jacobian(x, z)
            x = self.permute(x, n - i, reverse=True)
            z = x
        return x, log_jacob_inv.reshape(z.shape[:-1])

    def log_prob(self, x):
        if self.x is not None and torch.equal(self.x, x):
            z, logp = self.z, self.log_jacob
        else:
            z, logp = self.forward(x)
        return self.prior.log_prob(z) + logp

    # def log_prob(self, x):
    #     z, logp = self.inverse(x)
    #     return self.prior.log_prob(z) + logp

    def sample(self, shape):
        z = self.prior.sample(shape)
        x, log_jacob_inv = self.inverse(z)
        self.log_jacob = -log_jacob_inv
        self.x = x
        self.z = z
        return x
