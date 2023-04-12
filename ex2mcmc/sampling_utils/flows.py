import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions.transforms import AffineCoupling


# from pyro.nn import DenseNN


class ConditionalDenseNN(nn.Module):
    """
    An implementation of a simple dense feedforward network taking a context variable, for use in, e.g.,
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
        # torch.nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.sparse_(params, sparsity=0.3, std=self.init_weight_scale)

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
    An implementation of a simple dense feedforward network, for use in, e.g., some conditional flows such as
    :class:`pyro.distributions.transforms.ConditionalPlanarFlow` and other unconditional flows such as
    :class:`pyro.distributions.transforms.AffineCoupling` that do not require an autoregressive network.

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
        when p_n > 1 and dimension () when p_n == 1. The default is [1, 1], i.e. output two parameters of dimension ().
    :type param_dims: list[int]
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
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
        self, num_flows, dim, flows=None, device="cpu", init_weight_scale=1e-4
    ):
        super().__init__()
        self.device = device
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
                            init_weight_scale=init_weight_scale,
                        ),
                    )
                    for _ in range(num_flows)
                ],
            )
        self.flow.to(self.device)
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
                z = torch.index_select(z, 1, self.eo)
            else:
                z = torch.index_select(z, 1, self.oe)
        else:
            if i % 2 == 0:
                z = torch.index_select(z, 1, self.reverse_eo)
            else:
                z = torch.index_select(z, 1, self.reverse_oe)
        return z

    def forward(self, z):
        log_jacob = torch.zeros_like(z[:, 0], dtype=torch.float32).to(self.device)
        for i, current_flow in enumerate(self.flow):
            z = self.permute(z, i)
            z_new = current_flow(z)
            log_jacob += current_flow.log_abs_det_jacobian(z, z_new)
            z_new = self.permute(z_new, i, reverse=True)
            z = z_new
        return z, log_jacob

    def inverse(self, z):
        log_jacob = torch.zeros_like(z[:, 0], dtype=torch.float32).to(self.device)
        n = len(self.flow) - 1
        for i, current_flow in enumerate(self.flow[::-1]):
            z = self.permute(z, n - i)
            z_new = current_flow._inverse(z)
            log_jacob -= current_flow.log_abs_det_jacobian(z_new, z)
            z_new = self.permute(z_new, n - i, reverse=True)
            z = z_new
        return z, log_jacob


"""
=========================================
"""


class MLP(nn.Module):
    def __init__(self, layerdims, activation=torch.relu, init_scale=None):
        super().__init__()
        self.layerdims = layerdims
        self.activation = activation
        linears = [
            nn.Linear(layerdims[i], layerdims[i + 1]) for i in range(len(layerdims) - 1)
        ]

        if init_scale is not None:
            for l, layer in enumerate(linears):
                torch.nn.init.normal_(
                    layer.weight,
                    std=init_scale / np.sqrt(layerdims[l]),
                )
                torch.nn.init.zeros_(layer.bias)

        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        layers = list(enumerate(self.linears))
        for _, l in layers[:-1]:
            x = self.activation(l(x))
        y = layers[-1][1](x)
        return y


class ResidualAffineCoupling(nn.Module):
    """Residual Affine Coupling layer
    Implements coupling layers with a rescaling
    Args:
        s (nn.Module): scale network
        t (nn.Module): translation network
        mask (binary tensor): binary array of same
        dt (float): rescaling factor for s and t
    """

    def __init__(self, s=None, t=None, mask=None, dt=1):
        super().__init__()

        self.mask = mask
        self.scale_net = s
        self.trans_net = t
        self.dt = dt

    def forward(self, x, log_det_jac=None, inverse=False):
        if log_det_jac is None:
            log_det_jac = 0

        s = self.mask * self.scale_net(x * (1 - self.mask))
        s = torch.tanh(s)
        t = self.mask * self.trans_net(x * (1 - self.mask))

        s = self.dt * s
        t = self.dt * t

        if inverse:
            if torch.isnan(torch.exp(-s)).any():
                raise RuntimeError("Scale factor has NaN entries")
            log_det_jac -= s.view(s.size(0), -1).sum(-1)

            x = x * torch.exp(-s) - t

        else:
            log_det_jac += s.view(s.size(0), -1).sum(-1)
            x = (x + t) * torch.exp(s)
            if torch.isnan(torch.exp(s)).any():
                raise RuntimeError("Scale factor has NaN entries")

        return x, log_det_jac


class RealNVP_MLP(nn.Module):
    """Minimal Real NVP architecture

    Args:
        dims (int,): input dimension
        n_realnvp_blocks (int): number of pairs of coupling layers
        block_depth (int): repetition of blocks with shared param
        init_weight_scale (float): scaling factor for weights in s and t layers
        prior_arg (dict): specifies the base distribution
        mask_type (str): 'half' or 'inter' masking pattern
        hidden_dim (int): # of hidden neurones per layer (coupling MLPs)
    """

    def __init__(
        self,
        dim,
        n_realnvp_blocks,
        block_depth,
        init_weight_scale=None,
        prior_arg={"type": "standn"},
        mask_type="half",
        hidden_dim=10,
        device="cpu",
    ):
        super().__init__()

        self.device = device
        self.dim = dim
        self.n_blocks = n_realnvp_blocks
        self.block_depth = block_depth
        self.couplings_per_block = 2  # one update of entire layer per block
        self.n_layers_in_coupling = 3  # depth of MLPs in coupling layers
        self.hidden_dim_in_coupling = hidden_dim
        self.init_scale_in_coupling = init_weight_scale

        # beta_prior = prior_arg["beta"]
        coef = prior_arg["alpha"] * dim
        prec = torch.eye(dim) * (3 * coef + 1 / coef)
        prec -= coef * torch.triu(
            torch.triu(torch.ones_like(prec), diagonal=-1).T,
            diagonal=-1,
        )
        prec = prior_arg["beta"] * prec
        self.prior_prec = prec.to(device)
        self.prior_log_det = -torch.logdet(prec)

        # proposal = MultivariateNormal(
        #     torch.zeros((dim,), device=device),
        #     precision_matrix=prior_prec)

        mask = torch.ones(dim, device=self.device)
        if mask_type == "half":
            mask[: int(dim / 2)] = 0
        elif mask_type == "inter":
            idx = torch.arange(dim, device=self.device)
            mask = mask * (idx % 2 == 0)
        else:
            raise RuntimeError("Mask type is either half or inter")
        self.mask = mask.view(1, dim)

        self.coupling_layers = self.initialize()

        self.beta = 1.0  # effective temperature needed e.g. in Langevin

        # self.prior_arg = prior_arg

        # if prior_arg['type'] == 'standn':
        #     self.prior_prec =  torch.eye(dim).to(device)
        #     self.prior_log_det = 0
        #     self.prior_distrib = MultivariateNormal(
        #         torch.zeros((dim,), device=self.device), self.prior_prec)

        # elif prior_arg['type'] == 'uncoupled':
        #     self.prior_prec = prior_arg['a'] * torch.eye(dim).to(device)
        #     self.prior_log_det = - torch.logdet(self.prior_prec)
        #     self.prior_distrib = MultivariateNormal(
        #         torch.zeros((dim,), device=self.device),
        #         precision_matrix=self.prior_prec)

        # elif prior_arg['type'] == 'coupled':
        #     self.beta_prior = prior_arg['beta']
        #     self.coef = prior_arg['alpha'] * dim
        #     prec = torch.eye(dim) * (3 * self.coef + 1 / self.coef)
        #     prec -= self.coef * torch.triu(torch.triu(torch.ones_like(prec),
        #                                               diagonal=-1).T, diagonal=-1)
        #     prec = prior_arg['beta'] * prec
        #     self.prior_prec = prec.to(self.device)
        #     self.prior_log_det = - torch.logdet(prec)
        #     self.prior_distrib = MultivariateNormal(
        #         torch.zeros((dim,), device=self.device),
        #         precision_matrix=self.prior_prec)

        # elif prior_arg['type'] == 'bridge':
        #     self.bridge_kwargs = prior_arg['bridge_kwargs']

        # else:
        #     raise NotImplementedError("Invalid prior arg type")

    def forward(self, x, return_per_block=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)

        if return_per_block:
            xs = [x]
            log_det_jacs = [log_det_jac]

        for block in range(self.n_blocks):
            couplings = self.coupling_layers[block]

            for dt in range(self.block_depth):
                for coupling_layer in couplings:
                    x, log_det_jac = coupling_layer(x, log_det_jac)

                if return_per_block:
                    xs.append(x)
                    log_det_jacs.append(log_det_jac)

        if return_per_block:
            return xs, log_det_jacs
        else:
            return x, log_det_jac

    def backward(self, x, return_per_block=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)

        if return_per_block:
            xs = [x]
            log_det_jacs = [log_det_jac]

        for block in range(self.n_blocks):
            couplings = self.coupling_layers[::-1][block]

            for dt in range(self.block_depth):
                for coupling_layer in couplings[::-1]:
                    x, log_det_jac = coupling_layer(
                        x,
                        log_det_jac,
                        inverse=True,
                    )

                if return_per_block:
                    xs.append(x)
                    log_det_jacs.append(log_det_jac)

        if return_per_block:
            return xs, log_det_jacs
        else:
            return x, log_det_jac

    def initialize(self):
        dim = self.dim
        coupling_layers = []

        for block in range(self.n_blocks):
            layer_dims = [self.hidden_dim_in_coupling] * (self.n_layers_in_coupling - 2)
            layer_dims = [dim] + layer_dims + [dim]

            couplings = self.build_coupling_block(layer_dims)

            coupling_layers.append(nn.ModuleList(couplings))
        return nn.ModuleList(coupling_layers)

    def build_coupling_block(self, layer_dims=None, nets=None, reverse=False):
        count = 0
        coupling_layers = []
        for count in range(self.couplings_per_block):
            s = MLP(layer_dims, init_scale=self.init_scale_in_coupling)
            s = s.to(self.device)
            t = MLP(layer_dims, init_scale=self.init_scale_in_coupling)
            t = t.to(self.device)

            if count % 2 == 0:
                mask = 1 - self.mask
            else:
                mask = self.mask

            dt = self.n_blocks * self.couplings_per_block * self.block_depth
            dt = 2 / dt
            coupling_layers.append(ResidualAffineCoupling(s, t, mask, dt=dt))

        return coupling_layers

    def inverse(self, x):
        return self.backward(x)

    def nll(self, x):
        z, log_det_jac = self.backward(x)

        # if self.prior_arg['type']=='bridge':
        #     a_min = torch.tensor([self.bridge_kwargs["x0"],self.bridge_kwargs["y0"]])
        #     b_min = torch.tensor([self.bridge_kwargs["x1"],self.bridge_kwargs["y1"]])
        #     dt = self.bridge_kwargs["dt"]
        #     prior_nll = bridge_energy(z, dt=dt, a_min=a_min, b_min=b_min, device=self.device)
        #     return prior_nll - log_det_jac

        prior_ll = -0.5 * torch.einsum("ki,ij,kj->k", z, self.prior_prec, z)
        prior_ll -= 0.5 * (self.dim * np.log(2 * np.pi) + self.prior_log_det)

        ll = prior_ll + log_det_jac
        nll = -ll
        return nll
