from abc import ABC, abstractmethod

import numpy as np
import numpy.random as rng
import torch
import torch.distributions as td
import torch.nn.functional as F
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from torch import nn

from .linear_regression import RegressionDataset
from .logistic_regression import ClassificationDataset


torchType = torch.float32


class Distribution(ABC):
    """
    Base class for a custom target distribution
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.torchType = torchType
        self.xlim, self.ylim = [-1, 1], [-1, 1]
        self.scale_2d_log_prob = 1
        # self.device_zero = torch.tensor(0., dtype=self.torchType, device=self.device)
        # self.device_one = torch.tensor(1., dtype=self.torchType, device=self.device)

    def prob(self, x):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
        # You should define the class for your custom distribution
        return self.log_prob(x).exp()

    @abstractmethod
    def log_prob(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def energy(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        energy = -log p(x)
        """
        # You should define the class for your custom distribution
        return -self.log_prob(x)

    def sample(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def __call__(self, x):
        return self.log_prob(x)

    def log_prob_2d_slice(self, z):
        raise NotImplementedError

    def plot_2d(self, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        x = np.linspace(*self.xlim, 100)
        y = np.linspace(*self.ylim, 100)
        xx, yy = np.meshgrid(x, y)
        z = torch.FloatTensor(np.stack([xx, yy], -1))
        vals = (self.log_prob_2d_slice(z) / self.scale_2d_log_prob).exp()

        if ax is not None:
            ax.imshow(
                vals.flip(0),
                extent=[*self.xlim, *self.ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )
        else:
            plt.imshow(
                vals.flip(0),
                extent=[*self.xlim, *self.ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )

        return fig, self.xlim, self.ylim


class GaussianMixture(Distribution):
    """
    Mixture of n gaussians (multivariate)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.locs = kwargs.get(
            "locs",
            torch.FloatTensor([[-1, 0], [1, 0]]),
        )  # list of locations for each of these gaussians
        self.num = kwargs.get("num_gauss", len(self.locs))
        self.pis = kwargs.get(
            "p_gaussians",
            torch.FloatTensor([1.0 / self.num] * self.num),
        )
        self.dim = kwargs.get("dim", self.locs.shape[1])
        self.sigma = kwargs.get("sigma", 0.2)
        self.covs = kwargs.get(
            "covs",
            [self.sigma**2 * torch.eye(self.dim)] * self.num,
        )  # list of covariance matrices for each of these gaussians

        self.peak = [None] * self.num
        for i in range(self.num):
            self.peak[i] = torch.distributions.MultivariateNormal(
                loc=self.locs[i],
                covariance_matrix=self.covs[i],
            )

    def get_density(self, x):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        density = self.log_prob(x).exp()
        return density

    def log_prob(self, z):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent variable
        Output:
        log_density - log p(x)
        """
        log_p = torch.tensor([], device=self.device)
        # pdb.set_trace()
        for i in range(self.num):
            log_paux = (
                torch.log(self.pis[i]) + self.peak[i].log_prob(z.to(self.device))
            ).view(-1, 1)
            log_p = torch.cat([log_p, log_paux], dim=-1)
        log_density = torch.logsumexp(log_p, dim=1)
        return log_density.view(z.shape[:-1])

    def energy(self, z, x=None):
        return -self.log_prob(z, x)

    def plot_2d(self):
        if self.dim != 2:
            raise NotImplementedError("can't plot for gaussians not in 2d")

        fig, ax = plt.subplots()

        dim1 = 0
        dim2 = 1

        xlim = [
            self.locs[:, dim1].min().item() - 1,
            self.locs[:, dim1].max().item() + 1,
        ]
        ylim = [
            self.locs[:, dim2].min().item() - 1,
            self.locs[:, dim2].max().item() + 1,
        ]

        x = np.linspace(*xlim, 100)
        y = np.linspace(*ylim, 100)
        xx, yy = np.meshgrid(x, y)
        z = torch.FloatTensor(np.stack([xx, yy], -1))
        vals = self.log_prob(z).exp()

        plt.contourf(
            vals,
            extent=[*xlim, *ylim],
            cmap="Greens",
            alpha=0.5,
            aspect="auto",
        )

        return fig, xlim, ylim


class IndependentNormal(Distribution):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.dim = dim
        self.loc = kwargs.get("loc", torch.zeros(self.dim))
        self.scale = kwargs.get("scale", 1.0)
        self.distribution = torch.distributions.Normal(
            loc=self.loc,
            scale=self.scale,
        )

    def log_prob(self, z, x=None):
        log_density = (self.distribution.log_prob(z.to(self.device))).sum(
            dim=-1,
        )
        return log_density

    def sample(self, n):
        return self.distribution.sample(n)

    def energy(self, z, x=None):
        return -self.log_prob(z, x)


def init_independent_normal(scale, n_dim, device, loc=0.0):
    loc = loc * torch.ones(n_dim).to(device)
    scale = scale * torch.ones(n_dim).to(device)
    target_args = edict()
    target_args.device = device
    target_args.loc = loc
    target_args.scale = scale
    target = IndependentNormal(n_dim, **target_args)
    return target


def init_independent_normal_scale(scales, locs, device):
    target_args = edict()
    target_args.device = device
    target_args.loc = locs.to(device)
    target_args.scale = scales.to(device)
    target = IndependentNormal(**target_args)
    return target


class Cauchy(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loc = kwargs.get("loc")
        self.scale = kwargs.get("scale")
        self.dim = kwargs.get("dim")
        self.distr = torch.distributions.Cauchy(self.loc, self.scale)

    def log_prob(self, z, x=None):
        log_target = self.distr.log_prob(z)  # .sum(-1)

        return log_target

    def sample(self, n=(1,)):
        return self.distr.sample((n[0], self.dim))


class CauchyMixture(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.dim = kwargs.get("dim", 48)
        self.mu = kwargs.get("mu", 2 * torch.ones(self.dim))
        self.cov = kwargs.get("cov", 0.7 * torch.ones(self.dim))
        self.cauchy = Cauchy(loc=self.mu, scale=self.cov)
        self.cauchy_minus = Cauchy(loc=-self.mu, scale=self.cov)
        self.xlim = [-5, 5]
        self.ylim = [-5, 5]

    def get_density(self, x):
        return self.log_prob(x).exp()

    def log_prob_2d_slice(self, z, dim1=0, dim2=1):
        mu = self.mu[[dim1, dim2]]
        cov = self.cov[[dim1, dim2]]

        cauchy = Cauchy(loc=mu, scale=cov)
        cauchy_minus = Cauchy(loc=-mu, scale=cov)

        catted = torch.cat(
            [
                cauchy.log_prob(z)[None, ...],
                cauchy_minus.log_prob(z)[None, ...],
            ],
            0,
        )
        log_target = torch.logsumexp(catted, 0).sum(-1) - 2 * torch.tensor(2.0).log()
        # log_target = log_target[:, None] +

        return log_target  # + torch.tensor([8.]).log()

    def log_prob(self, z, x=None):
        catted = torch.cat(
            [
                self.cauchy.log_prob(z)[None, ...],
                self.cauchy_minus.log_prob(z)[None, ...],
            ],
            0,
        )
        log_target = (
            torch.logsumexp(catted, 0).sum(-1) - self.dim * torch.tensor(2.0).log()
        )

        # print(z.shape, log_target.shape)

        return log_target  # + torch.tensor([8.]).log()


class Funnel(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.a = (kwargs.get("a", 1.0) * torch.ones(1)).to(self.device)
        self.b = kwargs.get("b", 0.5)
        self.dim = kwargs.get("dim", 16)
        self.distr1 = torch.distributions.Normal(torch.zeros(1).to(self.device), self.a)
        # self.distr2 = lambda z1: torch.distributions.MultivariateNormal(torch.zeros(self.dim-1), (2*self.b*z1).exp()*torch.eye(self.dim-1))
        # self.distr2 = lambda z1: -(z[...,1:]**2).sum(-1) * (-2*self.b*z1).exp() - np.log(self.dim) + 2*self.b*z1
        self.xlim = [-2, 10]
        self.ylim = [-30, 30]
        self.scale_2d_log_prob = 20  # 30.0

    def log_prob(self, z, x=None):
        logprob1 = self.distr1.log_prob(z[..., 0])
        z1 = z[..., 0]
        logprob2 = (
            -0.5 * (z[..., 1:] ** 2).sum(-1) * torch.exp(-2 * self.b * z1)
            - (self.dim - 1) * self.b * z1
        )
        return logprob1 + logprob2

    def log_prob_2d_slice(self, z, dim1=0, dim2=1):
        if dim1 == 0 or dim2 == 0:
            logprob1 = self.distr1.log_prob(z[..., 0])
            dim2 = dim2 if dim2 != 0 else dim1
            z1 = z[..., 0]
            # logprob2 = self.distr2(z[...,0])
            logprob2 = (
                -0.5 * (z[..., dim2] ** 2) * torch.exp(-2 * self.b * z1) - self.b * z1
            )
        # else:
        #     logprob2 = -(z[...,dim2]**2) * (-2*self.b*z1).exp() - np.log(self.dim) + 2*self.b*z1
        return logprob1 + logprob2

    def plot_2d_countour(self, ax):
        x = np.linspace(-15, 15, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        inp = torch.from_numpy(np.stack([X, Y], -1))
        Z = self.log_prob(inp.reshape(-1, 2)).reshape(inp.shape[:-1])

        # levels = np.quantile(Z, np.linspace(0.9, 0.99, 5))
        ax.contour(
            X,
            Y,
            Z.exp(),
            # levels = levels,
            levels=3,
            alpha=1.0,
            cmap="inferno",
        )

    def plot_2d(self, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

            xlim = [-2, 13]
            ylim = [-60, 60]
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        x = np.linspace(*xlim, 100)
        y = np.linspace(*ylim, 100)
        xx, yy = np.meshgrid(x, y)
        z = torch.FloatTensor(np.stack([xx, yy], -1))
        vals = (self.log_prob_2d_slice(z) / self.scale_2d_log_prob).exp()
        if ax is not None:
            ax.imshow(
                vals.flip(0),
                extent=[*xlim, *ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )
        else:
            plt.imshow(
                vals.flip(0),
                extent=[*xlim, *ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )

        return fig, xlim, ylim


class HalfBanana(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Q = kwargs.get("Q", 0.01) * torch.ones(1)
        self.dim = kwargs.get("dim", 32)
        self.xlim = [-1, 9]
        self.ylim = [-2, 4]
        self.scale_2d_log_prob = 2.0
        # assert self.dim % 2 == 0, 'Dimension should be divisible by 2'

    def log_prob(self, z, x=None):
        # n = self.dim/2
        even = np.arange(0, self.dim, 2)
        odd = np.arange(1, self.dim, 2)

        ll = -((z[..., even] - z[..., odd] ** 2) ** 2) / self.Q - (z[..., odd] - 1) ** 2
        return ll.sum(-1)

    def log_prob_2d_slice(self, z, dim1=0, dim2=1):
        if dim1 % 2 == 0 and dim2 % 2 == 1:
            ll = (
                -((z[..., dim1] - z[..., dim2] ** 2) ** 2) / self.Q
                - (z[..., dim2] - 1) ** 2
            )
        return ll  # .sum(-1)


class Banana(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.b = kwargs.get("b", 0.02)  # * torch.ones(1).to(self.device)
        self.sigma = kwargs.get("sigma", 100.0)  # * torch.ones(1).to(self.device)
        self.dim = kwargs.get("dim", 32)
        self.xlim = [-1, 5]
        self.ylim = [-2, 2]
        self.scale_2d_log_prob = 2.0
        # assert self.dim % 2 == 0, 'Dimension should be divisible by 2'

    def log_prob(self, z, x=None):
        # n = self.dim/2
        even = np.arange(0, self.dim, 2)
        odd = np.arange(1, self.dim, 2)

        ll = -0.5 * (
            z[..., odd] - self.b * z[..., even] ** 2 + (self.sigma**2) * self.b
        ) ** 2 - ((z[..., even]) ** 2) / (2 * self.sigma**2)
        return ll.sum(-1)

    def log_prob_2d_slice(self, z, dim1=0, dim2=1):
        if dim1 % 2 == 0 and dim2 % 2 == 1:
            ll = (
                -((z[..., dim1] - z[..., dim2] ** 2) ** 2) / self.Q
                - (z[..., dim1] - 1) ** 2
            )
        return ll  # .sum(-1)


class BayesianLogRegression(Distribution):
    tau: float
    # dataset : ClassificationDataset

    def __init__(self, dataset: ClassificationDataset, **kwargs):
        super().__init__(**kwargs)
        # self.dataset = dataset
        self.x_train = dataset.x_train
        self.y_train = dataset.y_train
        self.d = dataset.d
        self.n = dataset.n
        self.tau = kwargs.get("tau", 0.05)

    def log_prob(self, theta, x=None, y=None):
        if x is None:
            x = self.x_train
        if y is None:
            y = self.y_train

        max_val = 1e5

        prod = torch.clamp(
            torch.matmul(x, theta.transpose(0, 1)),
            min=-max_val,
        )
        # P = 1. / (1. + torch.exp(-prod))
        P = torch.sigmoid(prod)
        # mask = torch.isnan(P)
        # P = P[mask]
        # P[mask] = 1e-5
        ll = torch.matmul(
            y,
            torch.log(torch.clamp(P, min=1e-5)),
        ) + torch.matmul(1 - y, torch.log(torch.clamp(1 - P, min=1e-5)))

        ll = ll - self.tau / 2 * (theta**2).sum(-1)
        return ll

    # def grad_log_prob(self, theta, x= None, y = None):
    #     if x is None:
    #         x = self.x_train
    #     if y is None:
    #         y = self.y_train
    #     P = 1. / (1. + torch.exp(-torch.mm(x,theta)))
    #     return torch.matmul(torch.transpose(X, -2, -1), (y - P).view(self.n)) - self.tau * theta

    # def log_prob(self, theta, x= None, y = None):
    #     if x is None:
    #         x = self.x_train
    #     if y is None:
    #         y = self.y_train
    #     P = 1. / (1. + torch.exp(-torch.matmul(x,theta.transpose(0,1))))
    #     ll =  torch.matmul(y,torch.log(P)) + torch.matmul(1-y,torch.log(1-P)) - self.tau/2 * (theta**2).sum(-1)
    #     return ll


class BayesianLinearRegression(Distribution):
    def __init__(self, dataset: RegressionDataset, **kwargs):
        super().__init__(**kwargs)
        self.x_train = dataset.x_train
        self.y_train = dataset.y_train
        self.d = dataset.d
        self.n = dataset.n
        self.tau = kwargs.get("tau", 0.05)
        self.noise_scale = kwargs.get("noise_scale", 0.1)

    def log_prob(self, theta, x=None, y=None):
        if x is None:
            x = self.x_train
        if y is None:
            y = self.y_train

        # max_val = 1e5

        ll = -((y - torch.dot(x, theta)) ** 2) / (2 * self.noise_scale**2)

        # prod = torch.clamp(torch.matmul(x, theta.transpose(0,1)), min=-max_val)
        # #P = 1. / (1. + torch.exp(-prod))
        # P = torch.sigmoid(prod)
        # #print(P.max())
        # mask = torch.isnan(P)
        # #P = P[mask]
        # #P[mask] = 1e-5
        # ll =  torch.matmul(y, torch.log(torch.clamp(P, min=1e-5))) + torch.matmul(1-y, torch.log(torch.clamp(1 - P, min=1e-5)))

        ll = ll - self.tau / 2 * (theta**2).sum(-1)
        return ll


class GMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(self.logits_prior.sample((self.n_samples,)))
        self.log_vars = nn.Parameter(
            self.vars_prior.sample(
                (self.n_samples, self.n_components, self.dim),
            ).double(),
        )
        self.means = nn.Parameter(
            torch.normal(0.0, 2 * (0.5 * self.log_vars).exp()),
        )


class GaussianMixtureModel(Distribution, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_components: int = kwargs.get("n_components", 5)
        self.dim: int = kwargs.get("n_components", 5)
        self.n_samples: int = kwargs.get("n_samples", 1)
        self.stick_break = td.StickBreakingTransform()

        self.logits_prior = td.TransformedDistribution(
            td.Dirichlet(torch.ones(self.n_components, dtype=torch.double)),
            self.stick_break.inv,
        )

        self.vars_prior = td.TransformedDistribution(
            td.Gamma(1, 1),
            td.ComposeTransform(
                [td.ExpTransform().inv, td.AffineTransform(0, -1)],
            ),
        )

        # self.model = GMM(self.logits_prior, self.vars_prior, self.n_samples, self.n_components, self.dim)

        with torch.no_grad():
            self.logits = nn.Parameter(
                self.logits_prior.sample((self.n_samples,)),
            )
            self.log_vars = nn.Parameter(
                self.vars_prior.sample(
                    (self.n_samples, self.n_components, self.dim),
                ).double(),
            )
            self.means = nn.Parameter(
                torch.normal(0.0, 2 * (0.5 * self.log_vars).exp()),
            )

        self.data = self.sample_data(n_samples=1000)

    def forward(self, x):
        sigmas = (0.5 * self.log_vars).exp()
        betas = self.stick_break(self.logits)

        x = x[:, None, None, :]

        log_ps = td.Normal(self.means, sigmas).log_prob(x).sum(dim=-1)
        log_ps = torch.logsumexp(log_ps + betas.log(), dim=2)

        return log_ps

    def log_prior(self):
        p_logits = self.logits_prior.log_prob(self.logits)
        p_sigma = self.vars_prior.log_prob(self.log_vars).sum(dim=-1).sum(dim=-1)
        means_prior = td.Normal(0.0, 2 * (0.5 * self.log_vars).exp())
        p_means = means_prior.log_prob(self.means).sum(dim=-1).sum(dim=-1)

        return p_logits + p_sigma + p_means

    def sample_data(self, n_samples: int = 1000):
        with torch.no_grad():
            sigmas = (0.5 * self.log_vars[0]).exp()
            betas = self.stick_break(self.logits[0])
            mus = self.means[0]

            zs = rng.choice(self.n_components, p=betas.numpy(), size=n_samples)
            xs = torch.normal(mus[zs], sigmas[zs])

        return xs


class PhiFour(Distribution):
    """
    Borrowed from oficial implementation: https://github.com/marylou-gabrie/flonaco
    Original paper: https://arxiv.org/pdf/2105.12603.pdf
    """

    def __init__(
        self,
        a,
        b,
        dim_grid,
        dim_phys=1,
        beta=1,
        bc=("dirichlet", 0),
        tilt=None,
        dtype=torch.float32,
        device="cpu",
    ):
        """
        Class to handle operations around Stochastic Allen-Cahn model
        Args:
            a: coupling term coef
            b: local field coef
            dim_grid: grid size along one physical dimension
            dim_phys: number of dimensions of the physical grid
            beta: inverse temperature
            tilt: None or {"val":0.7, "lambda":1} - mean value + Lagrange param
        """
        super().__init__(device=device)
        self.device = device

        self.a = a
        self.b = b
        self.beta = beta
        self.dim_grid = dim_grid
        self.dim_phys = dim_phys
        self.sum_dims = tuple(i + 1 for i in range(dim_phys))

        self.bc = bc
        self.tilt = tilt

    def init_field(self, n_or_values):
        if isinstance(n_or_values, int):
            x = torch.rand((n_or_values,) + (self.dim_grid,) * self.dim_phys)
            x = x * 2 - 1
        else:
            x = n_or_values
        return x

    def V(self, x):
        coef = self.a * self.dim_grid
        V = ((1 - x**2) ** 2 / 4 + self.b * x).sum(self.sum_dims) / coef
        if self.tilt is not None:
            tilt = (self.tilt["val"] - x.mean(self.sum_dims)) ** 2
            tilt = self.tilt["lambda"] * tilt / (4 * self.dim_grid)
            V += tilt
        return V

    def log_prob(self, x):
        # Does not include the temperature! need to be explicitely added in Gibbs factor

        if self.bc[0] == "dirichlet":
            x_ = F.pad(
                input=x,
                pad=(1,) * (2 * self.dim_phys),
                mode="constant",
                value=self.bc[1],
            )
        else:
            raise NotImplementedError("Only dirichlet BC implemeted")

        grad_term = ((x_[:, 1:, ...] - x_[:, :-1, ...]) ** 2 / 2).sum(
            self.sum_dims,
        )
        if self.dim_phys == 2:
            grad_term += ((x_[:, :, 1:] - x_[:, :, :-1]) ** 2 / 2).sum(
                self.sum_dims,
            )

        coef = self.a * self.dim_grid
        return -(grad_term * coef + self.V(x))

    def U_coupling_per_site(self, x):
        """
        return the (\nabla phi) ** 2 to be used in direct computation
        """
        assert self.dim_phys == 1
        if self.bc[0] == "dirichlet":
            x_ = F.pad(
                input=x,
                pad=(1,) * (2 * self.dim_phys),
                mode="constant",
                value=self.bc[1],
            )
        else:
            raise NotImplementedError("Only dirichlet BC implemeted")

        return ((x_[:, 1:] - x_[:, :-1]) ** 2 / 2) * self.a * self.dim_grid
