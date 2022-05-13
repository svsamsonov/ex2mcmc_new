import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from distributions import Gaussian_mixture
from general_utils import DotDict
from scipy import linalg
from sklearn.mixture import GaussianMixture


dim = 2
device = torch.device("cpu")

target_args = DotDict()
target_args.device = device
target_args.num_gauss = 25
n_col = 5
n_row = target_args.num_gauss // n_col
s = 1
u = torch.ones(dim)
# create points
sigma = 0.05
coef_gaussian = 1.0 / target_args.num_gauss
target_args.p_gaussians = [torch.tensor(coef_gaussian)] * target_args.num_gauss
locs = [
    torch.tensor([(i - 2) * s, (j - 2) * s] + [0] * (dim - 2)).to(device)
    for i in range(n_col)
    for j in range(n_row)
]
target_args.locs = locs
target_args.covs = [
    (sigma ** 2) * torch.eye(dim).to(device),
] * target_args.num_gauss
target_args.dim = dim
target = Gaussian_mixture(target_args).log_prob


n_components = 25
std = 0.05

weights_ = np.array([1.0 / n_components] * n_components)
means_init = np.array(
    list(itertools.product(np.arange(-2, 3), repeat=2)),
    dtype=np.float32,
)
covs_init = np.array([np.eye(2) * (std ** 2)] * n_components)
# precisions_init = np.array([np.ones((2, 2))*(std ** 2)] * n_components)
precisions_init = np.array(
    [np.linalg.inv(np.eye(2) * (std ** 2))] * n_components,
)

gm = GaussianMixture(
    n_components=n_components,
    covariance_type="full",
    weights_init=weights_,
    means_init=means_init,
    precisions_init=precisions_init,
    random_state=42,
)
gm.weights_ = weights_
gm.means_ = means_init
gm.covariances_ = covs_init
gm.precisions_cholesky_ = np.array(
    [
        linalg.cholesky(prec_init, lower=True)
        for prec_init in gm.precisions_init
    ],
)

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
batch_size = 3
point = torch.randn(batch_size, dim, device=device).data.cpu().numpy()
print(point)
pdfs = np.exp(gm.score_samples(point))
print(pdfs)


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
point = torch.randn(batch_size, dim, device=device)
density = Gaussian_mixture(target_args).get_density
print(point)
print(density(point))
