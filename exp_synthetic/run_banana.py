import gc
import pickle
import sys
import time
import tkinter

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ot
import pyro
import torch
import tqdm
from cifar10_exp.models import Discriminator, Generator
from IPython.display import clear_output
from pyro.infer import HMC, MCMC, NUTS
from torch import nn, optim

from ex2mcmc.samplers import ex2_mcmc, i_sir, mala
from ex2mcmc.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from ex2mcmc.sampling_utils.adaptive_sir_loss import MixKLLoss
from ex2mcmc.sampling_utils.distributions import (
    Banana,
    CauchyMixture,
    Distribution,
    Funnel,
    HalfBanana,
    IndependentNormal,
)
from ex2mcmc.sampling_utils.ebm_sampling import MALA
from ex2mcmc.sampling_utils.flows import RNVP
from ex2mcmc.sampling_utils.metrics import ESS, acl_spectrum
from ex2mcmc.sampling_utils.total_variation import average_total_variation


def sample_nuts(
    target,
    proposal,
    device="cpu",
    num_samples=1000,
    burn_in=1000,
    batch_size=1,
    rand_seed=42,
):
    def true_target_energy(z):
        return -target(z)

    def energy(z):
        z = z["points"]
        return true_target_energy(z).sum()

    start_time = time.time()
    # kernel = HMC(potential_fn=energy, step_size = 0.1, num_steps = K, full_mass = False)
    kernel_true = NUTS(potential_fn=energy, full_mass=False)
    # kernel_true = HMC(potential_fn=energy, full_mass=False)
    pyro.set_rng_seed(rand_seed)
    init_samples = proposal.sample((batch_size,)).to(device)
    print(init_samples.shape)
    # init_samples = torch.zeros_like(init_samples)
    dim = init_samples.shape[-1]
    init_params = {"points": init_samples}
    mcmc_true = MCMC(
        kernel=kernel_true,
        num_samples=num_samples,
        initial_params=init_params,
        warmup_steps=burn_in,
    )
    mcmc_true.run()
    q_true = mcmc_true.get_samples(group_by_chain=True)["points"].cpu()
    samples_true = np.array(q_true.view(-1, batch_size, dim))
    end_time = time.time()
    return end_time - start_time, samples_true


def compute_metrics(
    xs_true,
    xs_pred,
    name=None,
    n_samples=1000,
    scale=1.0,
    trunc_chain_len=None,
    ess_rar=1,
):
    metrics = dict()
    key = jax.random.PRNGKey(0)
    n_steps = 25
    # n_samples = 100

    ess = ESS(
        acl_spectrum(
            xs_pred[::ess_rar] - xs_pred[::ess_rar].mean(0)[None, ...],
        ),
    ).mean()
    metrics["ess"] = ess

    xs_pred = xs_pred[-trunc_chain_len:]

    tracker = average_total_variation(
        key,
        xs_true,
        xs_pred,
        n_steps=n_steps,
        n_samples=n_samples,
    )

    metrics["tv_mean"] = tracker.mean()
    metrics["tv_conf_sigma"] = tracker.std_of_mean()

    mean = tracker.mean()
    std = tracker.std()

    metrics["emd"] = 0
    # Cost_matr_isir = ot.dist(x1 = isir_res[j][i], x2=gt_samples[i], metric='sqeuclidean', p=2, w=None)
    for b in range(xs_pred.shape[1]):
        M = ot.dist(xs_true / scale, xs_pred[:, b, :] / scale)
        emd = ot.lp.emd2([], [], M, numItermax=1e6)
        metrics["emd"] += emd / xs_pred.shape[1]

    if name is not None:
        print(f"===={name}====")
    print(
        f"TV distance. Mean: {mean:.3f}, Std: {std:.3f}. \nESS: {ess:.3f} \nEMD: {emd:.3f}",
    )

    return metrics


# begin script
dims = [20, 40, 60, 80, 100]
step_size = [0.2, 0.1, 5e-2, 5e-2, 5e-2]
# n_steps_training = [200,200,200,400,400]
num_replications = 20
device = "cuda"

res_nuts = {"time": [], "ess": [], "emd": [], "tv": []}
res_ex2 = {"time": [], "ess": [], "emd": [], "tv": []}
res_mala = {"time": [], "ess": [], "emd": [], "tv": []}
# res_isir = {"time":[],"ess":[],"emd":[],"tv":[]}
res_adaptive_isir = {"time": [], "ess": [], "emd": [], "tv": []}
res_flex = {"time": [], "ess": [], "emd": [], "tv": []}


for i in range(num_replications):
    for j in range(len(dims)):
        dim = dims[j]
        # initialize distribution params
        scale_proposal = 1.0
        scale_isir = 5.0
        dist_class = "Banana"
        dist_params = {
            "b": 0.02,
            "sigma": 5.0,
        }
        sigma = 5.0
        b = 0.02

        target = Banana(
            dim=dim,
            device=device,
            b=b,
            sigma=sigma,
            # b = b
            # **dist_params.dict,
        )

        loc_proposal = torch.zeros(dim).to(device)
        scale_proposal = scale_proposal * torch.ones(dim).to(device)
        scale_isir = scale_isir * torch.ones(dim).to(device)

        proposal = IndependentNormal(
            dim=dim,
            loc=loc_proposal,
            scale=scale_proposal,
            device=device,
        )

        proposal_ex2 = IndependentNormal(
            dim=dim,
            loc=loc_proposal,
            scale=scale_isir,
            device=device,
        )
        # generate ground-truth samples
        N_samples = 5 * 10**3
        np.random.seed(42)
        True_samples = np.random.randn(N_samples, dim)
        for k in range(dim):
            if k % 2 == 0:
                True_samples[:, k] *= sigma
            else:
                True_samples[:, k] += b * True_samples[:, k - 1] ** 2 - (sigma**2) * b
        # sample NUTS
        # samples to compute ground-truth metrics
        Nuts_samples_ground_truth = 2000
        # Nuts_samples_comparison
        trunc_chain_len = 1000
        # nuts samples burn_in
        nuts_burn_in = 500
        # nuts batch size
        nuts_batch = 1
        rand_seed = 42 + i
        batch_size = 1
        time_cur, sample_nuts_ref = sample_nuts(
            target,
            proposal,
            device,
            num_samples=trunc_chain_len,
            batch_size=nuts_batch,
            burn_in=nuts_burn_in,
            rand_seed=rand_seed,
        )
        res_nuts["time"].append(time_cur)
        metrics = compute_metrics(
            True_samples,
            sample_nuts_ref,
            name="NUTS",
            trunc_chain_len=trunc_chain_len,
            ess_rar=1,
        )
        res_nuts["ess"].append(metrics["ess"])
        res_nuts["emd"].append(metrics["emd"])
        res_nuts["tv"].append(metrics["tv_mean"])
        # sample MALA
        params = {
            "N": 1,
            "grad_step": step_size[j],
            "adapt_stepsize": True,  # True
            "corr_coef": 0.0,
            "bernoulli_prob_corr": 0.0,  # 0.75
            "mala_steps": 5,
        }
        n_steps_ex2 = 1000
        batch_size = 1
        mcmc = Ex2MCMC(**params, dim=dim)
        pyro.set_rng_seed(rand_seed)
        start = proposal_ex2.sample((batch_size,)).to(device)
        start_time = time.time()
        out = mcmc(start, target, proposal_ex2, n_steps=n_steps_ex2)
        if isinstance(out, tuple):
            sample = out[0]
        else:
            sample = out
        sample = np.array(
            [_.detach().numpy() for _ in sample],
        ).reshape(-1, batch_size, dim)
        end_time = time.time()
        res_mala["time"].append(end_time - start_time)
        metrics = compute_metrics(
            True_samples,
            sample,
            name="MALA",
            trunc_chain_len=trunc_chain_len,
            ess_rar=1,
        )
        res_mala["ess"].append(metrics["ess"])
        res_mala["emd"].append(metrics["emd"])
        res_mala["tv"].append(metrics["tv_mean"])
        # sample Ex2-MCMC
        params = {
            "N": 200,
            "grad_step": step_size[j],
            "adapt_stepsize": True,  # True
            "corr_coef": 0.0,
            "bernoulli_prob_corr": 0.0,  # 0.75
            "mala_steps": 5,
        }
        n_steps_ex2 = 1000
        batch_size = 1
        mcmc = Ex2MCMC(**params, dim=dim)
        pyro.set_rng_seed(rand_seed)
        start = proposal_ex2.sample((batch_size,)).to(device)
        start_time = time.time()
        out = mcmc(start, target, proposal_ex2, n_steps=n_steps_ex2)
        if isinstance(out, tuple):
            sample = out[0]
        else:
            sample = out
        sample = np.array(
            [_.detach().numpy() for _ in sample],
        ).reshape(-1, batch_size, dim)
        end_time = time.time()
        res_ex2["time"].append(end_time - start_time)
        metrics = compute_metrics(
            True_samples,
            sample,
            name="Ex2MCMC",
            trunc_chain_len=trunc_chain_len,
            ess_rar=1,
        )
        res_ex2["ess"].append(metrics["ess"])
        res_ex2["emd"].append(metrics["emd"])
        res_ex2["tv"].append(metrics["tv_mean"])
        # sample Flex2MCMC
        params_flex = {
            "N": 200,
            "grad_step": step_size[j],
            "adapt_stepsize": True,
            "corr_coef": 0.0,
            "bernoulli_prob_corr": 0.0,
            "mala_steps": 0,
            "flow": {
                "num_flows": 4,  # number of normalizing layers
                "lr": 1e-3,  # learning rate
                "batch_size": 100,
                "n_steps": 1000,
            },
        }
        pyro.set_rng_seed(rand_seed)
        start_time = time.time()
        mcmc = Ex2MCMC(**params_flex, dim=dim)
        verbose = mcmc.verbose
        mcmc.verbose = False
        flow = RNVP(params_flex["flow"]["num_flows"], dim=dim, device=device)
        flow_mcmc = FlowMCMC(
            target,
            proposal,
            device,
            flow,
            mcmc,
            batch_size=params_flex["flow"]["batch_size"],
            lr=params_flex["flow"]["lr"],
        )
        flow.train()
        out_samples, nll = flow_mcmc.train(
            n_steps=params_flex["flow"]["n_steps"],
        )
        assert not torch.isnan(
            next(flow.parameters())[0, 0],
        ).item()
        gc.collect()
        torch.cuda.empty_cache()
        flow.eval()
        mcmc.flow = flow
        mcmc.verbose = verbose
        end_train_time = time.time()
        # sample adaptive i-sir
        n_steps_flex2 = 1000
        pyro.set_rng_seed(rand_seed)
        mcmc.mala_steps = 0
        start = proposal.sample((batch_size,))
        # s = time.time()
        out = mcmc(start, target, proposal, n_steps=n_steps_flex2)
        if isinstance(out, tuple):
            sample = out[0]
        else:
            sample = out
        sample = np.array(
            [_.detach().numpy() for _ in sample],
        ).reshape(-1, batch_size, dim)
        sample_flex2_new = sample
        end_time = time.time()
        res_adaptive_isir["time"].append(end_time - start_time)
        metrics = compute_metrics(
            True_samples,
            sample_flex2_new,
            name="Flex2",
            trunc_chain_len=trunc_chain_len,
            ess_rar=1,
        )
        res_adaptive_isir["ess"].append(metrics["ess"])
        res_adaptive_isir["emd"].append(metrics["emd"])
        res_adaptive_isir["tv"].append(metrics["tv_mean"])
        # sample Flex2
        n_steps_flex2 = 1000
        pyro.set_rng_seed(rand_seed)
        mcmc.mala_steps = 5
        start = proposal.sample((batch_size,))
        # s = time.time()
        out = mcmc(start, target, proposal, n_steps=n_steps_flex2)
        if isinstance(out, tuple):
            sample = out[0]
        else:
            sample = out
        sample = np.array(
            [_.detach().numpy() for _ in sample],
        ).reshape(-1, batch_size, dim)
        sample_flex2_new = sample
        end_flex_time = time.time()
        res_flex["time"].append(end_train_time - start_time + end_flex_time - end_time)
        metrics = compute_metrics(
            True_samples,
            sample_flex2_new,
            name="Flex2",
            trunc_chain_len=trunc_chain_len,
            ess_rar=1,
        )
        res_flex["ess"].append(metrics["ess"])
        res_flex["emd"].append(metrics["emd"])
        res_flex["tv"].append(metrics["tv_mean"])
        del mcmc.flow
        gc.collect()
        torch.cuda.empty_cache()
        with open("./banana/res_nuts.pickle", "wb") as handle:
            pickle.dump(res_nuts, handle)
        with open("./banana/res_mala.pickle", "wb") as handle:
            pickle.dump(res_mala, handle)
        # with open('res_isir.pickle', 'wb') as handle:
        #    pickle.dump(res_isir, handle)
        with open("./banana/res_ex2.pickle", "wb") as handle:
            pickle.dump(res_ex2, handle)
        with open("./banana/adaptive_isir.pickle", "wb") as handle:
            pickle.dump(res_adaptive_isir, handle)
        with open("./banana/res_flex.pickle", "wb") as handle:
            pickle.dump(res_flex, handle)
