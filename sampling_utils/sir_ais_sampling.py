import random

import numpy as np
import sklearn
import torch

from . import ebm_sampling
from .distributions import GaussianMixture, init_independent_normal
from .general_utils import DotDict
from .metrics import Evolution


def compute_sir_log_weights(x, target, proposal):
    return target(x) - proposal.log_prob(x)


def sir_independent_dynamics(z, target, proposal, n_steps, N):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    for _ in range(n_steps):
        z_sp.append(z)
        ind = torch.randint(0, N, (batch_size,)).tolist()
        X = proposal.sample([batch_size, N])
        X[np.arange(batch_size), ind, :] = z
        X_view = X.view(-1, z_dim)

        log_weight = compute_sir_log_weights(X_view, target, proposal)
        log_weight = log_weight.view(batch_size, N)

        max_logs = torch.max(log_weight, dim=1)[0][:, None]
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim=1)
        weight = weight / sum_weight[:, None]

        weight[weight != weight] = 0.0
        weight[weight.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weight, 1).squeeze().tolist()

        z = X[np.arange(batch_size), indices, :]
        z = z.data

    z_sp.append(z)
    return z_sp


sir_independent_sampling = ebm_sampling.sampling_from_dynamics(
    sir_independent_dynamics,
)


def sir_correlated_dynamics(z, target, proposal, n_steps, N, alpha=0.0):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    for _ in range(n_steps):
        z_sp.append(z)
        z_copy = z.unsqueeze(1).repeat(1, N, 1)
        ind = torch.randint(0, N, (batch_size,)).tolist()
        W = proposal.sample([batch_size, N])
        U = proposal.sample([batch_size]).unsqueeze(1).repeat(1, N, 1)
        X = (
            (alpha ** 2) * z_copy
            + alpha * ((1 - alpha ** 2) ** 0.5) * U
            + W * ((1 - alpha ** 2) ** 0.5)
        )
        X[np.arange(batch_size), ind, :] = z
        X_view = X.view(-1, z_dim)

        log_weight = compute_sir_log_weights(X_view, target, proposal)
        log_weight = log_weight.view(batch_size, N)
        max_logs = torch.max(log_weight, dim=1)[0][:, None]
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim=1)
        weight = weight / sum_weight[:, None]

        weight[weight != weight] = 0.0
        weight[weight.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weight, 1).squeeze().tolist()

        z = X[np.arange(batch_size), indices, :]
        z = z.data

    z_sp.append(z)
    return z_sp


sir_correlated_sampling = ebm_sampling.sampling_from_dynamics(
    sir_correlated_dynamics,
)


def run_experiments_gaussians(
    dim_arr,
    scale_proposal,
    scale_target,
    loc_target,
    num_points_in_chain,
    strategy_mean,
    device,
    batch_size,
    method_params,
    random_seed=42,
    loc_proposal=0.0,
    mode_init="proposal",
    method="sir_independent",
    print_results=True,
):
    dict_results = {
        mode_init: {
            "mean_loc": [],
            "mean_var": [],
            "ess": [],
            "history_first": [],
            "history_norm": [],
            "acceptence": [],
        },
    }

    if print_results:
        print("------------------")
        print(f"mode = {mode_init}")

    for dim in dim_arr:
        if print_results:
            print(f"dim = {dim}")
        target = init_independent_normal(scale_target, dim, device, loc_target)
        proposal = init_independent_normal(
            scale_proposal,
            dim,
            device,
            loc_proposal,
        )
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        if (mode_init == "target") and not method.startswith("citer"):
            start = target.sample([batch_size])
        elif mode_init == "proposal" and not method.startswith("citer"):
            start = proposal.sample([batch_size])
        elif (mode_init == "target") and method.startswith("citer"):
            start = target.sample([batch_size, len(method_params["betas"])])
        elif mode_init == "proposal" and method.startswith("citer"):
            start = proposal.sample([batch_size, len(method_params["betas"])])
        else:
            raise ValueError("Unknown initialization method")
        if method == "sir_correlated":
            alpha = (1 - method_params["c"] / dim) ** 0.5
            history = sir_correlated_dynamics(
                start,
                target,
                proposal,
                method_params["n_steps"],
                method_params["N"],
                alpha,
            )
            acceptence = 1.0
        elif method == "sir_independent":
            history = sir_independent_dynamics(
                start,
                target,
                proposal,
                method_params["n_steps"],
                method_params["N"],
            )
            acceptence = 1.0
        elif method == "citerais_mala":
            (history, acceptence) = ebm_sampling.citerais_mala_dynamics(
                start,
                target.log_prob,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
                method_params["rhos"],
            )

        elif method == "citerais_ula":
            (history, acceptence, _) = ebm_sampling.citerais_ula_dynamics(
                start,
                target.log_prob,
                proposal,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
                method_params["rhos"],
            )
        # history = [x[:, -1, :] for x in history]

        elif method == "i_ais_z":
            (history, acceptence) = ebm_sampling.i_ais_z_dynamics(
                start,
                target.log_prob,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
            )

        elif method == "i_ais_v":
            (history, acceptence) = ebm_sampling.i_ais_v_dynamics(
                start,
                target.log_prob,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
                method_params["rho"],
            )

        elif method == "i_ais_b":
            (history, acceptence) = ebm_sampling.i_ais_b_dynamics(
                start,
                target.log_prob,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
                method_params["rho"],
            )
        else:
            raise ValueError("Unknown sampling method")
        last_history = history[
            max(1, len(history) - num_points_in_chain - 1) :
        ]
        all_history_np = torch.stack(history, dim=0).cpu().numpy()

        result_np = torch.stack(last_history, dim=0).cpu().numpy()
        if strategy_mean == "starts":
            result_var = np.var(result_np, axis=1, ddof=1).mean(axis=0).mean()
            result_mean = np.mean(result_np, axis=1).mean(axis=0).mean()

        elif strategy_mean == "chain":
            result_var = np.var(result_np, axis=0, ddof=1).mean(axis=0).mean()
            result_mean = np.mean(result_np, axis=0).mean(axis=0).mean()

        else:
            raise ValueError("Unknown method of mean")

        # print(result_np.shape)
        result_np_1 = result_np[:-1]
        result_np_2 = result_np[1:]
        diff = (result_np_1 == result_np_2).sum(axis=2)
        # print(diff)
        ess_bs = (diff != dim).mean(axis=0)
        ess = ess_bs.mean()
        first_coord_history = all_history_np[:, :, 0]
        norm_history = np.linalg.norm(all_history_np, axis=-1)

        if print_results:
            print(f"mean estimation of acceptence rate = {acceptence}")
            print(f"mean estimation of variance = {result_var}")
            print(f"mean estimation of mean = {result_mean}")
            print(f"mean estimation of ess = {ess}")
            print("------")
        dict_results[mode_init]["acceptence"].append(acceptence)
        dict_results[mode_init]["mean_loc"].append(result_mean)
        dict_results[mode_init]["mean_var"].append(result_var)
        dict_results[mode_init]["ess"].append(ess)
        dict_results[mode_init]["history_first"].append(first_coord_history)
        dict_results[mode_init]["history_norm"].append(norm_history)

    return dict_results


def run_experiments_2_gaussians(
    dim_arr,
    scale_proposal,
    scale_target,
    loc_1_target,
    loc_2_target,
    num_points_in_chain,
    strategy_mean,
    device,
    batch_size,
    method_params,
    random_seed=42,
    loc_proposal=0.0,
    mode_init="proposal",
    method="sir_independent",
    print_results=True,
):
    dict_results = {
        mode_init: {
            "mean_loc_1": [],
            "mean_loc_2": [],
            "mean_var": [],
            "mean_jsd": [],
            "mean_hqr": [],
            "ess": [],
            "history_first": [],
            "history_norm": [],
            "acceptence": [],
        },
    }

    if print_results:
        print("------------------")
        print(f"mode = {mode_init}")

    for dim in dim_arr:
        if print_results:
            print(f"dim = {dim}")

        target_args = DotDict()
        target_args.device = device
        target_args.num_gauss = 2

        coef_gaussian = 1.0 / target_args.num_gauss
        target_args.p_gaussians = [
            torch.tensor(coef_gaussian),
        ] * target_args.num_gauss
        locs = [
            loc_1_target * torch.ones(dim, dtype=torch.float64).to(device),
            loc_2_target * torch.ones(dim, dtype=torch.float64).to(device),
        ]
        locs_numpy = torch.stack(locs, dim=0).cpu().numpy()
        target_args.locs = locs
        target_args.covs = [
            (scale_target ** 2)
            * torch.eye(dim, dtype=torch.float64).to(device),
        ] * target_args.num_gauss
        target_args.dim = dim
        target = GaussianMixture(**target_args)
        proposal = init_independent_normal(
            scale_proposal,
            dim,
            device,
            loc_proposal,
        )
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if mode_init == "target":
            dataset = sklearn.datasets.make_blobs(
                n_samples=batch_size,
                n_features=dim,
                centers=locs_numpy,
                cluster_std=scale_target,
                random_state=random_seed,
            )[0]
            start = torch.FloatTensor(dataset).to(device)
        elif mode_init == "proposal" and not method.startswith("citerais"):
            start = proposal.sample([batch_size])
        elif mode_init == "proposal" and method.startswith("citerais"):
            start = proposal.sample([batch_size, len(method_params["betas"])])
            start = start.float()
        else:
            raise ValueError("Unknown initialization method")
        if method == "sir_correlated":
            alpha = (1 - method_params["c"] / dim) ** 0.5
            history = sir_correlated_dynamics(
                start,
                target,
                proposal,
                method_params["n_steps"],
                method_params["N"],
                alpha,
            )
            acceptence = 1.0
        elif method == "sir_independent":
            history = sir_independent_dynamics(
                start,
                target,
                proposal,
                method_params["n_steps"],
                method_params["N"],
            )
            acceptence = 1.0
        elif method == "citerais_mala":
            (history, acceptence) = ebm_sampling.citerais_mala_dynamics(
                start,
                target.log_prob,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
                method_params["rhos"],
            )

        elif method == "citerais_ula":
            (history, acceptence, _) = ebm_sampling.citerais_ula_dynamics(
                start,
                target.log_prob,
                proposal,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
                method_params["rhos"],
            )
            # history = [x[:, -1, :] for x in history]

        elif method == "i_ais_z":
            (history, acceptence) = ebm_sampling.i_ais_z_dynamics(
                start,
                target.log_prob,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
            )

        elif method == "i_ais_v":
            (history, acceptence) = ebm_sampling.i_ais_v_dynamics(
                start,
                target.log_prob,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
                method_params["rho"],
            )

        elif method == "i_ais_b":
            (history, acceptence) = ebm_sampling.i_ais_b_dynamics(
                start,
                target.log_prob,
                method_params["n_steps"],
                method_params["grad_step"],
                method_params["eps_scale"],
                method_params["N"],
                method_params["betas"],
                method_params["rho"],
            )

        else:
            raise ValueError("Unknown sampling method")
        last_history = history[(-num_points_in_chain - 1) : -1]
        all_history_np = torch.stack(history, dim=0).cpu().numpy()
        torch_last_history = torch.stack(last_history, dim=0).cpu()

        evolution = Evolution(
            None,
            locs=torch.stack(locs, 0).cpu(),
            sigma=scale_target,
        )

        result_np = torch.stack(last_history, dim=0).cpu().numpy()

        modes_var_arr = []
        modes_mean_arr = []
        h_q_r_arr = []
        jsd_arr = []
        means_est_1 = torch.zeros(dim)
        means_est_2 = torch.zeros(dim)
        num_found_1_mode = 0
        num_found_2_mode = 0

        if strategy_mean == "starts":
            for i in range(num_points_in_chain):
                X_gen = torch_last_history[i, :, :]
                assignment = Evolution.make_assignment(
                    X_gen,
                    evolution.locs,
                    evolution.sigma,
                )
                mode_var = (
                    Evolution.compute_mode_std(X_gen, assignment).item() ** 2
                )
                modes_mean, found_modes_ind = Evolution.compute_mode_mean(
                    X_gen,
                    assignment,
                )
                if 0 in found_modes_ind:
                    num_found_1_mode += 1
                    means_est_1 += modes_mean[0]
                if 1 in found_modes_ind:
                    num_found_2_mode += 1
                    means_est_2 += modes_mean[1]

                h_q_r = Evolution.compute_high_quality_rate(assignment).item()
                jsd = Evolution.compute_jsd(assignment).item()

                modes_var_arr.append(mode_var)
                modes_mean_arr.append(modes_mean)
                h_q_r_arr.append(h_q_r)
                jsd_arr.append(jsd)

        elif strategy_mean == "chain":
            for i in range(batch_size):
                X_gen = torch_last_history[:, i, :]
                assignment = Evolution.make_assignment(
                    X_gen,
                    evolution.locs,
                    evolution.sigma,
                )
                mode_var = (
                    Evolution.compute_mode_std(X_gen, assignment).item() ** 2
                )

                modes_mean, found_modes_ind = Evolution.compute_mode_mean(
                    X_gen,
                    assignment,
                )
                # print(found_modes_ind)
                if 0 in found_modes_ind:
                    num_found_1_mode += 1
                    means_est_1 += modes_mean[0]
                if 1 in found_modes_ind:
                    num_found_2_mode += 1
                    means_est_2 += modes_mean[1]
                h_q_r = Evolution.compute_high_quality_rate(assignment).item()
                jsd = Evolution.compute_jsd(assignment).item()

                modes_var_arr.append(mode_var)

                h_q_r_arr.append(h_q_r)
                jsd_arr.append(jsd)

        else:
            raise ValueError("Unknown method of mean")

        # print(modes_var_arr)
        jsd_result = np.array(jsd_arr).mean()
        modes_var_result = np.array(modes_var_arr).mean()
        h_q_r_result = np.array(h_q_r_arr).mean()
        if num_found_1_mode == 0:
            print(
                "Unfortunalely, no points were assigned to 1st mode, default estimation - zero",
            )
            modes_mean_1_result = np.nan  # 0.0
        else:
            modes_mean_1_result = (
                (means_est_1 / num_found_1_mode).mean().item()
            )
        if num_found_2_mode == 0:
            print(
                "Unfortunalely, no points were assigned to 2nd mode, default estimation - zero",
            )
            modes_mean_2_result = np.nan  # 0.0
        else:
            modes_mean_2_result = (
                (means_est_2 / num_found_2_mode).mean().item()
            )
        if num_found_1_mode == 0 and num_found_2_mode == 0:
            modes_mean_1_result = (
                modes_mean_2_result
            ) = torch_last_history.mean().item()

        result_np_1 = result_np[:-1]
        result_np_2 = result_np[1:]
        diff = (result_np_1 == result_np_2).sum(axis=2)
        ess_bs = (diff != dim).mean(axis=0)
        ess = ess_bs.mean()
        first_coord_history = all_history_np[:, :, 0]
        norm_history = np.linalg.norm(all_history_np, axis=-1)

        if print_results:
            print(f"mean estimation of acceptence rate = {acceptence}")
            print(f"mean estimation of target variance = {modes_var_result}")
            print(f"mean estimation of 1 mode mean  = {modes_mean_1_result}")
            print(f"mean estimation of 2 mode mean  = {modes_mean_2_result}")
            print(f"mean estimation of JSD  = {jsd_result}")
            print(f"mean estimation of HQR  = {h_q_r_result}")
            print(f"mean estimation of ESS = {ess}")
            print("------")
        dict_results[mode_init]["acceptence"].append(acceptence)
        dict_results[mode_init]["mean_loc_1"].append(modes_mean_1_result)
        dict_results[mode_init]["mean_loc_2"].append(modes_mean_2_result)
        dict_results[mode_init]["mean_var"].append(modes_var_result)
        dict_results[mode_init]["mean_jsd"].append(jsd_result)
        dict_results[mode_init]["mean_hqr"].append(h_q_r_result)
        dict_results[mode_init]["ess"].append(ess)
        dict_results[mode_init]["history_first"].append(first_coord_history)
        dict_results[mode_init]["history_norm"].append(norm_history)

    return dict_results
