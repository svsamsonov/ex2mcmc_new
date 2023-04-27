import os

import numpy as np
import torch
from torch.distributions import Uniform
from tqdm import tqdm, trange

from .distributions import IndependentNormal, init_independent_normal
from .mcmc_base import AbstractMCMC, adapt_stepsize_dec, increment_steps


def grad_energy(point, target, x=None):
    point = point.detach().requires_grad_()
    if x is not None:
        energy = -target.log_prob(z=point, x=x)
    else:
        energy = -target.log_prob(point)

    grad = torch.autograd.grad(energy.sum(), point)[0]
    return energy, grad


def sampling_f(
    dynamics,
    target,
    proposal,
    batch_size,
    n,
    path_to_save=None,
    file_name=None,
    every_step=None,
    continue_z=None,
    *args,
    **kwargs,
):
    z_last = []
    zs = []
    z = None
    for i in tqdm(range(0, n, batch_size)):
        if continue_z is None:
            z = proposal.sample([batch_size])
        else:
            j = i // batch_size
            z = continue_z[j][-1].clone().detach()
        z.requires_grad_(True)
        out = dynamics(z, target, proposal, *args, **kwargs)
        if isinstance(out, tuple):
            z_sp = out[0]
        else:
            z_sp = out
        last = z_sp[-1].data.cpu().numpy()
        zs_append = np.stack([o.data.cpu().numpy() for o in z_sp], axis=0)
        z_last.append(last)
        zs.append(zs_append)
        if (
            (file_name is not None)
            and (path_to_save is not None)
            and (every_step is not None)
        ):
            cur_file_name = file_name + f"_batch_num_{i}.npy"
            cur_path_to_save = os.path.join(path_to_save, cur_file_name)
            save_np_file = zs_append[::every_step, :, :]
            np.save(cur_path_to_save, save_np_file)
            print(
                f"file {cur_path_to_save} was saved, file shape = {save_np_file.shape}",
            )

    z_last_np = np.asarray(z_last).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return z_last_np, zs


def load_data_from_batches(n, batch_size, path_to_save, path_to_batches):
    load_np = []

    for i in tqdm(range(0, n, batch_size)):
        cur_file_name = path_to_batches + f"_batch_num_{i}.npy"
        cur_path_to_save = os.path.join(path_to_save, cur_file_name)
        cur_zs = np.load(cur_path_to_save)
        load_np.append(cur_zs)

    load_np = np.array(load_np)
    load_np = np.concatenate(
        load_np.transpose((0, 2, 1, 3)),
        axis=0,
    ).transpose((1, 0, 2))
    return load_np


def sampling_from_dynamics(dynamics):
    def sampling_dynamics_f(
        target,
        proposal,
        batch_size,
        n,
        path_to_save=None,
        file_name=None,
        every_step=None,
        continue_z=None,
        *args,
        **kwargs,
    ):
        return sampling_f(
            dynamics,
            target,
            proposal,
            batch_size,
            n,
            path_to_save,
            file_name,
            every_step,
            continue_z,
            *args,
            **kwargs,
        )

    return sampling_dynamics_f


def aggregate_sampling_output(z):
    return np.concatenate(z.transpose((0, 2, 1, 3)), axis=0).transpose(
        (1, 0, 2),
    )


def langevin_dynamics(
    z, target, proposal, n_steps, grad_step, eps_scale, verbose=False
):
    z_sp = []
    batch_size, _ = z.shape[0], z.shape[1]

    range_gen = trange if verbose else range

    for _ in range_gen(n_steps):
        z_sp.append(z)
        eps = eps_scale * proposal.sample([batch_size])

        E, grad = grad_energy(z, target, x=None)
        z = z - grad_step * grad + eps
        z = z.data
        z.requires_grad_(True)
    z_sp.append(z)
    return z_sp


langevin_sampling = sampling_from_dynamics(langevin_dynamics)


class ULA(AbstractMCMC):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.grad_step = kwargs.get("grad_step", 1e-2)
        self.noise_scale = kwargs.get(
            "noise_scale",
            (2 * self.grad_step) ** 0.5,
        )
        self.verbose = kwargs.get("verbose", True)

    def __call__(
        self,
        z,
        target,
        proposal,
        n_steps,
        grad_step=None,
        noise_scale=None,
        verbose=None,
    ):
        grad_step = grad_step if grad_step is not None else self.grad_step
        noise_scale = noise_scale if noise_scale is not None else self.noise_scale
        verbose = verbose if verbose is None else self.verbose
        return langevin_dynamics(
            z, target, proposal, n_steps, grad_step, noise_scale, verbose
        )


def heuristics_step_size(
    mean_acceptance,
    target_acceptance,
    stepsize,
    factor=1.05,
    tol=0.03,
):
    if mean_acceptance - target_acceptance > tol:
        return stepsize * factor
    if target_acceptance - mean_acceptance > tol:
        return stepsize / factor
    return stepsize


def mala_dynamics(
    z,
    target,
    proposal,
    n_steps,
    grad_step,
    eps_scale,
    acceptance_rule="Hastings",
    adapt_stepsize=False,
):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    device = z.device

    # !!
    std_norm = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(z_dim).to(device),
        torch.eye(z_dim).to(device),
    )

    if adapt_stepsize:
        eps_scale = (2 * grad_step) ** (1 / 2)
    uniform = Uniform(low=0.0, high=1.0)
    acceptence = torch.zeros(batch_size).to(device)

    for _ in range(n_steps):
        eps = eps_scale * std_norm.sample([batch_size])

        E, grad = grad_energy(z, target, x=None)

        new_z = z - grad_step * grad + eps
        new_z = new_z.data
        new_z.requires_grad_(True)

        E_new, grad_new = grad_energy(new_z, target, x=None)

        energy_part = E - E_new

        propose_vec_1 = z - new_z + grad_step * grad_new
        propose_vec_2 = new_z - z + grad_step * grad

        # propose_part_1 = proposal.log_prob(propose_vec_1/eps_scale)
        # propose_part_2 = proposal.log_prob(propose_vec_2/eps_scale)
        propose_part_1 = std_norm.log_prob(propose_vec_1 / eps_scale)
        propose_part_2 = std_norm.log_prob(propose_vec_2 / eps_scale)

        propose_part = propose_part_1 - propose_part_2

        log_accept_prob = torch.zeros_like(propose_part)

        if acceptance_rule == "Hastings":
            log_accept_prob = propose_part + energy_part

        elif acceptance_rule == "Barker":
            log_ratio = propose_part + energy_part
            log_accept_prob = -torch.log(1.0 + torch.exp(-log_ratio))

        generate_uniform_var = uniform.sample([batch_size]).to(z.device)
        log_generate_uniform_var = torch.log(generate_uniform_var)
        mask = log_generate_uniform_var < log_accept_prob

        # Adapting heuristics
        if adapt_stepsize:
            mean_acceptance = mask.float().mean()
            grad_step = heuristics_step_size(
                mean_acceptance,
                target_acceptance=0.45,
                stepsize=grad_step,
            )
            eps_scale = (2 * grad_step) ** (1 / 2)
        ########

        acceptence += mask
        with torch.no_grad():
            z[mask] = new_z[mask].detach().clone()
            z = z.data
            z.requires_grad_(True)
            z_sp.append(z.clone().detach())

    return z_sp, acceptence


mala_sampling = sampling_from_dynamics(mala_dynamics)


class MALA(AbstractMCMC):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim
        self.device = kwargs.get("device", "cpu")
        self.grad_step = kwargs.get("grad_step", 1e-2)
        self.noise_scale = kwargs.get(
            "noise_scale",
            (2 * self.grad_step) ** 0.5,
        )
        self.adapt_stepsize = kwargs.get("adapt_stepsize", False)
        self.mala_transition = MALATransition(dim, self.device)
        if self.adapt_stepsize:
            self.mala_transition.adapt_grad_step = self.grad_step
            self.mala_transition.adapt_sigma = self.noise_scale
        self.verbose = kwargs.get("verbose", True)
        self.beta = kwargs.get("beta", 1.0)  # inverse temperature

    # @increment_steps
    # @adapt_stepsize_dec
    def __call__(
        self,
        z,
        target,
        proposal,
        n_steps,
        grad_step=None,
        noise_scale=None,
        beta=None,
        adapt_stepsize=None,
        verbose=None,
    ):
        adapt_stepsize = (
            self.adapt_stepsize if adapt_stepsize is None else adapt_stepsize
        )
        grad_step = self.grad_step if grad_step is None else grad_step
        noise_scale = self.noise_scale if noise_scale is None else noise_scale
        verbose = self.verbose if verbose is None else verbose
        beta = self.beta if beta is None else beta

        zs = [z.clone().detach()]
        batch_size = z.shape[0]
        acceptance = torch.zeros(batch_size)

        range_gen = trange if verbose else range

        for _ in range_gen(n_steps):
            energy, grad = grad_energy(z, target)
            z, _, _, mask = self.mala_transition(
                z,
                energy,
                grad,
                grad_step=grad_step,
                sigma=noise_scale,
                beta=beta,
                target=target,
                adapt_stepsize=adapt_stepsize,
            )
            zs.append(z.detach().clone())
            acceptance += mask.float() / n_steps
        if adapt_stepsize:
            self.grad_step = self.mala_transition.adapt_grad_step

        return zs, acceptance


def mh_dynamics_normal_proposal(
    z,
    target,
    proposal,
    n_steps,
    eps_scale,
    acceptance_rule="Hastings",
):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    device = z.device

    # !!
    std_norm = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(z_dim).to(device),
        torch.eye(z_dim).to(device),
    )

    uniform = Uniform(low=0.0, high=1.0)
    acceptence = torch.zeros(batch_size).to(device)
    log_accept_prob = None

    for _ in range(n_steps):
        E = -target(z)

        eps = eps_scale * std_norm.sample([batch_size]).to(device)
        new_z = z + eps
        new_z = new_z.data

        E_new = -target(new_z)

        energy_part = E - E_new

        propose_vec_1 = z - new_z
        propose_vec_2 = new_z - z

        #    propose_part_1 = proposal.log_prob(propose_vec_1/eps_scale)
        #    propose_part_2 = proposal.log_prob(propose_vec_2/eps_scale)
        propose_part_1 = std_norm.log_prob(propose_vec_1 / eps_scale)
        propose_part_2 = std_norm.log_prob(propose_vec_2 / eps_scale)

        propose_part = propose_part_1 - propose_part_2

        if acceptance_rule == "Hastings":
            log_accept_prob = propose_part + energy_part

        elif acceptance_rule == "Barker":
            log_ratio = propose_part + energy_part
            log_accept_prob = -torch.log(1.0 + torch.exp(-log_ratio))

        generate_uniform_var = uniform.sample([batch_size]).to(z.device)
        log_generate_uniform_var = torch.log(generate_uniform_var)
        mask = log_generate_uniform_var < log_accept_prob

        acceptence += mask
        with torch.no_grad():
            z[mask] = new_z[mask].detach().clone()
            z = z.data
            z_sp.append(z.clone().detach())

    return z_sp, acceptence


mh_sampling_normal_proposal = sampling_from_dynamics(
    mh_dynamics_normal_proposal,
)


class MHKernel:
    @staticmethod
    def log_prob(
        log_pi1,
        log_pi2,
        log_transition_forward,
        log_transition_backward,
    ):
        return (log_pi2 + log_transition_backward) - (log_pi1 + log_transition_forward)

    @staticmethod
    def get_mask(
        log_pi1,
        log_pi2,
        log_transition_forward,
        log_transition_backward,
    ):
        batch_size = log_pi1.shape[0]
        device = log_pi1.device
        acc_log_prob = MHKernel.log_prob(
            log_pi1,
            log_pi2,
            log_transition_forward,
            log_transition_backward,
        )
        generate_uniform_var = torch.rand([batch_size], device=device)
        log_generate_uniform_var = torch.log(generate_uniform_var)
        mask = log_generate_uniform_var < acc_log_prob
        return mask


class MALATransition:
    def __init__(self, z_dim, device):
        self.device = device
        self.z_dim = z_dim

        loc = torch.zeros(z_dim).to(device)
        scale = torch.ones(z_dim).to(device)
        self.stand_normal = IndependentNormal(
            dim=z_dim,
            device=device,
            loc=loc,
            scale=scale,
        )

        self.adapt_grad_step = 0.0
        self.adapt_sigma = 0.0

    @staticmethod
    def get_mh_kernel_log_prob(
        log_pi1,
        log_pi2,
        log_transition_forward,
        log_transition_backward,
    ):
        return (log_pi2 + log_transition_backward) - (log_pi1 + log_transition_forward)

    def get_langevin_transition_kernel(
        self,
        z1,
        z2,
        grad,
        grad_step,
        sigma=None,
    ):
        if sigma is None:
            sigma = (2 * grad_step) ** 0.5
        loc = z2 - (z1 - grad_step * grad)
        log_prob = self.stand_normal.log_prob(loc / sigma)
        return log_prob

    def compute_log_probs(
        self,
        z,
        z_new,
        energy,
        grad,
        grad_step,
        sigma,
        target=None,
        beta=1.0,
    ):
        energy_new, grad_new = grad_energy(z_new, target, x=None)
        log_transition_forward = self.get_langevin_transition_kernel(
            z,
            z_new,
            grad,
            beta * grad_step,
            sigma,
        )
        log_transition_backward = self.get_langevin_transition_kernel(
            z_new,
            z,
            grad_new,
            beta * grad_step,
            sigma,
        )
        log_prob = MHKernel.log_prob(
            -beta * energy,
            -beta * energy_new,
            log_transition_forward,
            log_transition_backward,
        )

        return log_prob, energy_new, grad_new

    def get_mask(
        self,
        z,
        z_new,
        energy,
        grad,
        grad_step,
        sigma,
        target=None,
        beta=1.0,
    ):
        energy_new, grad_new = grad_energy(z_new, target, x=None)
        log_transition_forward = self.get_langevin_transition_kernel(
            z,
            z_new,
            grad,
            beta * grad_step,
            sigma,
        )
        log_transition_backward = self.get_langevin_transition_kernel(
            z_new,
            z,
            grad_new,
            beta * grad_step,
            sigma,
        )

        mask = MHKernel.get_mask(
            -beta * energy,
            -beta * energy_new,
            log_transition_forward,
            log_transition_backward,
        )
        return mask, energy_new, grad_new

    def do_transition_step(
        self,
        z,
        z_new,
        energy,
        grad,
        grad_step,
        sigma,
        target=None,
        beta=1.0,
        adapt_stepsize=False,
    ):
        # if adapt_stepsize is True:
        #     if self.adapt_grad_step != 0 and self.adapt_sigma != 0:
        #         grad_step = self.adapt_grad_step
        #         sigma = self.adapt_sigma

        # acc_log_prob, energy_new, grad_new = self.compute_log_probs(z, z_new,
        #                                                             energy,
        #                                                             grad,
        #                                                             grad_step,
        #                                                             sigma,
        #                                                             target,
        #                                                             beta)

        # generate_uniform_var = self.uniform.sample([z.shape[0]]).to(z.device)
        # log_generate_uniform_var = torch.log(generate_uniform_var)
        # mask = log_generate_uniform_var < acc_log_prob
        mask, energy_new, grad_new = self.get_mask(
            z,
            z_new,
            energy,
            grad,
            grad_step,
            sigma,
            target,
            beta,
        )

        with torch.no_grad():
            z[mask] = z_new[mask].detach().clone()
            z = z.data
            z.requires_grad_(True)
            energy = energy.float()
            energy[mask] = energy_new[mask].float()
            energy[~mask] = energy[~mask]
            grad[mask] = grad_new[mask]
            grad[~mask] = grad[~mask]

        if adapt_stepsize:
            mean_acceptance = mask.float().mean()
            self.adapt_grad_step = heuristics_step_size(
                mean_acceptance,
                target_acceptance=0.45,
                stepsize=grad_step,
            )
            self.adapt_sigma = (2 * self.adapt_grad_step) ** 0.5

        return z, energy, grad, mask

    def __call__(
        self,
        z,
        energy,
        grad,
        grad_step=None,
        sigma=None,
        target=None,
        beta=1.0,
        adapt_stepsize=False,
    ):
        if adapt_stepsize is True or grad_step is None:
            if self.adapt_grad_step != 0 and self.adapt_sigma != 0:
                grad_step = self.adapt_grad_step
                sigma = self.adapt_sigma

        z_new = (
            z - grad_step * beta * grad + sigma * self.stand_normal.sample(z.shape[:-1])
        )
        z_new, energy, grad, mask = self.do_transition_step(
            z,
            z_new,
            energy,
            grad,
            grad_step,
            sigma,
            target=target,
            beta=beta,
            adapt_stepsize=adapt_stepsize,
        )
        return z_new, energy, grad, mask


def tempered_transitions_dynamics(
    z,
    target,
    proposal,
    n_steps,
    grad_step,
    eps_scale,
    betas,
):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    device = z.device

    mala_transition = MALATransition(z_dim, device)

    acceptence = torch.zeros(batch_size).to(device)

    betas = np.array(betas)
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:])
    E0, grad0 = grad_energy(z, target, x=None)
    for _ in range(n_steps):
        z_forward = z.clone().detach()
        z_forward.requires_grad_(True)
        energy_forward = torch.zeros(batch_size, len(betas) - 1)
        energy_forward[:, 0] = E0

        E = E0  # data #.detach().clone()
        grad = grad0  # data #detach().clone()

        for i, beta in enumerate(betas[1:-1]):
            eps = eps_scale * proposal.sample([batch_size])

            z_forward_new = z_forward - grad_step * beta * grad + eps
            (z_forward, E, grad, mask) = mala_transition.do_transition_step(
                z_forward,
                z_forward_new,
                E,
                grad,
                grad_step,
                eps_scale,
                target,
                beta=beta,
            )
            energy_forward[:, i + 1] = E

        z_backward = z_forward.clone().detach()
        z_backward.requires_grad_(True)
        energy_backward = torch.zeros(batch_size, len(betas) - 1)
        energy_backward[:, -1] = E
        for i, beta in enumerate(betas[::-1][1:-1]):
            eps = eps_scale * proposal.sample([batch_size])

            z_backward_new = z_backward - grad_step * beta * grad + eps
            (z_backward, E, grad, mask) = mala_transition.do_transition_step(
                z_backward,
                z_backward_new,
                E,
                grad,
                grad_step,
                eps_scale,
                target,
                beta=beta,
            )
            j = len(betas) - i - 3
            energy_backward[:, j] = E

        F_forward = (betas_diff * energy_forward).sum(-1)
        F_backward = (betas_diff * energy_backward).sum(-1)
        log_accept_prob = F_forward - F_backward

        generate_uniform_var = mala_transition.uniform.sample([batch_size]).to(
            z.device,
        )
        log_generate_uniform_var = torch.log(generate_uniform_var)
        mask = log_generate_uniform_var < log_accept_prob

        acceptence += mask
        with torch.no_grad():
            z[mask] = z_backward[mask].detach().clone()
            z = z.data
            z.requires_grad_(True)
            z_sp.append(z.clone().detach())

            E0[mask] = E[mask].data

            grad0[mask] = grad[mask].data
            grad0.requires_grad_(True)

    return z_sp, acceptence


tempered_transitions_sampling = sampling_from_dynamics(
    tempered_transitions_dynamics,
)


def i_ais_z_dynamics(z, target, n_steps, grad_step, eps_scale, N, betas):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    device = z.device

    mala_transition = MALATransition(z_dim, device)
    acceptence_rate = 0.0

    betas = np.array(betas)
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:])  # n-1

    scale_proposal = 1.0
    proposal = init_independent_normal(scale_proposal, z_dim, device)

    E_last, grad_last = grad_energy(z, target, x=None)
    energy_backward_last = None
    for step in tqdm(range(n_steps)):
        if step == 0:
            N_ = N
        else:
            N_ = N - 1

        Z = z.unsqueeze(1).repeat(1, N_, 1)
        z_batch = (
            torch.transpose(Z, 0, 1).reshape((batch_size * N_, z_dim)).detach().clone()
        )
        z_batch.requires_grad_(True)

        E = E_last.unsqueeze(1).repeat(1, N_)
        grad = grad_last.unsqueeze(1).repeat(1, N_, 1)
        grad = torch.transpose(grad, 0, 1).reshape((batch_size * N_, z_dim)).data
        # detach().clone()

        z_backward = z_batch
        z_backward.requires_grad_(True)
        energy_backward = torch.zeros(batch_size, N_, len(betas) - 1)  # n-1
        energy_backward[..., len(betas) - 2] = E  # detach().clone()

        E = E.T.reshape(-1)
        # betas[0] = 1, betas[n-1] = 0, betas[::-1][1:-1] - increasing
        # length is n-2
        for i, beta in enumerate(betas[::-1][1:-1]):
            j = len(betas) - i - 3
            eps = eps_scale * proposal.sample([batch_size * N_])

            z_backward_new = z_backward - beta * grad_step * grad + eps
            (z_backward, E, grad, mask) = mala_transition.do_transition_step(
                z_backward,
                z_backward_new,
                E,
                grad,
                grad_step,
                eps_scale,
                target,
                beta=beta,
            )
            accept = mask.sum()
            acceptence_rate += accept
            E_ = E.reshape(Z.shape[:-1][::-1]).T
            energy_backward[..., j] = E_.detach().clone()

        z_backward = torch.transpose(
            z_backward.reshape(list(Z.shape[:-1][::-1]) + [Z.shape[-1]]),
            0,
            1,
        )
        grad = torch.transpose(
            grad.reshape(list(Z.shape[:-1][::-1]) + [Z.shape[-1]]),
            0,
            1,
        )
        # E = E.reshape(Z.shape[:-1][::-1]).T

        if step > 0:
            z_backward = torch.cat([z_backward, z.unsqueeze(1)], dim=1)
            energy_backward = torch.cat(
                [energy_backward, energy_backward_last.unsqueeze(1)],
                dim=1,
            )
            grad = torch.cat([grad, grad_last.unsqueeze(1)], dim=1)

        E = energy_backward[..., 0]

        F_backward = (betas_diff[None, None, :] * energy_backward).sum(-1)
        log_weights = -F_backward

        max_logs = torch.max(log_weights, dim=1)[0]  # .unsqueeze(-1).repeat((1, N))
        log_weights = log_weights - max_logs[:, None]
        sum_weights = torch.logsumexp(log_weights, dim=1)
        log_weights = log_weights - sum_weights[:, None]
        weights = log_weights.exp()
        weights[weights != weights] = 0.0
        weights[weights.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weights, 1).squeeze().tolist()

        z = z_backward[np.arange(batch_size), indices, :]
        z = z.data
        z.requires_grad_(True)
        z_sp.append(z.detach().clone())

        E_last = E[np.arange(batch_size), indices].data

        grad_last = grad[np.arange(batch_size), indices, :].data
        grad_last.requires_grad_(True)

        energy_backward_last = energy_backward[np.arange(batch_size), indices, :]

    acceptence_rate = (
        acceptence_rate / (batch_size * N) / len(betas[::-1][1:-1]) / n_steps
    )
    return z_sp, acceptence_rate


def i_ais_v_dynamics(z, target, n_steps, grad_step, eps_scale, N, betas, rho):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    device = z.device

    # mala_transition = CiterMALATransition(z_dim, device)
    mala_transition = MALATransition(z_dim, device)
    acceptence_rate = 0.0

    betas = np.array(betas)
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:])  # n-1

    scale_proposal = 1.0
    proposal = init_independent_normal(scale_proposal, z_dim, device)

    z_last = z
    z_backward_last = None
    energy_backward_last = None
    for step in tqdm(range(n_steps)):
        if step == 0:
            N_ = N
        else:
            N_ = N - 1

        z = z_last.unsqueeze(1).repeat(1, N_, 1)

        z = rho * z + ((1 - rho**2) ** 0.5) * proposal.sample(
            [batch_size, N_],
        )

        z_batch = (
            torch.transpose(z, 0, 1).reshape((batch_size * N_, z_dim)).detach().clone()
        )
        z_batch.requires_grad_(True)

        # E = E.unsqueeze(1).repeat(1, N)
        # grad = grad.unsqueeze(1).repeat(1, N, 1)
        # grad = torch.transpose(grad, 0, 1).reshape((batch_size*N, z_dim)).data #detach().clone()

        E, grad = grad_energy(z_batch, target)
        E = E.reshape(N_, batch_size).T

        z_backward = z_batch
        z_backward.requires_grad_(True)
        energy_backward = torch.zeros(batch_size, N_, len(betas) - 1)  # n-1
        energy_backward[..., len(betas) - 2] = E  # detach().clone()

        E = E.T.reshape(-1)
        # betas[0] = 1, betas[n-1] = 0, betas[::-1][1:-1] - increasing
        # length is n-2
        for i, beta in enumerate(betas[::-1][1:-1]):
            j = len(betas) - i - 3
            eps = eps_scale * proposal.sample([batch_size * N_])

            # z_backward_new = (1. - grad_step) * z_backward - beta * grad_step * grad + eps
            z_backward_new = z_backward - beta * grad_step * grad + eps
            (z_backward, E, grad, mask) = mala_transition.do_transition_step(
                z_backward,
                z_backward_new,
                E,
                grad,
                grad_step,
                eps_scale,
                target,
                beta=beta,
            )
            accept = mask.sum()
            acceptence_rate += accept
            E_ = E.reshape(z.shape[:-1][::-1]).T
            energy_backward[..., j] = E_.detach().clone()

        z_backward = torch.transpose(
            z_backward.reshape(list(z.shape[:-1][::-1]) + [z.shape[-1]]),
            0,
            1,
        )
        # grad = torch.transpose(grad.reshape(list(z.shape[:-1][::-1]) + [z.shape[-1]]), 0, 1)
        # E = E.reshape(z.shape[:-1][::-1]).T

        if step > 0:
            z = torch.cat([z, z_last.unsqueeze(1)], dim=1)
            z_backward = torch.cat(
                [z_backward, z_backward_last.unsqueeze(1)],
                dim=1,
            )
            energy_backward = torch.cat(
                [energy_backward, energy_backward_last.unsqueeze(1)],
                dim=1,
            )
            # grad = torch.cat([grad, grad_last.unsqueeze(1)], dim=1)

        # E = energy_backward[..., 0]

        F_backward = (betas_diff[None, None, :] * energy_backward).sum(-1)
        log_weights = -F_backward

        max_logs = torch.max(log_weights, dim=1)[0]
        log_weights = log_weights - max_logs[:, None]
        sum_weights = torch.logsumexp(log_weights, dim=1)
        log_weights = log_weights - sum_weights[:, None]
        weights = log_weights.exp()
        weights[weights != weights] = 0.0
        weights[weights.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weights, 1).squeeze().tolist()

        z_last = z[np.arange(batch_size), indices, :]
        z_last = z_last.data
        z_last.requires_grad_(True)

        z_backward_last = z_backward[np.arange(batch_size), indices, :]
        z_sp.append(z_backward_last.detach().clone())

        # grad_last = grad[np.arange(batch_size), indices, :].data
        # grad_last.requires_grad_(True)

        energy_backward_last = energy_backward[np.arange(batch_size), indices, :]

    acceptence_rate = 1.0
    # acceptence_rate/(batch_size*N)/len(betas[::-1][1:-1])/n_steps
    return z_sp, acceptence_rate


def i_ais_b_dynamics(z, target, n_steps, grad_step, eps_scale, N, betas, rho):
    z_sp = [z.clone().detach()]
    batch_size, z_dim = z.shape[0], z.shape[1]
    device = z.device

    mala_transition = MALATransition(z_dim, device)
    acceptence_rate = 0.0

    betas = np.array(betas)
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:])  # n-1

    scale_proposal = 1.0
    proposal = init_independent_normal(scale_proposal, z_dim, device)

    # E_last, grad_last = grad_energy(z, target, x=None)
    z_last = z
    z_backward_last = None
    energy_backward_last = None
    grad_last = None
    for step in tqdm(range(n_steps)):
        if step == 0:
            N_ = N
        else:
            N_ = N - 1

        z = z_last.unsqueeze(1).repeat(1, N_, 1)

        z = rho * z + ((1 - rho**2) ** 0.5) * proposal.sample(
            [batch_size, N_],
        )

        z_batch = (
            torch.transpose(z, 0, 1).reshape((batch_size * N_, z_dim)).detach().clone()
        )
        z_batch.requires_grad_(True)

        # E = E.unsqueeze(1).repeat(1, N)
        # grad = grad.unsqueeze(1).repeat(1, N, 1)
        # grad = torch.transpose(grad, 0, 1).reshape((batch_size*N, z_dim)).data #detach().clone()

        E, grad = grad_energy(z_batch, target)
        E = E.reshape(N_, batch_size).T

        z_backward = z_batch
        z_backward.requires_grad_(True)
        energy_backward = torch.zeros(batch_size, N_, len(betas) - 1)  # n-1
        energy_backward[..., len(betas) - 2] = E  # .detach().clone()

        E = E.T.reshape(-1)
        # betas[0] = 1, betas[n-1] = 0, betas[::-1][1:-1] - increasing
        # length is n-2
        for i, beta in enumerate(betas[::-1][1:-1]):
            j = len(betas) - i - 3
            eps = eps_scale * proposal.sample([batch_size * N_])

            # z_backward_new = (1 - grad_step) * z_backward - beta * grad_step * grad + eps
            z_backward_new = z_backward - beta * grad_step * grad + eps
            (z_backward, E, grad, mask) = mala_transition.do_transition_step(
                z_backward,
                z_backward_new,
                E,
                grad,
                grad_step,
                eps_scale,
                target,
                beta=beta,
            )
            accept = mask.sum()
            acceptence_rate += accept

            E_ = E.reshape(z.shape[:-1][::-1]).T
            energy_backward[..., j] = E_.detach().clone()

        z_backward = torch.transpose(
            z_backward.reshape(list(z.shape[:-1][::-1]) + [z.shape[-1]]),
            0,
            1,
        )
        grad = torch.transpose(
            grad.reshape(list(z.shape[:-1][::-1]) + [z.shape[-1]]),
            0,
            1,
        )
        # E = E.reshape(z.shape[:-1][::-1]).T

        if step > 0:
            # z = torch.cat([z, z_last.unsqueeze(1)], dim=1)
            z_backward = torch.cat(
                [z_backward, z_backward_last.unsqueeze(1)],
                dim=1,
            )
            energy_backward = torch.cat(
                [energy_backward, energy_backward_last.unsqueeze(1)],
                dim=1,
            )
            grad = torch.cat([grad, grad_last.unsqueeze(1)], dim=1)

        E = energy_backward[..., 0]

        F_backward = (betas_diff[None, None, :] * energy_backward).sum(-1)
        log_weights = -F_backward

        max_logs = torch.max(log_weights, dim=1)[0]
        log_weights = log_weights - max_logs[:, None]
        sum_weights = torch.logsumexp(log_weights, dim=1)
        log_weights = log_weights - sum_weights[:, None]
        weights = log_weights.exp()
        weights[weights != weights] = 0.0
        weights[weights.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weights, 1).squeeze().tolist()

        z_backward = z_backward[np.arange(batch_size), indices, :]
        z_backward = z_backward.data
        z_backward.requires_grad_(True)

        z_sp.append(z_backward.detach().clone())

        z_backward_last = z_backward
        energy_backward_last = energy_backward[np.arange(batch_size), indices, :]

        E = E[np.arange(batch_size), indices].data

        grad = grad[np.arange(batch_size), indices, :].data
        grad.requires_grad_(True)
        grad_last = grad

        z_backward = z_backward.detach().clone()
        z_backward.requires_grad_(True)

        for i, beta in enumerate(betas[:-1]):
            eps = eps_scale * proposal.sample([batch_size])

            # z_backward_new = (1 - grad_step) * z_backward - beta * grad_step * grad + eps
            z_backward_new = z_backward - beta * grad_step * grad + eps
            (z_backward, E, grad, mask) = mala_transition.do_transition_step(
                z_backward,
                z_backward_new,
                E,
                grad,
                grad_step,
                eps_scale,
                target,
                beta=beta,
            )

        z = z_backward.data
        z.requires_grad_(True)
        z_last = z
        # z_sp.append(z.detach().clone())
    acceptence_rate = (
        acceptence_rate / (batch_size * N) / len(betas[::-1][1:-1]) / n_steps
    )
    return z_sp, acceptence_rate


def compute_probs_from_log_probs(log_probs):
    mask_zeros = log_probs > 0.0
    log_probs[mask_zeros] = 0.0
    probs = log_probs.exp()
    return probs


def citerais_mala_dynamics(
    z,
    target,
    n_steps,
    grad_step,
    eps_scale,
    N,
    betas,
    rhos,
):
    z_sp = [z[:, -1, :].clone().detach()]
    batch_size, T, z_dim = z.shape[0], z.shape[1], z.shape[2]
    T = T - 1  # ??
    # T = len(betas) - 2
    device = z.device

    mala_transition = MALATransition(z_dim, device)

    betas = np.array(betas)
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:]).to(device)  # n-1

    z_flat = (
        torch.transpose(z, 0, 1)
        .reshape((batch_size * z.shape[1], z_dim))
        .detach()
        .clone()
    )
    z_flat.requires_grad_(True)
    E_flat, grad_flat = grad_energy(z_flat, target, None)
    grad = (
        torch.transpose(
            grad_flat.reshape(list(z.shape[:-1][::-1]) + [z.shape[-1]]),
            0,
            1,
        )
        .detach()
        .clone()
    )
    E = E_flat.reshape(z.shape[:-1][::-1]).T.data

    for _ in trange(n_steps):
        v = torch.zeros((batch_size, T + 1, N, z_dim), dtype=z.dtype).to(
            device,
        )
        u = torch.zeros((batch_size, T + 1, N), dtype=z.dtype).to(device)

        # step 1
        kappa = torch.zeros((batch_size, T + 1, z_dim), dtype=z.dtype).to(
            device,
        )
        kappa_t_noise = mala_transition.stand_normal.sample(
            [batch_size, T + 1],
        )
        kappa[:, 0, :] = (
            rhos[-1] * z[:, 0, :]
            + ((1 - rhos[-1] ** 2) ** 0.5) * kappa_t_noise[:, 0, :]
        )

        for t in range(1, T + 1):
            beta = betas[::-1][t]
            rho = rhos[::-1][t]

            not_equal_mask = torch.ne(z[:, t, :], z[:, t - 1, :]).max(dim=-1)[0]
            equal_mask = ~not_equal_mask
            num_not_equal = not_equal_mask.sum()
            num_equal = equal_mask.sum()

            if num_not_equal > 0:
                z_t_not_equal = z[not_equal_mask, t, :].detach().clone()
                z_t_1_not_equal = z[not_equal_mask, t - 1, :].detach().clone()
                z_t_not_equal.requires_grad_(True)
                z_t_1_not_equal.requires_grad_(True)

                # E_t_1_not_equal, grad_t_1_not_equal = grad_energy(z_t_1_not_equal, target, x=None)
                E_t_1_not_equal = E[not_equal_mask, t - 1]
                grad_t_1_not_equal = grad[not_equal_mask, t - 1, :]

                (log_probs, _, _) = mala_transition.compute_log_probs(
                    z_t_1_not_equal,
                    z_t_not_equal,
                    E_t_1_not_equal,
                    grad_t_1_not_equal,
                    grad_step,
                    eps_scale,
                    target,
                    beta=beta,
                )

                # v[not_equal_mask, t, 0, :] = (z_t_not_equal - (1. - grad_step) * z_t_1_not_equal
                #                               + grad_step * beta * grad_t_1_not_equal)/ (2 * grad_step)**.5
                #                               # eps_scale
                v[not_equal_mask, t, 0, :] = (
                    z_t_not_equal
                    - z_t_1_not_equal
                    + grad_step * beta * grad_t_1_not_equal
                ) / eps_scale

                probs = compute_probs_from_log_probs(log_probs)
                generate_uniform_var = mala_transition.uniform.sample(
                    [probs.shape[0]],
                ).to(probs.device)
                weight_uniform_var = generate_uniform_var * probs

                u[not_equal_mask, t, 0] = weight_uniform_var.float()

            if num_equal > 0:
                z_t_equal = z[equal_mask, t, :]
                z_t_1_equal = z[equal_mask, t - 1, :].detach().clone()
                z_t_1_equal.requires_grad_(True)

                # E_t_1_equal, grad_t_1_equal = grad_energy(z_t_1_equal,
                #                                           target, x=None)
                E_t_1_equal = E[equal_mask, t - 1].detach().clone()
                grad_t_1_equal = grad[equal_mask, t - 1, :].detach().clone()

                second_part_no_noise = (
                    1.0 - grad_step
                ) * z_t_1_equal - grad_step * beta * grad_t_1_equal
                # second_part_no_noise = z_t_1_equal - grad_step * beta * grad_t_1_equal

                stop = False
                # num_updates = 0
                update_mask = torch.zeros(num_equal, dtype=torch.bool).to(
                    z_t_equal.device,
                )
                z_t_1_equal = z_t_1_equal.detach().clone()
                z_t_1_equal.requires_grad_(True)

                while not stop:
                    cur_u = mala_transition.uniform.sample([num_equal]).to(
                        z_t_equal.device,
                    )
                    cur_v = mala_transition.stand_normal.sample(
                        [num_equal],
                    ).to(z_t_equal.device)
                    second_part = second_part_no_noise + cur_v * eps_scale
                    second_part = second_part.detach().clone()
                    second_part.requires_grad_(True)

                    (log_probs, _, _) = mala_transition.compute_log_probs(
                        z_t_1_equal,
                        second_part,
                        E_t_1_equal,
                        grad_t_1_equal,
                        grad_step,
                        eps_scale,
                        target,
                        beta=beta,
                    )

                    probs = compute_probs_from_log_probs(log_probs)
                    mask_assign = cur_u <= probs
                    new_assign = torch.logical_and(mask_assign, ~update_mask)

                    u[equal_mask, t, 0][new_assign] = cur_u[new_assign]
                    v[equal_mask, t, 0, :][new_assign] = cur_v[new_assign]

                    update_mask = torch.logical_or(update_mask, new_assign)
                    updates_num = update_mask.sum()
                    if updates_num == num_equal:
                        stop = True

            kappa[:, t, :] = (
                rho * v[:, t, 0, :] + ((1 - rho**2) ** 0.5) * kappa_t_noise[:, t, :]
            )

        # step 2
        # W = mala_transition.stand_normal.sample([batch_size, N - 1])
        # Z - tensor (bs, T + 1, N, dim)
        Z = torch.zeros((batch_size, T + 1, N, z_dim), dtype=z.dtype).to(
            device,
        )

        kappa_repeat = kappa[:, 0, :].unsqueeze(1).repeat(1, N - 1, 1)
        kappa_N_noise = mala_transition.stand_normal.sample(
            [batch_size, N - 1],
        )
        kappa_repeat_N = kappa.unsqueeze(2).repeat(1, 1, N - 1, 1)

        Z[:, :, 0, :] = z
        Z[:, 0, 1:, :] = (
            rhos[-1] * kappa_repeat + ((1 - rhos[-1] ** 2) ** 0.5) * kappa_N_noise
        )

        kappa_flat = (
            torch.transpose(Z[:, 0, 1:, :], 0, 1)
            .reshape((batch_size * (N - 1), z_dim))
            .detach()
            .clone()
        )
        kappa_E_flat, kappa_grad_flat = grad_energy(kappa_flat, target)
        kappa_E = kappa_E_flat.reshape(N - 1, batch_size).T
        kappa_grad = torch.transpose(
            kappa_grad_flat.reshape(N - 1, batch_size, z_dim),
            0,
            1,
        )

        energy = torch.zeros(batch_size, T + 1, N).to(device)

        energy[:, :, 0] = E.data
        energy[:, 0, 1:] = kappa_E

        grads = torch.zeros(batch_size, T + 1, N, z_dim).to(device)
        grads[:, :, 0, :] = grad.detach().clone()
        grads[:, 0, 1:, :] = kappa_grad

        W_2 = mala_transition.stand_normal.sample([batch_size, T, N - 1])

        for t in range(1, T + 1):
            beta = betas[::-1][t]
            rho = rhos[::-1][t]

            v[:, t, 1:, :] = (
                rho * kappa_repeat_N[:, t, :, :]
                + ((1 - rho**2) ** 0.5) * W_2[:, t - 1, :, :]
            )
            # z_t_1_j_shape = Z[:, t - 1, 1:, :].shape
            z_t_1_j_flatten = (
                torch.transpose(Z[:, t - 1, 1:, :], 0, 1)
                .reshape((batch_size * (N - 1), z_dim))
                .detach()
                .clone()
                .to(device)
            )
            z_t_1_j_flatten.requires_grad_(True)

            # _, grad_z_t_1_j_flatten = grad_energy(z_t_1_j_flatten, target, x=None)
            # grad_z_t_1_j = (torch.transpose(grad_z_t_1_j_flatten
            #                 .reshape(list(z_t_1_j_shape[:-1][::-1]) + [z_t_1_j_shape[-1]]), 0, 1))
            grad_z_t_1_j = grads[:, t - 1, 1:, :]
            grad_z_t_1_j_flatten = (
                torch.transpose(grad_z_t_1_j, 0, 1)
                .reshape((batch_size * (N - 1), z_dim))
                .detach()
                .clone()
                .to(device)
            )
            E_z_t_1_j = energy[:, t - 1, 1:]
            E_z_t_1_j_flatten = E_z_t_1_j.T.reshape(batch_size * (N - 1))

            Z_t_1_j = Z[:, t - 1, 1:, :]
            # Z_t_1_j_shape = Z_t_1_j.shape
            # p_t_j = (1. - grad_step) * Z_t_1_j - grad_step * beta * grad_z_t_1_j \
            #                                        + eps_scale * v[:, t, 1:, :]
            p_t_j = (
                Z_t_1_j - grad_step * beta * grad_z_t_1_j + eps_scale * v[:, t, 1:, :]
            )

            p_t_j_flatten = (
                torch.transpose(p_t_j, 0, 1)
                .reshape((batch_size * (N - 1), z_dim))
                .detach()
                .clone()
                .to(device)
            )
            p_t_j_flatten.requires_grad_(True)

            Z_t_1_j_flatten = (
                torch.transpose(Z_t_1_j, 0, 1)
                .reshape((batch_size * (N - 1), z_dim))
                .detach()
                .clone()
                .to(device)
            )
            Z_t_1_j_flatten.requires_grad_(True)

            (z_t, E_t, grad_t, mask) = mala_transition.do_transition_step(
                Z_t_1_j_flatten,
                p_t_j_flatten,
                E_z_t_1_j_flatten,
                grad_z_t_1_j_flatten,
                grad_step,
                eps_scale,
                target,
                beta,
            )

            z_t = torch.transpose(z_t.reshape(N - 1, batch_size, z_dim), 0, 1)
            E_t = E_t.reshape(N - 1, batch_size).T
            grad_t = torch.transpose(
                grad_t.reshape(N - 1, batch_size, z_dim),
                0,
                1,
            )
            Z[:, t, 1:, :] = z_t
            energy[:, t, 1:] = E_t.detach().clone()
            grads[:, t, 1:, :] = grad_t.detach().clone()

        log_weights = -(betas_diff[None, :, None] * energy[:, 1:, :]).sum(1)

        max_logs = torch.max(log_weights, dim=1)[0]
        log_weights = log_weights - max_logs[:, None]
        weights = torch.exp(log_weights)
        sum_weights = torch.sum(weights, dim=1)
        weights = weights / sum_weights[:, None]
        weights[weights != weights] = 0.0
        weights[weights.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weights, 1).squeeze().tolist()
        z = Z[np.arange(batch_size), :, indices, :]
        E = energy[np.arange(batch_size), :, indices]
        grad = grads[np.arange(batch_size), :, indices, :]
        z_sp.append(z[:, -1, :].detach().clone())

    return z_sp, 1.0


def ais_sampling(
    target,
    proposal,
    n_steps,
    grad_step,
    eps_scale,
    n,
    batch_size,
    N,
    betas,
    rhos,
):
    z_last = []
    zs = []
    z = None
    for _ in tqdm(range(0, n, batch_size)):
        z = proposal.sample([batch_size, len(betas)])
        z.requires_grad_(True)

        z_sp, _ = citerais_mala_dynamics(
            z,
            target,
            n_steps,
            grad_step,
            eps_scale,
            N,
            betas,
            rhos,
        )
        last = z_sp[-1].data.cpu().numpy()
        z_last.append(last)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    z_last_np = np.asarray(z_last).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return z_last_np, zs


def citerais_ula_dynamics(
    z,
    target,
    proposal,
    n_steps,
    grad_step,
    eps_scale,
    N,
    betas,
    rhos,
    do_ar=True,
    max_n_rej=10,
    pbern=0.5,
):
    """
    do_ar: bool - include path from the privious step to N current paths
    max_n_rej: int - maximum number of rejections together, if reached - don't reject this step (aka refreshing)
    """
    # T = len(betas)
    batch_size, T, z_dim = z.shape
    device = z.device
    z = proposal.sample([batch_size])
    z.requires_grad_(True)
    E, grad = grad_energy(z, target, None)

    # step 0: Generate start ULA path
    start = [z.detach().clone()]
    mala_transition = MALATransition(z_dim, device)
    W = mala_transition.stand_normal.sample([batch_size, T])

    for t in range(1, T):
        beta = betas[::-1][t]
        # rho = rhos[::-1][t]
        z = z - grad_step * beta * grad + eps_scale * W[:, t, :]
        E, grad = grad_energy(z, target, None)
        start.append(z.detach().clone())

    z = torch.stack(start, 1)
    # end of step 0

    z_sp = [z[:, -1, :].clone().detach()]
    # z_sp = [z.clone().detach()]
    traj_hist = [z[:5].clone().detach()]
    # batch_size, T, z_dim = z.shape[0], z.shape[1], z.shape[2]
    T = T - 1  # ??
    # T = len(betas) - 2

    betas = np.array(betas)
    betas_diff = torch.FloatTensor(betas[:-1] - betas[1:]).to(device)  # n-1

    z_flat = (
        torch.transpose(z, 0, 1)
        .reshape((batch_size * z.shape[1], z_dim))
        .detach()
        .clone()
    )
    z_flat.requires_grad_(True)
    E_flat, grad_flat = grad_energy(z_flat, target, None)
    grad = (
        torch.transpose(
            grad_flat.reshape(list(z.shape[:-1][::-1]) + [z.shape[-1]]),
            0,
            1,
        )
        .detach()
        .clone()
    )
    E = E_flat.reshape(z.shape[:-1][::-1]).T.data

    n_rej = torch.zeros(batch_size)
    for _ in trange(n_steps):
        v = torch.zeros((batch_size, T + 1, N, z_dim), dtype=z.dtype).to(
            device,
        )
        # step 1
        kappa = torch.zeros((batch_size, T + 1, z_dim), dtype=z.dtype).to(
            device,
        )
        kappa_t_noise = mala_transition.stand_normal.sample(
            [batch_size, T + 1],
        )
        kappa[:, 0, :] = (
            rhos[-1] * z[:, 0, :]
            + ((1 - rhos[-1] ** 2) ** 0.5) * kappa_t_noise[:, 0, :]
        )

        for t in range(1, T + 1):
            beta = betas[::-1][t]
            rho = rhos[::-1][t]
            v[:, t, 0, :] = (
                z[:, t, :] - z[:, t - 1, :] + grad_step * beta * grad[:, t - 1, :]
            ) / eps_scale
            kappa[:, t, :] = (
                rho * v[:, t, 0, :] + ((1 - rho**2) ** 0.5) * kappa_t_noise[:, t, :]
            )

        # step 2
        kappa_repeat = kappa[:, 0, :].unsqueeze(1).repeat(1, N - 1, 1)
        kappa_N_noise = mala_transition.stand_normal.sample(
            [batch_size, N - 1],
        )
        kappa_repeat_N = kappa.unsqueeze(2).repeat(1, 1, N - 1, 1)

        Z = torch.zeros((batch_size, T + 1, N, z_dim), dtype=z.dtype).to(
            device,
        )
        Z[:, :, 0, :] = z
        mask = (
            torch.distributions.bernoulli.Bernoulli(
                probs=torch.tensor([pbern]),
            )
            .sample(torch.Size([batch_size, N - 1]))
            .squeeze(-1)
            .to(device)
        )

        Z[:, 0, 1:, :] = (
            rhos[-1] * kappa_repeat * mask[..., None]
            + ((1 - rhos[-1] ** 2 * mask[..., None]) ** 0.5) * kappa_N_noise
        )

        kappa_flat = (
            torch.transpose(Z[:, 0, 1:, :], 0, 1)
            .reshape((batch_size * (N - 1), z_dim))
            .detach()
            .clone()
        )
        kappa_E_flat, kappa_grad_flat = grad_energy(kappa_flat, target)
        kappa_E = kappa_E_flat.reshape(N - 1, batch_size).T
        kappa_grad = torch.transpose(
            kappa_grad_flat.reshape(N - 1, batch_size, z_dim),
            0,
            1,
        )

        energy = torch.zeros(batch_size, T + 1, N).to(device)
        energy[:, :, 0] = E.data
        energy[:, 0, 1:] = kappa_E

        grads = torch.zeros(batch_size, T + 1, N, z_dim).to(device)
        grads[:, :, 0, :] = grad.detach().clone()
        grads[:, 0, 1:, :] = kappa_grad

        W_2 = mala_transition.stand_normal.sample([batch_size, T, N - 1])

        for t in range(1, T + 1):
            beta = betas[::-1][t]
            rho = rhos[::-1][t]

            v[:, t, 1:, :] = (
                rho * kappa_repeat_N[:, t, :, :] * mask[..., None]
                + ((1 - rho**2 * mask[..., None]) ** 0.5) * W_2[:, t - 1, :, :]
            )
            # v[:, t, 1:, :] = (rho*kappa_repeat_N[:, t - 1, :, :]*mask[..., None] +
            #                   ((1 - rho**2 * mask[..., None])**0.5) * W_2[:, t - 1, :, :])

            # z_t_1_j_flatten = (torch.transpose(Z[:, t - 1, 1:, :], 0, 1)
            #                    .reshape((batch_size*(N-1), z_dim))
            #                    .detach().clone().to(device))
            # z_t_1_j_flatten.requires_grad_(True)

            grad_z_t_1_j = grads[:, t - 1, 1:, :]

            grad_z_t_1_j_flatten = (
                torch.transpose(grad_z_t_1_j, 0, 1)
                .reshape((batch_size * (N - 1), z_dim))
                .detach()
                .clone()
                .to(device)
            )
            E_z_t_1_j = energy[:, t - 1, 1:]
            E_z_t_1_j_flatten = E_z_t_1_j.T.reshape(batch_size * (N - 1))

            Z_t_1_j = Z[:, t - 1, 1:, :]
            p_t_j = (
                Z_t_1_j - grad_step * beta * grad_z_t_1_j + eps_scale * v[:, t, 1:, :]
            )

            p_t_j_flatten = (
                torch.transpose(p_t_j, 0, 1)
                .reshape((batch_size * (N - 1), z_dim))
                .detach()
                .clone()
                .to(device)
            )
            p_t_j_flatten.requires_grad_(True)

            Z_t_1_j_flatten = (
                torch.transpose(Z_t_1_j, 0, 1)
                .reshape((batch_size * (N - 1), z_dim))
                .detach()
                .clone()
                .to(device)
            )
            Z_t_1_j_flatten.requires_grad_(True)

            (_, E_t, grad_t) = mala_transition.compute_log_probs(
                Z_t_1_j_flatten,
                p_t_j_flatten,
                E_z_t_1_j_flatten,
                grad_z_t_1_j_flatten,
                grad_step,
                eps_scale,
                target,
                beta,
            )
            z_t = p_t_j_flatten
            z_t = torch.transpose(z_t.reshape(N - 1, batch_size, z_dim), 0, 1)
            E_t = E_t.reshape(N - 1, batch_size).T
            grad_t = torch.transpose(
                grad_t.reshape(N - 1, batch_size, z_dim),
                0,
                1,
            )
            Z[:, t, 1:, :] = z_t
            energy[:, t, 1:] = E_t.detach().clone()
            grads[:, t, 1:, :] = grad_t.detach().clone()

        traj_hist.append(Z[:5, ..., 1, :].detach().clone())

        log_weights = -(betas_diff[None, :, None] * energy[:, 1:, :]).sum(1)

        max_logs = torch.max(log_weights, dim=1)[0]
        log_weights = log_weights - max_logs[:, None]
        weights = torch.exp(log_weights)
        sum_weights = torch.sum(weights, dim=1)
        weights = weights / sum_weights[:, None]
        weights[weights != weights] = 0.0
        weights[weights.sum(1) == 0.0] = 1.0

        if not do_ar:
            weights[:, 0, ...] = 0.0

        indices = torch.multinomial(weights, 1).squeeze().tolist()
        n_rej += (torch.tensor(indices) == 0).float()

        weights[n_rej > max_n_rej, 0, ...] = 0.0
        indices = torch.multinomial(weights, 1).squeeze().tolist()
        n_rej[n_rej > max_n_rej] = 0

        z = Z[np.arange(batch_size), :, indices, :]
        E = energy[np.arange(batch_size), :, indices]
        grad = grads[np.arange(batch_size), :, indices, :]
        z_sp.append(z[:, -1, :].detach().clone())
        # z_sp.append(z.detach().clone())

    return z_sp, 1.0, traj_hist


def citerais_ula_sampling(
    target,
    proposal,
    n_steps,
    grad_step,
    eps_scale,
    n,
    batch_size,
    N,
    betas,
    rhos,
):
    z_last = []
    zs = []
    z = None
    for _ in tqdm(range(0, n, batch_size)):
        z = proposal.sample([batch_size, len(betas)])
        z.requires_grad_(True)
        z_sp, _, _ = citerais_ula_dynamics(
            z,
            target,
            proposal,
            n_steps,
            grad_step,
            eps_scale,
            N,
            betas,
            rhos,
        )
        last = z_sp[-1].data.cpu().numpy()
        z_last.append(last)
        zs.append(np.stack([o.data.cpu().numpy() for o in z_sp], axis=0))

    z_last_np = np.asarray(z_last).reshape(-1, z.shape[-1])
    zs = np.stack(zs, axis=0)
    return z_last_np, zs
