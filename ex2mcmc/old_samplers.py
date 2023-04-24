import numpy as np
import torch
import tqdm
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal as MNormal


def grad_log_and_output_target_dens(log_target_dens, z):
    """
    returns the gradient of log-density
    """
    inp = z.clone().detach()
    inp.requires_grad_(True)

    output = log_target_dens(inp, detach=False)
    output.sum().backward()

    grad = inp.grad.data.detach()

    return grad, output.data.detach()


def mala_step(log_target_dens, x0, gamma, mala_iters, stats=None):
    """
    function to perform mala_iters times MALA step
    """
    N_samples, lat_size = x0.shape
    device = x0.device

    # generate proposals
    x_cur = x0
    for i in range(mala_iters):
        x_cur_grad, x_cur_log_target = grad_log_and_output_target_dens(
            log_target_dens, x_cur
        )
        x_cur_proposal = MNormal(
            x_cur + gamma * x_cur_grad,
            2
            * gamma
            * torch.eye(lat_size)[None, :, :].repeat(N_samples, 1, 1).to(device),
        )

        x_next = x_cur_proposal.sample()

        x_next_grad, x_next_log_target = grad_log_and_output_target_dens(
            log_target_dens, x_next
        )
        x_next_proposal = MNormal(
            x_next + gamma * x_next_grad,
            2
            * gamma
            * torch.eye(lat_size)[None, :, :].repeat(N_samples, 1, 1).to(device),
        )

        # compute accept-reject
        log_prob_accept = (
            x_next_log_target
            + x_next_proposal.log_prob(x_cur)
            - x_cur_log_target
            - x_cur_proposal.log_prob(x_next)
        )
        log_prob_accept = torch.clamp(log_prob_accept, max=0.0)

        # generate uniform distribution
        trhd = torch.log(torch.rand(N_samples).to(device))
        # accept-reject indicator
        indic_acc = (log_prob_accept > trhd).float()

        indic_acc = indic_acc[:, None]

        if stats is not None:
            stats["n_accepts"] += indic_acc.sum().item()

        x_cur = indic_acc * x_next + (1 - indic_acc) * x_cur
    return x_cur.detach()


def mala(log_target_dens, x0, N_steps, gamma, mala_iters, stats=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ### sample i-sir

    samples_traj = [x0]
    x_cur = x0

    for _ in tqdm.tqdm(range(N_steps)):
        x_cur = mala_step(log_target_dens, x_cur, gamma, mala_iters, stats=stats)
        samples_traj.append(x_cur)

    samples_traj = torch.stack(samples_traj).transpose(0, 1)

    return samples_traj


def i_sir_step(log_target_dens, x_cur, N_part, isir_proposal, return_all_stats=False):
    """
    function to sample with N-particles version of i-SIR
    args:
        N_part - number of particles, integer;
        x0 - current i-sir particle;
    return:
        x_next - selected i-sir particle
    """
    N_samples, lat_size = x_cur.shape

    # generate proposals
    proposals = isir_proposal.sample(
        (
            N_samples,
            N_part - 1,
        )
    )

    # put current particles
    proposals = torch.cat((x_cur[:, None, :], proposals), dim=1)

    # compute importance weights
    log_target_dens_proposals = log_target_dens(
        proposals.reshape(-1, lat_size)
    ).reshape(N_samples, N_part)

    logw = log_target_dens_proposals - isir_proposal.log_prob(proposals)

    # sample selected particle indexes
    idxs = Categorical(logits=logw).sample()

    cur_x = proposals[torch.arange(N_samples), idxs]

    cur_x = cur_x.detach()
    proposals = proposals.detach()
    log_target_dens_proposals = log_target_dens_proposals.detach()

    if return_all_stats:
        return cur_x, proposals, log_target_dens_proposals
    else:
        return cur_x


def mh_step(log_target_dens, x_cur, mh_proposal):
    """
    function to sample with N-particles version of i-SIR
    args:
        N_part - number of particles, integer;
        x0 - current i-sir particle;
    return:
        x_next - selected i-sir particle
    """
    N_samples, lat_size = x_cur.shape
    device = x_cur.device
    # generate proposals
    proposals = mh_proposal.sample((N_samples,))
    # put current particles
    log_target_cur = log_target_dens(x_cur)
    log_target_proposal = log_target_dens(proposals)
    # compute proposal density
    log_prop_cur = mh_proposal.log_prob(x_cur)
    log_prop_prop = mh_proposal.log_prob(proposals)
    # compute acceptance ratio
    log_prob_accept = (
        log_target_proposal + log_prop_cur - log_prop_prop - log_target_cur
    )
    log_prob_accept = torch.clamp(log_prob_accept, max=0.0)
    # generate uniform distribution
    trhd = torch.log(torch.rand(N_samples).to(device))
    # accept-reject indicator
    indic_acc = (log_prob_accept > trhd).float()
    indic_acc = indic_acc[:, None]
    cur_x = indic_acc * proposals + (1 - indic_acc) * x_cur
    return cur_x.detach()


def i_sir(log_target_dens, x0, N_steps, N_part, isir_proposal, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ### sample i-sir

    samples_traj = [x0]
    x_cur = x0

    for _ in tqdm.tqdm(range(N_steps)):
        x_cur = i_sir_step(log_target_dens, x_cur, N_part, isir_proposal)
        samples_traj.append(x_cur)

    samples_traj = torch.stack(samples_traj).transpose(0, 1)

    return samples_traj


def ex2mcmc(
    log_target_dens,
    x0,
    N_steps,
    N_part,
    isir_proposal,
    gamma,
    mala_iters,
    stats=None,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ### sample i-sir

    samples_traj = [x0]
    x_cur = x0

    for _ in tqdm.tqdm(range(N_steps)):
        x_cur = i_sir_step(log_target_dens, x_cur, N_part, isir_proposal)
        x_cur = mala_step(log_target_dens, x_cur, gamma, mala_iters, stats=stats)
        samples_traj.append(x_cur)

    samples_traj = torch.stack(samples_traj).transpose(0, 1)

    return samples_traj


def flex2mcmc(
    log_target_dens,
    x0,
    N_steps,
    N_part,
    isir_proposal,
    gamma,
    mala_iters,
    alpha=1.0,
    stats=None,
    seed=42,
):
    """
    Function to perform Flex2MCMC by optimizing alpha * KL_FW + (1-alpha)*KL_BACK
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    ### sample i-sir
    samples_traj = [x0]
    x_cur = x0
    proposal_opt = torch.optim.Adam(isir_proposal.parameters(), lr=1e-3)

    pbar = tqdm.tqdm(range(N_steps))

    for _ in pbar:
        x_cur, proposals, log_target_dens_proposals = i_sir_step(
            log_target_dens, x_cur, N_part, isir_proposal, return_all_stats=True
        )
        # train proposal
        proposal_opt.zero_grad()
        population_log_proposal_prob = isir_proposal.log_prob(proposals)
        logw = (log_target_dens_proposals - population_log_proposal_prob).detach()

        kl_forw = -(
            (population_log_proposal_prob * torch.softmax(logw, dim=-1)).sum(axis=-1)
        ).mean()

        # backward KL
        cur_samples_x = proposals[:, 1:, :].reshape(-1, proposals.shape[-1])
        cur_samples_z, log_det_J = isir_proposal.f(cur_samples_x)

        cur_samples_x_diff = isir_proposal.g(cur_samples_z.detach())[0]
        log_target_dens_proposals_diff = log_target_dens(
            cur_samples_x_diff, detach=False
        )

        kl_back = -(log_target_dens_proposals_diff - log_det_J).mean()

        # opt step
        loss = alpha * kl_forw + (1 - alpha) * kl_back
        loss.backward()
        proposal_opt.step()

        # perform MALA update
        x_cur = mala_step(log_target_dens, x_cur, gamma, mala_iters, stats=stats)
        samples_traj.append(x_cur)

        pbar.set_description(f"KL forw {kl_forw.item()}, KL back {kl_back.item()}")

    samples_traj = torch.stack(samples_traj).transpose(0, 1)
    return samples_traj
