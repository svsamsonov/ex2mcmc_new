import numpy as np


import torch
from torch.distributions import MultivariateNormal, Categorical

import pyro
from pyro.infer import HMC, MCMC, NUTS

import tqdm

# Flows

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

class RealNVPProposal(nn.Module):
    def __init__(self, lat_size, device, hidden=32, num_blocks=4):
        super(RealNVPProposal, self).__init__()
        
        self.prior = MultivariateNormal(
            torch.zeros(lat_size).to(device), 
            torch.eye(lat_size).to(device))
        
        masks = num_blocks * [[i % 2 for i in range(lat_size)], [(i + 1) % 2 for i in range(lat_size)]]
        masks = torch.FloatTensor(masks)
        self.masks = nn.Parameter(masks, requires_grad=False)

        self.t = torch.nn.ModuleList([nn.Sequential(
            nn.Linear(lat_size, hidden), nn.LeakyReLU(), 
            nn.Linear(hidden, hidden), nn.LeakyReLU(), 
            nn.Linear(hidden, lat_size)) 
                                      for _ in range(2 * num_blocks)])
        self.s = torch.nn.ModuleList([nn.Sequential(
            nn.Linear(lat_size, hidden), nn.LeakyReLU(), 
            nn.Linear(hidden, hidden), nn.LeakyReLU(), 
            nn.Linear(hidden, lat_size), nn.Tanh()) 
                                      for _ in range(2 * num_blocks)])

    def g_flatten(self, z):
        log_det_J_inv, x = z.new_zeros(z.shape[0]), z
        for i in range(len(self.t)):
            x_ = x*self.masks[i]
            s = self.s[i](x_)*(1 - self.masks[i])
            t = self.t[i](x_)*(1 - self.masks[i])
            x = x_ + (1 - self.masks[i]) * (x * torch.exp(s) + t)

            log_det_J_inv += s.sum(dim=1)
        return x, log_det_J_inv

    def g(self, z):
        lat_size = z.shape[-1]
        first_dims = z.shape[:-1]

        x, log_det_J_inv = self.g_flatten(z.reshape(-1, lat_size))
        x = x.reshape(*z.shape)
        log_det_J_inv = log_det_J_inv.reshape(*first_dims)
        
        return x, log_det_J_inv
    
    def f_flatten(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.masks[i] * z
            s = self.s[i](z_) * (1-self.masks[i])
            t = self.t[i](z_) * (1-self.masks[i])
            z = (1 - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def f(self, x):
        lat_size = x.shape[-1]
        first_dims = x.shape[:-1]

        z, log_det_J = self.f_flatten(x.reshape(-1, lat_size))
        z = z.reshape(*x.shape)
        log_det_J = log_det_J.reshape(*first_dims)
        
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, num_samples):
        if isinstance(num_samples, int):
            num_samples = (num_samples, )
        z = self.prior.sample(num_samples)
        x = self.g(z)[0]
        return x

# Samplers

# NUTS

def sample_nuts(target, num_samples=1000, burn_in=1000, batch_size=1, lat_size = 100):
    def true_target_energy(z):
        return target(z)

    def energy(z):
        z = z["points"]
        return true_target_energy(z)

    # kernel = HMC(potential_fn=energy, step_size = 0.1, num_steps = K, full_mass = False)
    kernel_true = NUTS(potential_fn=energy, full_mass=False)
    #kernel_true = HMC(potential_fn=energy, full_mass=False)
    pyro.set_rng_seed(45)
    init_samples = torch.FloatTensor(np.random.randn(batch_size,lat_size))
    print(init_samples.shape) 
    #init_samples = torch.zeros_like(init_samples)
    dim = init_samples.shape[-1]
    init_params = {"points": init_samples}
    mcmc_true = MCMC(
        kernel=kernel_true,
        num_samples=num_samples,
        initial_params=init_params,
        warmup_steps=burn_in,
    )
    mcmc_true.run()
    

    q_true = mcmc_true.get_samples(group_by_chain=True)["points"]
    samples_true = np.array(q_true.view(-1, batch_size, dim))

    return samples_true

# MALA

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
    function to perform n times MALA step 
    """
    N_samples, lat_size = x0.shape
    device = x0.device
    
    #generate proposals
    x_cur = x0
    for i in range(mala_iters):
        x_cur_grad, x_cur_log_target = grad_log_and_output_target_dens(log_target_dens, x_cur)
        x_cur_proposal = MultivariateNormal(x_cur + gamma * x_cur_grad, 2 * gamma * torch.eye(lat_size)[None, :, :].repeat(N_samples, 1, 1).to(device))
        
        x_next = x_cur_proposal.sample()     

        x_next_grad, x_next_log_target = grad_log_and_output_target_dens(log_target_dens, x_next)
        x_next_proposal = MultivariateNormal(x_next + gamma * x_next_grad, 2 * gamma * torch.eye(lat_size)[None, :, :].repeat(N_samples, 1, 1).to(device))

        #compute accept-reject
        log_prob_accept = x_next_log_target + x_next_proposal.log_prob(x_cur) - x_cur_log_target - x_cur_proposal.log_prob(x_next)
        log_prob_accept = torch.clamp(log_prob_accept, max=0.0)
        
        #generate uniform distribution
        trhd = torch.log(torch.rand(N_samples).to(device))
        indic_acc = (log_prob_accept > trhd).float()

        indic_acc = indic_acc[:, None]
        
        if stats is not None:
            stats['n_accepts'] += indic_acc.sum().item()

        x_cur = indic_acc * x_next + (1-indic_acc) * x_cur
    return x_cur.detach()

def mala(log_target_dens, x0, N_steps, gamma, mala_iters, stats=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ### sample i-sir

    samples_traj = [x0]
    x_cur = x0
    
    for _ in tqdm.tqdm(range(N_steps)):
        x_cur =  mala_step(log_target_dens, x_cur, gamma, mala_iters, stats=stats)
        samples_traj.append(x_cur)

    samples_traj = torch.stack(samples_traj).transpose(0, 1)
    
    return samples_traj

# I-SIR

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
    proposals = isir_proposal.sample((N_samples, N_part - 1, ))
    
    # put current particles
    proposals = torch.cat((x_cur[:, None, :], proposals), dim=1)

    # compute importance weights
    log_target_dens_proposals = log_target_dens(proposals.reshape(-1, lat_size)).reshape(N_samples, N_part)
    
    logw = log_target_dens_proposals - isir_proposal.log_prob(proposals)
    
    #sample selected particle indexes
    idxs = Categorical(logits=logw).sample()

    cur_x = proposals[torch.arange(N_samples), idxs]
    
    cur_x = cur_x.detach()
    proposals = proposals.detach()
    log_target_dens_proposals = log_target_dens_proposals.detach()
    
    
    if return_all_stats:
        return cur_x, proposals, log_target_dens_proposals
    else:
        return cur_x

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

# Ex2MCMC

def ex2_mcmc(log_target_dens, x0, N_steps, N_part, isir_proposal ,gamma, mala_iters, stats=None, seed=42):
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

def flex2_mcmc(log_target_dens, x0, N_steps, N_part, isir_proposal, gamma, mala_iters, add_pop_size_train=4096, stats=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ### sample i-sir
    samples_traj = [x0]
    x_cur = x0
    proposal_opt = torch.optim.Adam(isir_proposal.parameters(), lr=1e-3)
    
    pbar = tqdm.tqdm(range(N_steps))
    
    hist_proposals = None
    hist_log_target_dens_proposals = None
    
    for _ in pbar:
        x_cur, proposals, log_target_dens_proposals = i_sir_step(log_target_dens, x_cur, N_part, isir_proposal, return_all_stats=True)
        x_cur = mala_step(log_target_dens, x_cur, gamma, mala_iters, stats=stats)
        samples_traj.append(x_cur)
        
        # train proposal
        proposal_opt.zero_grad()
        
        # forward KL
        proposals_flattened = proposals.reshape(-1, proposals.shape[-1])
        log_target_dens_proposals_flattened = log_target_dens_proposals.reshape(-1)    
        
        population_proposals = proposals_flattened
        population_log_target_dens_proposals = log_target_dens_proposals_flattened
        
        if hist_proposals is not None:
            idxs = np.random.permutation(hist_proposals.shape[0])[:add_pop_size_train]
            population_proposals = torch.cat((population_proposals, hist_proposals[idxs]))
            population_log_target_dens_proposals = torch.cat((population_log_target_dens_proposals, hist_log_target_dens_proposals[idxs]))

        population_log_proposal_prob = isir_proposal.log_prob(population_proposals)
        logw = population_log_target_dens_proposals - population_log_proposal_prob
        
        kl_forw = -(population_log_proposal_prob * torch.softmax(logw, dim=-1)).sum()
        
        # backward KL
        cur_samples_x = proposals[:, 1:, :].reshape(-1, proposals.shape[-1])
        cur_samples_z, log_det_J = isir_proposal.f(cur_samples_x)
        
        cur_samples_x_diff = isir_proposal.g(cur_samples_z.detach())[0]
        log_target_dens_proposals_diff = log_target_dens(cur_samples_x_diff, detach=False)
        
        kl_back = -(log_target_dens_proposals_diff + log_det_J).mean()
        
        # entropy reg
        e = -isir_proposal.log_prob(torch.randn_like(proposals_flattened)).mean()
        
        # opt step
        loss = kl_forw + 0.1*kl_back + 0.1 * e
        loss.backward()
        proposal_opt.step()
        
        pbar.set_description(f"KL forw {kl_forw.item()}, KL back {kl_back.item()} Hentr {e.item()}")

        if hist_proposals is None:
            hist_proposals = proposals_flattened
            hist_log_target_dens_proposals = log_target_dens_proposals_flattened
        else:
            hist_proposals = torch.cat((hist_proposals, proposals_flattened), dim=0)
            hist_log_target_dens_proposals = torch.cat((hist_log_target_dens_proposals, log_target_dens_proposals_flattened), dim=0)
        
    samples_traj = torch.stack(samples_traj).transpose(0, 1)
    
    return samples_traj