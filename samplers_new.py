import numpy as np


import torch
from torch.distributions import MultivariateNormal, Categorical

import pyro
from pyro.infer import HMC, MCMC, NUTS

import tqdm

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
    inp = z.clone()
    inp.requires_grad_(True)

    output = log_target_dens(inp, detach=False)
    output.sum().backward()

    grad = inp.grad.data.detach()

    return grad, output.data.detach()

def mala_step(log_target_dens, x0, gamma, mala_iters, n_accepts=None):
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
        
        if n_accepts is not None:
            n_accepts += indic_acc.sum().item()

        x_cur = indic_acc * x_next + (1-indic_acc) * x_cur
    return x_cur

def mala(log_target_dens, x0, N_steps, gamma, mala_iters, n_accepts=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ### sample i-sir

    samples_traj = [x0]
    x_cur = x0
    
    for _ in tqdm.tqdm(range(N_steps)):
        x_cur =  mala_step(log_target_dens, x_cur, gamma, mala_iters, n_accepts=None)
        samples_traj.append(x_cur)

    samples_traj = torch.stack(samples_traj).transpose(0, 1)
    
    return samples_traj

# I-SIR

def i_sir_step(log_target_dens, x_cur, N_part, isir_proposal):
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
    proposals = isir_proposal.sample((N_samples, N_part, ))
    
    # put current particles
    proposals[:, 0, :] = x_cur
    
    # compute importance weights
    logw = log_target_dens(proposals.reshape(-1, lat_size)) - isir_proposal.log_prob(proposals.reshape(-1, lat_size))
    logw = logw.reshape(N_samples, N_part)
    
    #sample selected particle indexes
    idxs = Categorical(logits=logw).sample()

    return proposals[torch.arange(N_samples), idxs]

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

def ex2_mcmc(log_target_dens, x0, N_steps, N_part, isir_proposal ,gamma,mala_iters, n_accepts=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ### sample i-sir

    samples_traj = [x0]
    x_cur = x0
    
    for _ in tqdm.tqdm(range(N_steps)):
        x_cur = i_sir_step(log_target_dens, x_cur, N_part, isir_proposal)
        x_cur = mala_step(log_target_dens, x_cur, gamma, mala_iters, n_accepts=None)
        samples_traj.append(x_cur)

    samples_traj = torch.stack(samples_traj).transpose(0, 1)
    
    return samples_traj