import numpy as np
import torch
from pyro.infer import HMC, MCMC, NUTS
import pyro


def discretesampling(w):
    u = np.random.rand()
    bins = np.cumsum(w)
    return np.digitize(u,bins)

def grad_log_target_dens(log_target_dens, z):
    """
    returns the gradient of log-density 
    """
    x.requires_grad_(True)
    external_grad = torch.ones(x.shape[0])
    (log_target_dens(x)).backward(gradient=external_grad)
    return x.grad.data.detach().cpu().numpy()


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

def mala(log_target_dens, grad_log_target_dens, logp_mala, x0, gamma, n, n_accepts=None):
    """
    function to perform n times MALA step 
    """
    N_traj, lat_size = x0.shape
    #generate proposals
    x_cur = x0
    for i in range(n):
        y = x_cur + gamma*grad_log_target_dens(x_cur) + np.sqrt(2*gamma)*np.random.randn(N_traj, lat_size)
        #compute accept-reject
        log_prob_accept = np.minimum(0.0, log_target_dens(y) + logp_mala(grad_log_target_dens, x_cur,y,gamma) - log_target_dens(x_cur) - logp_mala(grad_log_target_dens, y,x_cur,gamma))
        
        #generate uniform distribution
        unif = np.log(np.random.uniform(size=N_traj))
        indic_acc = (log_prob_accept > unif)
        indic_acc = indic_acc[:, None]
        
        if n_accepts is not None:
            n_accepts += indic_acc.sum()
        x_cur = indic_acc*y + (1-indic_acc)*x_cur
    return x_cur

def i_sir(log_target_dens, log_dens_isir, x0, N_part, sigma_isir):
    """
    function to sample with N-particles version of i-SIR
    args:
        N_part - number of particles, integer;
        x0 - current i-sir particle;
    return:
        x_next - selected i-sir particle
    """
    N_traj, lat_size = x0.shape
    
    #generate proposals
    proposals = sigma_isir*np.random.randn(N_traj, N_part, lat_size)
    #put current particles
    proposals[:, 0, :] = x0
    #compute importance weights
    logw = log_target_dens(proposals.reshape(-1, lat_size)) - log_dens_isir(proposals.reshape(-1, lat_size), sigma_isir)
    logw = logw.reshape(N_traj, N_part)
    
    idxs = []
    for i in range(N_traj):
        maxLogW = np.max(logw[i])
        uw = np.exp(logw[i]-maxLogW)
        w = uw / np.sum(uw)
        #sample selected index
        idx = discretesampling(w)
        #retur seleted particle
        idxs.append(idx)

    return proposals[np.arange(N_traj), idxs]

def ex2_mcmc(log_target_dens, grad_log_target_dens, logp_mala, log_dens_isir, x0,N_part,sigma_isir,gamma,n_steps_mala, n_accepts=None):
    """
    function to sample with N-particles vrsion of i-SIR using MALA as a rejuvenation kernel with step size gamma
    """
    N_traj, lat_size = x0.shape
    
    #generate proposals
    proposals = sigma_isir*np.random.randn(N_traj, N_part, lat_size)
    #put current particles
    proposals[:, 0, :] = x0
    #compute importance weights
    logw = log_target_dens(proposals.reshape(-1, lat_size)) - log_dens_isir(proposals.reshape(-1, lat_size), sigma_isir)
    logw = logw.reshape(N_traj, N_part)
    
    idxs = []
    for i in range(N_traj):
        maxLogW = np.max(logw[i])
        uw = np.exp(logw[i]-maxLogW)
        w = uw / np.sum(uw)
        #sample selected index
        idx = discretesampling(w)
        #retur seleted particle
        idxs.append(idx)
        
    #return seleted particle
    x_new = proposals[np.arange(N_traj), idxs]
    #perform rejuvenation step
    x_rej = mala(log_target_dens, grad_log_target_dens, logp_mala, x_new,gamma,n_steps_mala, n_accepts)
    return x_rej
