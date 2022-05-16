import copy
from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from .adaptive_sir_loss import get_loss
from .ebm_sampling import MALATransition, MHKernel, grad_energy
from .mcmc_base import AbstractMCMC, adapt_stepsize_dec, increment_steps


class AugmentedULA(AbstractMCMC):
    def __init__(self, flow=None, ula_steps=1, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.flow = flow
        self.ula_steps = ula_steps
        self.beta = beta
        self.grad_step = kwargs.get("grad_step", 1e-2)
        self.noise_scale = kwargs.get(
            "noise_scale",
            (2 * self.grad_step) ** 0.5,
        )

    def __call__(
        self,
        start,
        target,
        proposal,
        n_steps=1,
        flow=None,
        ula_steps=None,
        beta=None,
        grad_step=None,
        noise_scale=None,
        **kwargs,
    ):
        flow = flow if flow is not None else self.flow
        ula_steps = ula_steps if ula_steps is not None else self.ula_steps
        beta = beta if beta is not None else self.beta
        grad_step = grad_step if grad_step is not None else self.grad_step
        noise_scale = (
            noise_scale if noise_scale is not None else self.noise_scale
        )

        zs = []
        batch_size = start.shape[0]
        device = start.device
        acc_rate = torch.zeros(batch_size)

        if flow is not None:
            z, _ = flow(start)
        else:
            z = start

        for step_id in range(n_steps):
            with torch.no_grad():
                x_new = proposal.sample((batch_size,))
                if flow is not None:
                    z_new, log_jac = flow(x_new)
                    x, minus_log_jac = flow.inverse(z)
                    log_prop1 = proposal(x) + minus_log_jac
                    log_prop2 = proposal(x_new) - log_jac
                else:
                    pass
                log_tar1 = beta * target(z)
                log_tar2 = beta * target(z_new)
                mask = MHKernel.get_mask(
                    log_tar1,
                    log_tar2,
                    log_prop2,
                    log_prop1,
                )
                z[mask] = z_new[mask].detach().clone()
                z = z.data
                acc_rate += mask

            for _ in range(ula_steps):
                _, grad = grad_energy(z, target)
                z = (
                    z
                    - grad_step * beta * grad
                    + noise_scale * torch.randn(grad.shape, device=device)
                )
            zs.append(z.detach().clone())

        acc_rate /= n_steps

        return zs, acc_rate


class CorrelatedKernel:
    def __init__(self, corr_coef=0, bernoulli_prob_corr=0):
        self.corr_coef = corr_coef
        self.bern = torch.distributions.Bernoulli(bernoulli_prob_corr)

    def __call__(self, z, proposal, N, batch_size=1):
        correlation = self.corr_coef * self.bern.sample((batch_size,)) # (batch_size,)

        latent_var = correlation[:, None] * z + (
            1.0 - correlation[:, None] ** 2
        ) ** 0.5 * proposal.sample(
            (batch_size,),
        ) # (batch_size, z_dim)
        correlation_new = self.corr_coef * self.bern.sample((batch_size, N)) # (batch_size, N)
        z_new = correlation_new[..., None] * latent_var[:, None, :] + (
            1.0 - correlation_new[..., None] ** 2
        ) ** 0.5 * proposal.sample(
            (
                batch_size,
                N,
            ),
        ) # (batch_size, N, z_dim)
        return z_new


def compute_sir_log_weights(x, target, proposal, flow, beta=1.0):
    x_pushed, log_jac = flow(x)
    log_weights = beta * target(x_pushed) + log_jac - proposal(x)
    return log_weights, x_pushed


def adaptive_sir_correlated_dynamics(
    z,
    target,
    proposal,
    n_steps,
    N,
    corr_coef=0.0,
    bernoulli_prob_corr=0.0,
    flow=None,
    verbose=True,
):  # z assumed from proposal !
    z_sp = []  # vector of samples
    batch_size, z_dim = z.shape[0], z.shape[1]
    acceptance = torch.zeros(batch_size)

    range_gen = trange if verbose else range

    corr_ker = CorrelatedKernel(corr_coef, bernoulli_prob_corr)

    for _ in range_gen(n_steps):
        if flow is not None:
            z_pushed, _ = flow(z)
        else:
            z_pushed = z

        z_sp.append(z_pushed)

        X = corr_ker(z, proposal, N, batch_size)

        X[np.arange(batch_size), [0] * batch_size, :] = z
        X_view = X.view(-1, z_dim)

        if flow is not None:
            log_weight, z_pushed = compute_sir_log_weights(
                X_view,
                target,
                proposal,
                flow,
            )
        else:
            z_pushed = X_view
            log_weight = target(z_pushed) - proposal(z_pushed)

        log_weight = log_weight.view(batch_size, N)
        max_logs = torch.max(log_weight, dim=1)[0][:, None]
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim=1)
        weight = weight / sum_weight[:, None]

        weight[weight != weight] = 0.0
        weight[weight.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weight, 1).squeeze().tolist()
        mask = indices == 0
        acceptance += mask

        z = X[np.arange(batch_size), indices, :]

    if flow is not None:
        z_pushed, _ = flow(z)
    else:
        z_pushed = z

    z_sp.append(z_pushed)
    acceptance /= n_steps

    return z_sp, acceptance


class CISIR(AbstractMCMC):
    def __init__(
        self,
        N=2,
        corr_coef=0.0,
        bernoulli_prob_corr=0.0,
        flow=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__()
        self.N = N
        self.corr_coef = corr_coef
        self.bernoulli_prob_corr = bernoulli_prob_corr
        self.flow = flow
        self.verbose = verbose
        self.n_steps = kwargs.get("n_steps", 1)  #

    # @increment_steps
    # @adapt_stepsize_dec
    def __call__(
        self,
        start: torch.Tensor,
        target,
        proposal,
        n_steps=None,
        N=None,
        corr_coef=None,
        bernoulli_prob_corr=None,
        flow=None,
        verbose=None,
    ):
        N = self.N if N is None else N
        corr_coef = self.corr_coef if corr_coef is None else corr_coef
        bernoulli_prob_corr = (
            self.bernoulli_prob_corr
            if bernoulli_prob_corr is None
            else bernoulli_prob_corr
        )
        flow = self.flow if flow is None else flow
        n_steps = self.n_steps if n_steps is None else n_steps
        verbose = self.verbose if verbose is None else verbose

        return adaptive_sir_correlated_dynamics(
            start,
            target,
            proposal,
            n_steps,
            N,
            corr_coef,
            bernoulli_prob_corr,
            flow,
            verbose,
        )
    
def ex2_mcmc_mala_new_proposal(
    z,
    target,
    proposal,
    n_steps,
    N,
    grad_step,
    noise_scale,
    beta=1.0,
    mala_steps=5,
    corr_coef=0.0,
    bernoulli_prob_corr=0.0,
    flow=None,
    adapt_stepsize=True,
    verbose=False,
    ind_chains=True
):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    mala_transition = MALATransition(z_dim, z.device)
    mala_transition.adapt_grad_step = grad_step
    mala_transition.adapt_sigma = noise_scale

    corr_ker = CorrelatedKernel(corr_coef, bernoulli_prob_corr)

    if ind_chains:
        acceptance = torch.zeros(batch_size)
    else:
        acceptance = torch.zeros(batch_size * N)

    range_gen = trange if verbose else range

    if flow is not None:
        z_pushed, log_jac = flow(z)
    else:
        z_pushed = z

    for step_id in range_gen(n_steps):
        # z_sp.append(z_pushed.detach().cpu())

        if corr_coef == 0 and bernoulli_prob_corr == 0:
            X = proposal.sample([batch_size, N])
        else:
            X = corr_ker(z, proposal, N, batch_size)

        ind = [0] * batch_size
        X[np.arange(batch_size), ind, :] = z

        X_view = X.view(-1, z_dim)


        if flow is not None:
            log_weight, z_pushed = compute_sir_log_weights(
                X_view,
                target,
                proposal,
                flow,
                beta=beta,
            )
        else:
            z_pushed = X_view
            log_weight = beta * target(X_view) - proposal(X_view)

        log_weight = log_weight.view(batch_size, N)
        z_pushed = z_pushed.view(X.shape)

        max_logs = torch.max(log_weight, dim=1)[0][:, None]
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim=1)
        weight = weight / sum_weight[:, None]

        weight[weight != weight] = 0.0
        weight[weight.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weight, 1).squeeze().tolist()

        if ind_chains:
            z_pushed = z_pushed[np.arange(batch_size), indices, :]
        else:
            z_pushed = z_pushed[
                np.arange(batch_size), indices, :
                ][:, None, :].repeat(1, N, 1).reshape(batch_size * N, z_dim) #z_pushed.reshape(batch_size * N, z_dim)

        if step_id != n_steps - 1:
            if flow is not None:
                z, _ = flow.inverse(z_pushed)
            else:
                z = z_pushed
        
        for _ in range(mala_steps):
            #compute values of energy and gradient of the energy function
            E, grad = grad_energy(z_pushed, target)
            z_pushed, _, _, mask = mala_transition(
                z_pushed,
                E,
                grad,
                target=target,
                adapt_stepsize=adapt_stepsize,
                beta=beta,
            )
            acceptance += mask.float() / mala_steps

        #if not ind_chains:
        #    z = z_pushed.reshape(batch_size, N, z_dim)[np.arange(batch_size), indices, :]

        if not ind_chains:
           z = z_pushed.reshape(batch_size, N, z_dim)[np.arange(batch_size), 0, :]

        z_sp.append(z_pushed.detach().cpu())

    # z_sp.append(z_pushed.detach().cpu())
    acceptance /= n_steps

    return z_sp, acceptance, mala_transition.adapt_grad_step


def ex2_mcmc_mala(
    z,
    target,
    proposal,
    n_steps,
    N,
    grad_step,
    noise_scale,
    beta=1.0,
    mala_steps=5,
    corr_coef=0.0,
    bernoulli_prob_corr=0.0,
    device = 'cpu',
    flow=None,
    adapt_stepsize=True,
    verbose=False,
    ind_chains=True
):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    mala_transition = MALATransition(z_dim, z.device)
    mala_transition.adapt_grad_step = grad_step
    mala_transition.adapt_sigma = noise_scale

    corr_ker = CorrelatedKernel(corr_coef, bernoulli_prob_corr)

    if ind_chains:
        acceptance = torch.zeros(batch_size)
    else:
        acceptance = torch.zeros(batch_size * N)

    range_gen = trange if verbose else range

    if flow is not None:
        z_pushed, log_jac = flow(z)
    else:
        z_pushed = z

    for step_id in range_gen(n_steps):
        # z_sp.append(z_pushed.detach().cpu())

        if corr_coef == 0 and bernoulli_prob_corr == 0:
            X = proposal.sample([batch_size, N])
        else:
            X = corr_ker(z, proposal, N, batch_size)

        ind = [0] * batch_size
        X[np.arange(batch_size), ind, :] = z

        X_view = X.view(-1, z_dim)


        if flow is not None:
            log_weight, z_pushed = compute_sir_log_weights(
                X_view,
                target,
                proposal,
                flow,
                beta=beta,
            )
        else:
            z_pushed = X_view
            log_weight = beta * target(X_view) - proposal(X_view)

        log_weight = log_weight.view(batch_size, N)
        z_pushed = z_pushed.view(X.shape)

        max_logs = torch.max(log_weight, dim=1)[0][:, None]
        log_weight = log_weight - max_logs
        weight = torch.exp(log_weight)
        sum_weight = torch.sum(weight, dim=1)
        weight = weight / sum_weight[:, None]

        weight[weight != weight] = 0.0
        weight[weight.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weight, 1).squeeze().tolist()

        if ind_chains:
            z_pushed = z_pushed[np.arange(batch_size), indices, :]
        else:
            z_pushed = z_pushed[
                np.arange(batch_size), indices, :
                ][:, None, :].repeat(1, N, 1).reshape(batch_size * N, z_dim) #z_pushed.reshape(batch_size * N, z_dim)
        
        for _ in range(mala_steps):
            E, grad = grad_energy(z_pushed, target)
            z_pushed, _, _, mask = mala_transition(
                z_pushed,
                E,
                grad,
                target=target,
                adapt_stepsize=adapt_stepsize,
                beta=beta,
            )
            acceptance += mask.cpu().float() / mala_steps
            
        if step_id != n_steps - 1:
            if flow is not None:
                z, _ = flow.inverse(z_pushed)
            else:
                z = z_pushed

        #if not ind_chains:
        #    z = z_pushed.reshape(batch_size, N, z_dim)[np.arange(batch_size), indices, :]

        if not ind_chains:
           z = z_pushed.reshape(batch_size, N, z_dim)[np.arange(batch_size), 0, :]

        z_sp.append(z_pushed.detach().cpu())
        gc.collect()
        torch.cuda.empty_cache()

    # z_sp.append(z_pushed.detach().cpu())
    acceptance /= n_steps

    return z_sp, acceptance, mala_transition.adapt_grad_step


class Ex2MCMC(AbstractMCMC):
    def __init__(
        self,
        N=2,
        grad_step=1e-2,
        noise_scale=None,
        beta=1.0,
        mala_steps=5,
        corr_coef=0.0,
        bernoulli_prob_corr=0.0,
        device = 'cpu',
        flow=None,
        adapt_stepsize=False,
        verbose=True,
        ind_chains=True,
        **kwargs,
    ):
        super().__init__()
        self.N = N
        self.grad_step = grad_step
        self.noise_scale = (
            (2 * grad_step) ** 0.5 if noise_scale is None else noise_scale
        )
        self.mala_steps = mala_steps
        self.corr_coef = corr_coef
        self.bernoulli_prob_corr = bernoulli_prob_corr
        self.device = device
        self.flow = flow
        self.adapt_stepsize = adapt_stepsize
        self.verbose = verbose
        self.ind_chains = ind_chains
        self.n_steps = kwargs.get("n_steps", 1)  #
        self.beta = beta

    @increment_steps
    @adapt_stepsize_dec
    def __call__(self, start: torch.Tensor, target, proposal, *args, **kwargs):
        self_kwargs = copy.copy(self.__dict__)
        self_kwargs.update(kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        n_steps = self_kwargs.pop("n_steps")
        if len(args) > 0:
            n_steps = args[0]

        self_kwargs.pop("_steps_done")

        return ex2_mcmc_mala(start, target, proposal, n_steps, **self_kwargs)


class FlowMCMC:
    def __init__(self, target, proposal, device, flow, mcmc_call: callable, **kwargs):
        self.flow = flow
        self.proposal = proposal
        self.target = target
        self.device = device
        self.batch_size = kwargs.get("batch_size", 64)
        self.mcmc_call = mcmc_call
        self.grad_clip = kwargs.get("grad_clip", 1.0)
        self.jump_tol = kwargs.get("jump_tol", 1e6)
        optimizer = kwargs.get("optimizer", "adam")
        loss = kwargs.get("loss", "mix_kl")
        self.flow.to(self.device)
        if isinstance(loss, (Callable, nn.Module)):
            self.loss = loss
        elif isinstance(loss, str):
            self.loss = get_loss(loss)(self.target, self.proposal, self.flow)
        else:
            ValueError

        lr = kwargs.get("lr", 1e-3)
        wd = kwargs.get("wd", 1e-4)
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    flow.parameters(), lr=lr, weight_decay=wd
                )

        self.loss_hist = []

    def train_step(self, inp=None, alpha=0.5, do_step=True, inv=True):
        if do_step:
            self.optimizer.zero_grad()
        if inp is None:
            inp = self.proposal.sample((self.batch_size,))
        elif inv:
            inp, _ = self.flow.inverse(inp)
        #print("before mcmc call")
        out = self.mcmc_call(inp, self.target, self.proposal, flow=self.flow)
        if isinstance(out, Tuple):
            acc_rate = out[1].mean()
            out = out[0]
        else:
            acc_rate = 1
        out = out[-1]
        out = out.to(self.device)
        nll = -self.target(out).mean().item()

        if do_step:
            loss_est, loss = self.loss(out, acc_rate=acc_rate, alpha=alpha)

            if (
                len(self.loss_hist) > 0
                and loss.item() - self.loss_hist[-1] > self.jump_tol
            ) or torch.isnan(loss):
                print("KL wants to jump, terminating learning")
                return out, nll

            self.loss_hist = self.loss_hist[-500:] + [loss_est.item()]
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.flow.parameters(),
                self.grad_clip,
            )
            self.optimizer.step()

        return out, nll

    def train(self, n_steps=100, start_optim=10, init_points=None, alpha=None):
        samples = []
        inp = self.proposal.sample((self.batch_size,))

        neg_log_likelihood = []

        for step_id in trange(n_steps):
            #if alpha is not None:
            #    if isinstance(alpha, Callable):
            #        a = alpha(step_id)
            #    elif isinstance(alpha, float):
            #        a = alpha
            #else:
            a = min(0.5, 3 * step_id / n_steps)
            #print("before train step")
            out, nll = self.train_step(
                alpha=a,
                do_step=step_id >= start_optim,
                inp=init_points
                if step_id == 0 and init_points is not None
                else inp,
                inv=True,
            ) 
            inp = out.detach().requires_grad_()
            samples.append(inp.detach().cpu())

            neg_log_likelihood.append(nll)

        return samples, neg_log_likelihood

    def sample(self):
        pass
