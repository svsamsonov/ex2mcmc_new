import numpy as np
import ot
import torch
from easydict import EasyDict as edict
from scipy.stats import chi2


class Evolution:
    def __init__(
        self,
        target_sample,
        **kwargs,
    ):
        self.locs = kwargs.get("locs", None)
        self.sigma = kwargs.get("sigma", None)
        self.target_sample = target_sample
        self.scaler = kwargs.get("scaler", None)
        self.q = kwargs.get("q", 0.999)

        self.mode_std = []
        self.high_quality_rate = []
        self.jsd = []
        self.emd = []
        self.n_found_modes = []

        self.kl_pis = []
        self.js_pis = []
        self.l2_div = []

    @staticmethod
    def make_assignment(
        X_gen,
        locs,
        sigma=0.05,
        q=0.95,
    ):
        n_modes, x_dim = locs.shape
        dists = torch.norm((X_gen[:, None, :] - locs[None, :, :]), p=2, dim=-1)
        chi2_quantile = chi2.ppf(q, x_dim)
        test_dist = chi2_quantile**0.5 * sigma
        assignment_chi2 = dists < test_dist
        assignment_2sigma = dists < 2 * sigma
        return assignment_chi2, assignment_2sigma

    @staticmethod
    def compute_mode_std(X_gen, assignment):
        """
        X_gen(torch.FloatTensor) - (n_pts, x_dim)

        """
        x_dim = X_gen.shape[-1]
        n_modes = assignment.shape[1]
        std = torch.FloatTensor([0.0]).to(X_gen.device)
        found_modes = 0
        for mode_id in range(n_modes):
            xs = X_gen[assignment[:, mode_id]]
            # print(xs.shape)
            if xs.shape[0] > 1:
                std_ = (
                    1 / (x_dim * (xs.shape[0] - 1)) * ((xs - xs.mean(0)) ** 2).sum()
                ) ** 0.5
                std += std_
                found_modes += 1
        if found_modes != 0:
            std /= found_modes
        return std, found_modes

    @staticmethod
    def compute_mode_mean(X_gen, assignment):
        """
        X_gen(torch.FloatTensor) - (n_pts, x_dim)
        """
        x_dim = X_gen.shape[-1]
        n_modes = assignment.shape[1]
        means = torch.zeros((n_modes, x_dim))
        found_modes_ind = []
        for mode_id in range(n_modes):
            xs = X_gen[assignment[:, mode_id]]
            # print(xs.shape)
            if xs.shape[0] > 1:
                means[mode_id] = xs.mean(0)
                found_modes_ind.append(mode_id)

        return means, found_modes_ind

    @staticmethod
    def compute_high_quality_rate(assignment):
        high_quality_rate = assignment.max(1)[0].sum() / float(
            assignment.shape[0],
        )
        return high_quality_rate

    @staticmethod
    def compute_jsd(assignment):
        device = assignment.device
        n_modes = assignment.shape[1]
        assign_ = torch.cat(
            [
                assignment,
                torch.zeros(assignment.shape[0]).to(device).unsqueeze(1),
            ],
            -1,
        )
        assign_[:, -1][assignment.sum(1) == 0] = 1
        sample_dist = assign_.sum(dim=0) / float(assign_.shape[0])
        sample_dist /= sample_dist.sum()
        uniform_dist = torch.FloatTensor(
            [1.0 / n_modes for _ in range(n_modes)] + [0],
        ).to(assignment.device)
        M = 0.5 * (uniform_dist + sample_dist)
        # JSD = .5 * (sample_dist * torch.log((sample_dist + 1e-7) / M)) + .5 * (uniform_dist * torch.log((uniform_dist + 1e-7) / M))

        # JSD[sample_dist == 0.] = 0.
        # JSD[uniform_dist == 0.] = 0.
        # JSD = JSD.sum()

        KL1 = sample_dist * (torch.log(sample_dist) - torch.log(M))
        KL1[sample_dist == 0.0] = 0.0
        KL2 = uniform_dist * (torch.log(uniform_dist) - torch.log(M))
        KL2[uniform_dist == 0.0] = 0.0
        JSD = 0.5 * (KL1 + KL2).sum()

        return JSD

    @staticmethod
    def compute_emd(target_sample, gen_sample):
        gen_sample = gen_sample[
            np.random.choice(
                np.arange(gen_sample.shape[0]),
                target_sample.shape[0],
                replace=target_sample.shape[0] > gen_sample.shape[0],
            )
        ]
        M = ot.dist(target_sample, gen_sample)
        emd = ot.emd2([], [], M)

        return emd

    def invoke(self, X_gen, compute_discrim_div=False):
        emd = Evolution.compute_emd(
            self.target_sample,
            X_gen.detach().cpu().numpy(),
        )
        self.emd.append(emd)

        if self.locs is not None and self.sigma is not None:
            assignment = Evolution.make_assignment(
                X_gen,
                self.locs,
                self.sigma,
                self.q,
            )
            # print(assignment.shape)
            mode_std, found_modes = Evolution.compute_mode_std(
                X_gen,
                assignment,
            )
            self.n_found_modes.append(found_modes)
            # print(mode_std)
            self.mode_std.append(mode_std.item())
            h_q_r = Evolution.compute_high_quality_rate(assignment)
            self.high_quality_rate.append(h_q_r.item())
            jsd = Evolution.compute_jsd(assignment)
            self.jsd.append(jsd.item())

    def as_dict(self):
        d = dict(
            mode_std=self.mode_std,
            hqr=self.high_quality_rate,
            jsd=self.jsd,
            emd=self.emd,
            kl_pis=self.kl_pis,
            js_pis=self.js_pis,
            l2_div=self.l2_div,
            n_found_modes=self.n_found_modes,
        )
        return d


def autocovariance(X, tau=0):
    # dT, dX = np.shape(X)
    dT = X.shape[0]
    s = 0.0
    dN = 1
    if tau > 0:
        x1 = X[:-tau, ...]
    else:
        x1 = X
    x2 = X[tau:, ...]
    s = np.sum(x1 * x2, axis=0) / dN

    return s / (dT - tau)


def acl_spectrum(X, n=150, scale=None):
    scale = np.array(scale) if scale is not None else np.sqrt(autocovariance(X, tau=0))
    return np.stack(
        [autocovariance(X / (scale[None, ...] + 1e-7), tau=t) for t in range(n - 1)],
        axis=0,
    )


def ESS(A):
    # ess = ESS(acl_spectrum((trunc_sample - trunc_sample.mean(0)[None, ...]))).mean()
    A = A * (A > 0.05)
    ess = 1.0 / (1.0 + 2 * np.sum(A[1:, ...], axis=0))
    return ess


class MetricsTracker:
    stor: dict

    def __init__(self, fields):
        self.fields = fields
        self.stor = edict()
        for field in self.fields:
            self.stor[field] = []
