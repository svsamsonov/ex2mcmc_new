import argparse
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


sns.set_theme()

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("lines", linewidth=3)
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_res(log_path, config, arange):
    Path(log_path, "figs").mkdir(exist_ok=True)
    print(log_path)

    try:
        is_values = np.loadtxt(Path(log_path, "is_values.txt"))[:, 0]
        fig = plt.figure()
        plt.plot(np.arange(len(is_values)) * arange[1], is_values)
        plt.xlabel("Iteration")
        plt.ylabel("IS")
        plt.title("Inception Score")
        fig.tight_layout()
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_is.png"))
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_is.pdf"))
        plt.close()
    except Exception:
        print("is failed")

    try:
        fid_values = np.loadtxt(Path(log_path, "fid_values.txt"))
        fig = plt.figure()
        plt.plot(np.arange(len(fid_values)) * arange[1], fid_values)
        plt.xlabel("Iteration")
        plt.ylabel("FID")
        plt.title("FID Score")
        fig.tight_layout()
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_fid.png"))
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_fid.pdf"))
        plt.close()
    except Exception:
        print("fid failed")

    try:
        callback_results = np.loadtxt(Path(log_path, "callback_results.txt"))
        energy_results = callback_results[0]
        dgz_results = callback_results[1]
        fig = plt.figure()
        plt.plot(np.arange(len(energy_results)) * arange[1], energy_results)
        plt.xlabel("Iteration")
        plt.ylabel(r"$U(z)$")
        plt.title("Energy")
        fig.tight_layout()
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_energy.png"))
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_energy.pdf"))
        plt.close()
    except Exception:
        print("energy failed")

    try:
        callback_results = np.loadtxt(Path(log_path, "callback_results.txt"))
        energy_results = callback_results[0]
        dgz_results = callback_results[1]
        fig = plt.figure()
        plt.plot(arange, dgz_results)
        plt.xlabel("Iteration")
        plt.ylabel(r"$d(G(z))$")
        plt.axhline(
            config.thermalize[False]["real_score"],
            linestyle="--",
            label="avg real score",
            color="r",
        )
        plt.title("Discriminator scores")
        plt.legend()
        fig.tight_layout()
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_dgz.png"))
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_dgz.pdf"))
        plt.close()
    except Exception:
        print("dgz failed")

    if Path(log_path, "weight_norm.txt").exists():
        weight_norms = np.loadtxt(Path(log_path, "weight_norm.txt")).reshape(
            -1, len(arange)
        )
        mean = weight_norms.mean(0)
        std = weight_norms.std(0)
        fig = plt.figure()
        plt.plot(np.arange(len(mean)) * arange[1], mean)
        plt.fill_between(
            np.arange(len(mean)) * arange[1],
            mean - 1.96 * std,
            mean + 1.96 * std,
            alpha=0.3,
            label="95% CI",
        )
        for weight_norm in weight_norms[:5]:
            plt.plot(np.arange(len(mean)) * arange[1], weight_norm, alpha=0.3)
        plt.xlabel("Iteration")
        plt.ylabel(r"$\Vert \theta\Vert_2$")
        # plt.axhline(config.thermalize[False]['real_score'], linestyle='--', \
        # label='avg real score', color='r')
        plt.title("Weight convergence")
        plt.legend()
        fig.tight_layout()
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_weight.png"))
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_weight.pdf"))
        plt.close()

    if Path(log_path, "out.txt").exists():
        outs = np.loadtxt(Path(log_path, "out.txt")).reshape(-1, len(arange))
        mean = outs.mean(0)
        std = outs.std(0)
        fig = plt.figure()
        plt.plot(np.arange(len(mean)) * arange[1], mean)
        plt.fill_between(
            np.arange(len(mean)) * arange[1],
            mean - 1.96 * std,
            mean + 1.96 * std,
            alpha=0.3,
            label="95% CI",
        )
        for out in outs[:5]:
            plt.plot(np.arange(len(mean)) * arange[1], out, alpha=0.3)

        plt.xlabel("Iteration")
        plt.ylabel(r"$F(x)$")
        # plt.axhline(config.thermalize[False]['real_score'], linestyle='--', \
        # label='avg real score', color='r')
        plt.title("F(x)")
        plt.legend()
        fig.tight_layout()
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_out.png"))
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_out.pdf"))
        plt.close()

    if Path(log_path, "residual.txt").exists():
        residuals = np.loadtxt(Path(log_path, "residual.txt")).reshape(-1, len(arange))
        mean = residuals.mean(0)
        std = residuals.std(0)
        fig = plt.figure()
        plt.plot(np.arange(len(mean)) * arange[1], mean)
        plt.fill_between(
            np.arange(len(mean)) * arange[1],
            mean - 1.96 * std,
            mean + 1.96 * std,
            alpha=0.3,
            label="95% CI",
        )
        for res in residuals[:5]:
            plt.plot(np.arange(len(mean)) * arange[1], res, alpha=0.3)

        plt.xlabel("Iteration")
        plt.ylabel(
            r"$\Vert\hat{\mathbb{E}}_{\pi_{\theta}} F(G(z)))-\pi_{data}(F)\Vert$"
        )
        # plt.axhline(config.thermalize[False]['real_score'], linestyle='--', \
        # label='avg real score', color='r')
        plt.title("Residual")
        plt.legend()
        fig.tight_layout()
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_residual.png"))
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_residual.pdf"))
        plt.close()

    if Path(log_path, "dot_pr.txt").exists():
        residuals = np.loadtxt(Path(log_path, "dot_pr.txt")).reshape(-1, len(arange))
        mean = residuals.mean(0)
        std = residuals.std(0)
        fig = plt.figure()
        plt.plot(np.arange(len(mean)) * arange[1], mean)
        plt.fill_between(
            np.arange(len(mean)) * arange[1],
            mean - 1.96 * std,
            mean + 1.96 * std,
            alpha=0.3,
            label="95% CI",
        )
        for res in residuals[:5]:
            plt.plot(np.arange(len(mean)) * arange[1], res, alpha=0.3)

        plt.xlabel("Iteration")
        plt.ylabel(r"$\langle \theta, F(x) \rangle$")
        # plt.axhline(config.thermalize[False]['real_score'], linestyle='--', \
        # label='avg real score', color='r')
        plt.title("Weigth-feauture dot product")
        plt.legend()
        fig.tight_layout()
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_dot_pr.png"))
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_dot_pr.pdf"))
        plt.close()

    if Path(log_path, "dot_pr.txt").exists() and Path(log_path, "Energy.txt").exists():
        residuals = np.loadtxt(Path(log_path, "dot_pr.txt")).reshape(-1, len(arange))
        mean = residuals.mean(0)
        std = residuals.std(0)

        energies = np.loadtxt(Path(log_path, "Energy.txt")).reshape(-1, len(arange))
        mean += energies.mean(0)
        std += energies.std(0)

        fig = plt.figure()
        plt.plot(np.arange(len(mean)) * arange[1], mean)
        plt.fill_between(
            np.arange(len(mean)) * arange[1],
            mean - 1.96 * std,
            mean + 1.96 * std,
            alpha=0.3,
            label="95% CI",
        )

        plt.xlabel("Iteration")
        plt.ylabel(r"$\langle \theta, F(G(z)) \rangle + U(z)$")
        # plt.axhline(config.thermalize[False]['real_score'], linestyle='--', \
        # label='avg real score', color='r')
        plt.title("Energy of MaxEnt model")
        plt.legend()
        fig.tight_layout()
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_sum.png"))
        plt.savefig(Path(log_path, "figs", f"{log_path.name}_sum.pdf"))
        plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logdir", dest="logdir", type=str)
    parser.add_argument("-f", "--feature", dest="feature", type=str)
    parser.add_argument("-t", "--target", dest="target", type=str)
    parser.add_argument("-m", "--model_pattern", dest="model_pattern", type=str)
    args = parser.parse_args()
    return args
