from typing import Iterable, List

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


# sns.set_theme('talk', style="white")


def ema(series: Iterable, n: int) -> List:
    """
    returns an n period exponential moving average for
    the time series
    """
    series = np.array(series)
    ema = []
    j = 1

    # get n sma first and calculate the next n period ema
    sma = sum(series[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    # EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(((series[n] - sma) * multiplier) + sma)

    # now calculate the rest of the values
    for i in series[n + 1 :]:
        tmp = ((i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema


def plot_result(
    chains, dist, flow=None, chain_id=0, grid_n=100, proj_dim1=-1, proj_dim2=-2
):
    proj_slice = [proj_dim1, proj_dim2]
    proj_dim1 = dist.dim + proj_dim1 + 1 if proj_dim1 < 0 else proj_dim1 + 1
    proj_dim2 = dist.dim + proj_dim2 + 1 if proj_dim2 < 0 else proj_dim2 + 1

    if flow:
        fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    result = chains.reshape(-1, chains.shape[-1])
    dist.plot_2d_countour(axs[0])
    xmin, xmax = axs[0].get_xlim()
    ymin, ymax = axs[0].get_ylim()

    axs[0].scatter(
        *result[:, proj_slice].T, alpha=min(0.6, 1000.0 / result.shape[0]), s=30, c='coral', edgecolors='black', linewidth=0.5,
    )  # , c='r', marker='o')
    axs[0].set_title(f"Projected samples from {chains.shape[1]} chains")

    kernel = gaussian_kde(result[:, proj_slice].T)
    x = np.linspace(xmin, xmax, grid_n)
    y = np.linspace(ymin, ymax, grid_n)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde = np.reshape(kernel(positions).T, X.shape)
    axs[1].contour(X, Y, kde, colors='midnightblue', linewidths=1)
    axs[1].set_title(f"KDE")

    chain_id = 0
    result = chains[:, chain_id]
    dist.plot_2d_countour(axs[2])
    axs[2].plot(
        *result[:, proj_slice].T, "-", alpha=min(0.6, 1000.0 / result.shape[0]), c='coral', linewidth=1, marker='o', markersize=1, mec='black'
    )  # , c='k')
    axs[2].set_title(f"Trajectory of chain {chain_id}")

    if flow:
        flow_sample = flow.sample((10000,)).detach().cpu()
        kernel = gaussian_kde(flow_sample[:, proj_slice].T)
        kde = np.reshape(kernel(positions).T, X.shape)
        axs[3].contour(X, Y, kde, colors='midnightblue', linewidths=1)
        axs[3].set_title(f"KDE of NF samples")

    for ax in axs:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(rf"$X{proj_dim1}$")
        ax.set_ylabel(rf"$X{proj_dim2}$")

    fig.tight_layout()
