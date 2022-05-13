import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .ebm_sampling import (
    langevin_sampling,
    mala_sampling,
    mh_sampling_normal_proposal,
)
from .general_utils import send_file_to_remote
from .mh_sampling import mh_sampling, mh_sampling_from_scratch
from .sir_ais_sampling import sir_correlated_sampling, sir_independent_sampling


figsize = (8, 8)


def sample_fake_data(
    generator,
    X_train,
    x_range,
    y_range,
    path_to_save=None,
    epoch=None,
    scaler=None,
    batch_size_sample=5000,
    path_to_final_save=None,
    path_to_save_remote=None,
    port_to_remote=None,
):
    fake_data = generator.sampling(batch_size_sample).data.cpu().numpy()
    if scaler is not None:
        fake_data = scaler.inverse_transform(fake_data)
    plt.figure(figsize=(8, 8))
    plt.xlim(-x_range, x_range)
    plt.ylim(-y_range, y_range)
    plt.title("Training and generated samples", fontsize=20)
    plt.scatter(
        X_train[:, :1],
        X_train[:, 1:],
        alpha=0.3,
        color="gray",
        marker="o",
        label="training samples",
    )
    plt.scatter(
        fake_data[:, :1],
        fake_data[:, 1:],
        alpha=0.3,
        color="blue",
        marker="o",
        label="samples by G",
    )
    plt.legend()
    plt.grid(True)
    plt.xlabel(r"$x_{1}$")
    plt.ylabel(r"$x_{2}$")
    if path_to_save is not None:
        cur_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if epoch is not None:
            plot_name = cur_time + f"_gan_sampling_{epoch}_epoch.pdf"
        else:
            plot_name = cur_time + f"_gan_sampling.pdf"

        if path_to_final_save is not None:
            path_to_plot = path_to_final_save
        else:
            path_to_plot = os.path.join(path_to_save, plot_name)
        plt.savefig(path_to_plot)
        send_file_to_remote(path_to_plot, port_to_remote, path_to_save_remote)
        plt.close()
    else:
        plt.show()


def plot_fake_data_mode(
    fake,
    X_train,
    mode,
    path_to_save=None,
    scaler=None,
    path_to_save_remote=None,
    port_to_remote=None,
    params=None,
):
    # fake_data = fake.data.cpu().numpy()
    if scaler is not None:
        fake = scaler.inverse_transform(fake)
    plt.figure(figsize=figsize)
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title(f"Training and {mode} samples", fontsize=20)
    plt.scatter(
        X_train[:, :1],
        X_train[:, 1:],
        alpha=0.3,
        color="gray",
        marker="o",
        label="training samples",
    )
    label = f"{mode} samples"
    if params is not None:
        label += ", " + params
    plt.scatter(
        fake[:, :1],
        fake[:, 1:],
        alpha=0.3,
        color="blue",
        marker="o",
        label=label,
    )
    plt.legend()
    plt.grid(True)
    plt.xlabel(r"$x_{1}$")
    plt.ylabel(r"$x_{2}$")
    if path_to_save is not None:
        plt.savefig(path_to_save)
        send_file_to_remote(path_to_save, port_to_remote, path_to_save_remote)
        plt.close()
    else:
        plt.show()


def plot_real_data(
    X_train,
    title,
    path_to_save=None,
    path_to_save_remote=None,
    port_to_remote=None,
):
    plt.figure(figsize=figsize)
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title(title, fontsize=20)
    plt.scatter(
        X_train[:, :1],
        X_train[:, 1:],
        alpha=0.3,
        color="gray",
        marker="o",
        label="training samples",
    )
    plt.legend()
    plt.grid(True)
    plt.xlabel(r"$x_{1}$")
    plt.ylabel(r"$x_{2}$")
    if path_to_save is not None:
        plt.savefig(path_to_save)
        send_file_to_remote(path_to_save, port_to_remote, path_to_save_remote)
        plt.show()
        plt.close()
    else:
        plt.show()


def plot_fake_data_projection(
    fake,
    X_train,
    proj_1,
    proj_2,
    title,
    fake_label,
    path_to_save=None,
    scaler=None,
    path_to_save_remote=None,
    port_to_remote=None,
):
    if scaler is not None:
        fake = scaler.inverse_transform(fake)
    fake_proj = fake[:, [proj_1, proj_2]]

    plt.figure(figsize=figsize)
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.title(title, fontsize=20)
    X_train_proj = X_train[:, [proj_1, proj_2]]

    plt.scatter(
        X_train_proj[:, 0],
        X_train_proj[:, 1],
        alpha=0.3,
        color="gray",
        marker="o",
        label="training samples",
    )
    plt.scatter(
        fake_proj[:, 0],
        fake_proj[:, 1],
        alpha=0.3,
        color="blue",
        marker="o",
        label=fake_label,
    )
    plt.xlabel(f"proj ind = {proj_1 + 1}")
    plt.ylabel(f"proj ind = {proj_2 + 1}")
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
        plt.savefig(path_to_save)
        send_file_to_remote(path_to_save, port_to_remote, path_to_save_remote)
        plt.close()
    else:
        plt.show()


def plot_discriminator_2d(
    discriminator,
    x_range,
    y_range,
    path_to_save=None,
    epoch=None,
    scaler=None,
    port_to_remote=None,
    path_to_save_remote=None,
    normalize_to_0_1=True,
    num_points=700,
):
    x = torch.linspace(-x_range, x_range, num_points)
    y = torch.linspace(-y_range, y_range, num_points)
    x_t = x.view(-1, 1).repeat(1, y.size()[0])
    y_t = y.view(1, -1).repeat(x.size()[0], 1)
    x_t_batch = x_t.view(-1, 1)
    y_t_batch = y_t.view(-1, 1)
    batch = torch.zeros((x_t_batch.shape[0], 2))
    batch[:, 0] = x_t_batch[:, 0]
    batch[:, 1] = y_t_batch[:, 0]
    if scaler is not None:
        batch = batch.numpy()
        batch = scaler.transform(batch)
        batch = torch.FloatTensor(batch)
    discr_batch = discriminator(batch.to(discriminator.device))
    heatmap = discr_batch[:, 0].view((num_points, num_points)).detach().cpu()
    if normalize_to_0_1:
        heatmap = heatmap.sigmoid().numpy()
    else:
        heatmap = heatmap.numpy()

    x_numpy = x.numpy()
    y_numpy = y.numpy()
    y, x = np.meshgrid(x_numpy, y_numpy)
    l_x = x_numpy.min()
    r_x = x_numpy.max()
    l_y = y_numpy.min()
    r_y = y_numpy.max()
    # small_heatmap = sigmoid_heatmap[:-1, :-1]
    figure, axes = plt.subplots(figsize=figsize)
    z = axes.contourf(x, y, heatmap, 10, cmap="viridis")
    if epoch is not None:
        title = f"Discriminator heatmap, epoch = {epoch}"
    else:
        title = f"Discriminator heatmap"
    axes.set_title(title)
    axes.axis([l_x, r_x, l_y, r_y])
    figure.colorbar(z)
    axes.set_xlabel(r"$x_{1}$")
    axes.set_ylabel(r"$x_{2}$")
    if path_to_save is not None:
        figure.savefig(path_to_save)
        send_file_to_remote(path_to_save, port_to_remote, path_to_save_remote)
        plt.close()


def plot_potential_energy(
    target_energy,
    x_range,
    y_range,
    device,
    path_to_save=None,
    norm_grads=False,
    num_points=100,
    mode_visual="image",
    title=None,
):
    x = torch.linspace(-x_range, x_range, num_points)
    y = torch.linspace(-y_range, y_range, num_points)
    x_t = x.view(-1, 1).repeat(1, y.size()[0])
    y_t = y.view(1, -1).repeat(x.size()[0], 1)
    x_t_batch = x_t.view(-1, 1)
    y_t_batch = y_t.view(-1, 1)
    batch = torch.zeros((x_t_batch.shape[0], 2))
    batch[:, 0] = x_t_batch[:, 0]
    batch[:, 1] = y_t_batch[:, 0]

    batch = batch.to(device)

    if norm_grads:
        batch.requires_grad_(True)
        batch_energy = target_energy(batch).sum()
        batch_energy.backward()
        batch_grads = batch.grad.detach().cpu()
        batch_grads_norm = torch.norm(batch_grads, p=2, dim=-1)
        result = (
            batch_grads_norm.view((num_points, num_points))
            .detach()
            .cpu()
            .numpy()
        )
        if title is None:
            title = "Latent energy norm gradients"
    else:
        batch_energy = target_energy(batch)
        result = (
            batch_energy.view((num_points, num_points)).detach().cpu().numpy()
        )
        if title is None:
            title = "Latent energy"

    x_numpy = x.numpy()
    y_numpy = y.numpy()
    y, x = np.meshgrid(x_numpy, y_numpy)
    l_x = x_numpy.min()
    r_x = x_numpy.max()
    l_y = y_numpy.min()
    r_y = y_numpy.max()
    # small_heatmap = sigmoid_heatmap[:-1, :-1]
    figure, axes = plt.subplots(figsize=figsize)
    z = axes.contourf(x, y, result, 10, cmap="viridis")
    axes.set_title(title)
    axes.axis([l_x, r_x, l_y, r_y])
    if mode_visual == "image":
        axes.set_xlabel(r"$x_{1}$")
        axes.set_ylabel(r"$x_{2}$")
    elif mode_visual == "latent":
        axes.set_xlabel(r"$z_{1}$")
        axes.set_ylabel(r"$z_{2}$")
    figure.colorbar(z)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    else:
        plt.show()


def langevin_sampling_plot_2d(
    target,
    proposal,
    X_train,
    path_to_save=None,
    scaler=None,
    batch_size_sample=5000,
    path_to_save_remote=None,
    port_to_remote=None,
    grad_step=1e-3,
    eps_scale=1e-2,
    n_steps=5000,
    n_batches=1,
    latent_transform=None,
):
    batchsize = batch_size_sample // n_batches
    X_langevin, zs = langevin_sampling(
        target,
        proposal,
        batchsize,
        batch_size_sample,
        None,
        None,
        None,
        None,
        n_steps,
        grad_step,
        eps_scale,
    )
    if latent_transform is not None:
        X_langevin = torch.FloatTensor(X_langevin).to(proposal.device)
        X_langevin = latent_transform(X_langevin).data.cpu().numpy()
    mode = "ULA"
    params = f"n_steps = {n_steps}, lr = {grad_step}, std noise = {round(eps_scale, 3)}"
    plot_fake_data_mode(
        X_langevin,
        X_train,
        mode,
        path_to_save=path_to_save,
        scaler=scaler,
        path_to_save_remote=path_to_save_remote,
        port_to_remote=port_to_remote,
        params=params,
    )


def mh_sampling_normal_proposal_plot_2d(
    target,
    proposal,
    X_train,
    path_to_save=None,
    scaler=None,
    batch_size_sample=5000,
    path_to_save_remote=None,
    port_to_remote=None,
    eps_scale=1e-2,
    n_steps=5000,
    acceptance_rule="Hastings",
    n_batches=1,
    latent_transform=None,
):
    batchsize = batch_size_sample // n_batches
    X_mh, zs = mh_sampling_normal_proposal(
        target,
        proposal,
        batchsize,
        batch_size_sample,
        None,
        None,
        None,
        None,
        n_steps,
        eps_scale,
        acceptance_rule,
    )
    if latent_transform is not None:
        X_mh = torch.FloatTensor(X_mh).to(proposal.device)
        X_mh = latent_transform(X_mh).data.cpu().numpy()
    mode = f"MH, acceptance rule = {acceptance_rule}"
    params = f"n_steps = {n_steps}, std noise = {round(eps_scale, 3)}"
    plot_fake_data_mode(
        X_mh,
        X_train,
        mode,
        path_to_save=path_to_save,
        scaler=scaler,
        path_to_save_remote=path_to_save_remote,
        port_to_remote=port_to_remote,
        params=params,
    )


def sir_correlated_plot_2d(
    target,
    proposal,
    X_train,
    path_to_save=None,
    scaler=None,
    batch_size_sample=5000,
    path_to_save_remote=None,
    port_to_remote=None,
    N=2,
    n_steps=5000,
    alpha=0.95,
    n_batches=1,
    latent_transform=None,
):
    batchsize = batch_size_sample // n_batches
    X_mh, zs = sir_correlated_sampling(
        target,
        proposal,
        batchsize,
        batch_size_sample,
        None,
        None,
        None,
        None,
        n_steps,
        N,
        alpha,
    )
    if latent_transform is not None:
        X_mh = torch.FloatTensor(X_mh).to(proposal.device)
        X_mh = latent_transform(X_mh).data.cpu().numpy()
    mode = "CISIR"
    params = f"n_steps={n_steps}, N = {N}, alpha = {round(alpha, 3)}"
    plot_fake_data_mode(
        X_mh,
        X_train,
        mode,
        path_to_save=path_to_save,
        scaler=scaler,
        path_to_save_remote=path_to_save_remote,
        port_to_remote=port_to_remote,
        params=params,
    )


def sir_independent_plot_2d(
    target,
    proposal,
    X_train,
    path_to_save=None,
    scaler=None,
    batch_size_sample=5000,
    path_to_save_remote=None,
    port_to_remote=None,
    N=2,
    n_steps=5000,
    n_batches=1,
    latent_transform=None,
):
    batchsize = batch_size_sample // n_batches
    X_mh, zs = sir_independent_sampling(
        target,
        proposal,
        batchsize,
        batch_size_sample,
        None,
        None,
        None,
        None,
        n_steps,
        N,
    )
    if latent_transform is not None:
        X_mh = torch.FloatTensor(X_mh).to(proposal.device)
        X_mh = latent_transform(X_mh).data.cpu().numpy()
    mode = "SIR"
    params = f"n_steps={n_steps}, N = {N}"
    plot_fake_data_mode(
        X_mh,
        X_train,
        mode,
        path_to_save=path_to_save,
        scaler=scaler,
        path_to_save_remote=path_to_save_remote,
        port_to_remote=port_to_remote,
        params=params,
    )


def mala_sampling_plot_2d(
    target,
    proposal,
    X_train,
    path_to_save=None,
    scaler=None,
    batch_size_sample=5000,
    path_to_save_remote=None,
    port_to_remote=None,
    grad_step=1e-3,
    eps_scale=1e-2,
    n_steps=5000,
    n_batches=1,
    acceptance_rule="Hastings",
    latent_transform=None,
):
    batchsize = batch_size_sample // n_batches
    X_mala, zs = mala_sampling(
        target,
        proposal,
        batchsize,
        batch_size_sample,
        None,
        None,
        None,
        None,
        n_steps,
        grad_step,
        eps_scale,
        acceptance_rule,
    )
    if latent_transform is not None:
        X_mala = torch.FloatTensor(X_mala).to(proposal.device)
        X_mala = latent_transform(X_mala).data.cpu().numpy()
    mode = f"MALA/{acceptance_rule}"
    params = f"n_steps={n_steps}, lr = {grad_step}, std noise = {round(eps_scale, 3)}"
    plot_fake_data_mode(
        X_mala,
        X_train,
        mode,
        path_to_save=path_to_save,
        scaler=scaler,
        path_to_save_remote=path_to_save_remote,
        port_to_remote=port_to_remote,
        params=params,
    )


def mh_sampling_plot_2d(
    generator,
    discriminator,
    X_train,
    n_steps,
    path_to_save=None,
    n_calib_pts=10000,
    scaler=None,
    batch_size_sample=5000,
    path_to_save_remote=None,
    port_to_remote=None,
    type_calibrator="isotonic",
    normalize_to_0_1=True,
):
    if scaler is not None:
        X_train_scale = scaler.transform(X_train)
    else:
        X_train_scale = X_train
    print("Start to do MH sampling....")
    X_mh = mh_sampling_from_scratch(
        X_train_scale,
        generator,
        discriminator,
        generator.device,
        n_calib_pts,
        batch_size_sample=batch_size_sample,
        n_steps=n_steps,
        type_calibrator=type_calibrator,
        normalize_to_0_1=normalize_to_0_1,
    )
    mode = "MHGAN"
    params = f"n_steps = {n_steps}, calibrator = {type_calibrator}"
    plot_fake_data_mode(
        X_mh,
        X_train,
        mode,
        path_to_save=path_to_save,
        scaler=scaler,
        path_to_save_remote=path_to_save_remote,
        port_to_remote=port_to_remote,
        params=params,
    )


def epoch_visualization(
    X_train,
    generator,
    discriminator,
    use_gradient_penalty,
    discriminator_mean_loss_arr,
    epoch,
    Lambda,
    generator_mean_loss_arr,
    path_to_save,
    batch_size_sample=5000,
    loss_type="Jensen",
    mode="25_gaussians",
    port_to_remote=None,
    path_to_save_remote=None,
    scaler=None,
    proj_list=None,
    n_calib_pts=10000,
    normalize_to_0_1=True,
    plot_mhgan=False,
):
    X_mh = None
    if path_to_save is not None:
        cur_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        plot_name = cur_time + f"_gan_losses_{epoch}_epoch.pdf"
        path_to_plot = os.path.join(path_to_save, plot_name)
        subtitle_for_losses = (
            "Training process for discriminator and generator"
        )
        if use_gradient_penalty:
            subtitle_for_losses += (
                fr" with gradient penalty, $\lambda = {Lambda}$"
            )
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        fig.suptitle(subtitle_for_losses)
        axs[0].set_xlabel("#epoch")
        axs[0].set_ylabel("loss")
        axs[1].set_xlabel("#epoch")
        axs[1].set_ylabel("loss")
        axs[0].grid(True)
        axs[1].grid(True)
        axs[0].set_title("D-loss")
        axs[1].set_title("G-loss")
        axs[0].plot(
            discriminator_mean_loss_arr,
            "b",
            label=f"discriminator loss = {loss_type}",
        )
        axs[1].plot(generator_mean_loss_arr, "r", label="generator loss")
        axs[0].legend()
        axs[1].legend()
        fig.savefig(path_to_plot)
        if mode in ["25_gaussians", "swissroll"]:
            if mode == "25_gaussians":
                x_range = 3.0
                y_range = 3.0
            else:
                x_range = 2.0
                y_range = 2.0
            plot_name = cur_time + f"_discriminator_{epoch}_epoch.pdf"
            path_to_plot_discriminator = os.path.join(path_to_save, plot_name)
            plot_discriminator_2d(
                discriminator,
                x_range,
                y_range,
                path_to_plot_discriminator,
                epoch,
                scaler=scaler,
                port_to_remote=port_to_remote,
                path_to_save_remote=path_to_save_remote,
            )
            if plot_mhgan:
                print("nothing")
                # mh_mode = 'mhgan'
                # plot_name = cur_time + f'_{mh_mode}_sampling.pdf'
                # path_to_plot_mhgan = os.path.join(path_to_save, plot_name)

                # type_calibrator = 'iso'
                # mh_sampling_visualize(generator,
                #                      discriminator,
                #                      X_train,
                #                      path_to_plot_mhgan,
                #                      n_calib_pts = n_calib_pts,
                #                      scaler = scaler,
                #                      batch_size_sample = batch_size_sample,
                #                      port_to_remote=port_to_remote,
                #                      path_to_save_remote = path_to_save_remote,
                #                      normalize_to_0_1 = normalize_to_0_1,
                #                      type_calibrator = type_calibrator)

            sample_fake_data(
                generator,
                X_train,
                x_range,
                y_range,
                path_to_save,
                epoch=epoch,
                scaler=scaler,
                batch_size_sample=batch_size_sample,
                port_to_remote=port_to_remote,
                path_to_save_remote=path_to_save_remote,
            )

        elif mode == "5d_gaussians":
            fake_generator = (
                generator.sampling(batch_size_sample).data.cpu().numpy()
            )
            cur_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

            if plot_mhgan:
                if scaler is not None:
                    X_train_scale = scaler.transform(X_train)
                else:
                    X_train_scale = X_train

                print("Start to do MH sampling....")
                type_calibrator = "iso"
                X_mh = mh_sampling(
                    X_train_scale,
                    generator,
                    discriminator,
                    generator.device,
                    n_calib_pts,
                    batch_size_sample=batch_size_sample,
                    normalize_to_0_1=normalize_to_0_1,
                    type_calibrator=type_calibrator,
                )

            for i in range(len(proj_list)):
                proj_1 = proj_list[i][0]
                proj_2 = proj_list[i][1]
                plot_name = (
                    cur_time
                    + f"_gan_sampling_epoch_{epoch}_proj1_{proj_1}_proj2_{proj_2}.pdf"
                )
                path_to_plot_generator = os.path.join(path_to_save, plot_name)

                title_generator = "Training and generated samples"
                fake_label_generator = "samples by G"

                plot_fake_data_projection(
                    fake=fake_generator,
                    X_train=X_train,
                    path_to_save=path_to_plot_generator,
                    proj_1=proj_1,
                    proj_2=proj_2,
                    title=title_generator,
                    fake_label=fake_label_generator,
                    scaler=scaler,
                    path_to_save_remote=path_to_save_remote,
                    port_to_remote=port_to_remote,
                )
                if plot_mhgan:
                    title_mhgan = "Training and MHGAN samples"
                    fake_label_mhgan = "MHGAN samples"
                    mh_mode = "mhgan"
                    plot_name = (
                        cur_time
                        + f"_{mh_mode}_epoch_{epoch}_proj1_{proj_1}_proj2_{proj_2}.pdf"
                    )
                    path_to_plot_mhgan = os.path.join(path_to_save, plot_name)
                    plot_fake_data_projection(
                        fake=X_mh,
                        X_train=X_train,
                        path_to_save=path_to_plot_mhgan,
                        proj_1=proj_1,
                        proj_2=proj_2,
                        title=title_mhgan,
                        fake_label=fake_label_mhgan,
                        scaler=scaler,
                        path_to_save_remote=path_to_save_remote,
                        port_to_remote=port_to_remote,
                    )


def plot_chain_metrics(
    evols,
    every=50,
    colors=None,
    name=None,
    savepath=None,
    sigma=0.05,
    std=True,
    keys=["mode_std", "hqr", "jsd", "n_found_modes", "emd"],
):
    instance = list(evols.values())[0]
    # keys = ["mode_std", "hqr", "jsd", "n_found_modes", "emd"]
    ncols = int(np.sum([int(len(instance[k]) > 0) for k in keys]))

    fig, axs = plt.subplots(
        ncols=ncols, nrows=1, figsize=(6 * ncols, 6), squeeze=False
    )
    axs = axs[0]

    # fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(6*3, 12))#ncols, nrows=1, figsize=(6*ncols, 6))
    # axs = axs.flatten()

    if name is not None:
        fig.suptitle(name)
    k = 0
    if sigma is not None and len(instance["mode_std"][0]) > 0:
        axs[k].axhline(sigma, label="real", color="black")
        axs[k].set_xlabel("Iterations")
        axs[k].set_ylabel("mode std")
        axs[k].set_title("Estimation of mode std of samples")
        k += 1

    if len(instance["hqr"][0]) > 0:
        axs[k].axhline(1, label="real", color="black")
        axs[k].set_xlabel("Iterations")
        axs[k].set_ylabel("HQR")
        axs[k].set_title("High quality rate of samples")
        k += 1

    if len(instance["jsd"][0]) > 0:
        axs[k].axhline(0, label="real", color="black")
        axs[k].set_xlabel("Iterations")
        axs[k].set_ylabel("JSD")
        axs[k].set_title(r"JSD btw. $U\{1, M+1\}$ and empirical dist.")
        k += 1

    if len(instance["n_found_modes"][0]) > 0:
        # axs[k].axhline(25, label='real', color='black')
        axs[k].set_xlabel("Iterations")
        axs[k].set_ylabel("# of captured modes")
        axs[k].set_title("Number of captured modes")
        k += 1

    if len(instance["emd"][0]) > 0:
        axs[k].axhline(0, label="real", color="black")
        axs[k].set_xlabel("Iterations")
        axs[k].set_ylabel("EMD")
        axs[k].set_title("Earth Mover's Distance")
        k += 1

    # emd_ax = axs if k == 0 else axs[k]
    # emd_ax.axhline(0, label='real', color='black')
    # emd_ax.set_xlabel('iter')
    # emd_ax.set_ylabel('EMD')
    # emd_ax.set_title('Earth Mover\'s Distance')

    for i, label in enumerate(evols.keys()):
        evol = evols[label]
        # color = evol['color']
        if colors is not None:
            color = colors[i]
            # color = evol['color'] #colors[i]
            # print()
        else:
            color = sns.color_palette()[i]
        k = 0

        def plot_evol(ax, ev, label):
            if std is True:
                mean_arr, std_arr = ev
                ax.plot(
                    np.arange(1, len(mean_arr) + 1) * every,
                    mean_arr,
                    label=label,
                    marker="o",
                    color=color,
                )
                ax.fill_between(
                    np.arange(1, len(mean_arr) + 1) * every,
                    mean_arr - 1.96 * std_arr,
                    mean_arr + 1.96 * std_arr,
                    alpha=0.2,
                    color=color,
                )
            else:
                mean_arr = ev
                ax.plot(
                    np.arange(1, len(mean_arr) + 1) * every,
                    mean_arr,
                    label=label,
                    marker="o",
                    color=color,
                )

        for i, key in enumerate(keys):
            if len(instance[key]) > 0:
                k += 1
                plot_evol(axs[i], evol[key], label)

        # if len(instance['mode_std']) > 0:
        #     k += 1
        #     axs[0].plot(np.arange(len(evol['mode_std'])) * every, evol['mode_std'], label=label, marker='o')
        # if len(instance['hqr']) > 0:
        #     k += 1
        #     axs[1].plot(np.arange(len(evol['hqr'])) * every, evol['hqr'], label=label, marker='o')
        # if len(instance['jsd']) > 0:
        #     k += 1
        #     axs[2].plot(np.arange(len(evol['jsd'])) * every, evol['jsd'], label=label, marker='o')
        # emd_ax.plot(np.arange(1, len(evol['emd'])+1) * every, evol['emd'], label=label, marker='o')

    if k > 0:
        for ax in axs:
            ax.grid()
            ax.legend()
    # else:
    #     emd_ax.grid()
    #     emd_ax.legend()
    # fig.delaxes(axs[-1])

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
    # plt.show()
