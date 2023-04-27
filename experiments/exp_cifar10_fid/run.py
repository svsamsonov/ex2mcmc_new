import argparse
import datetime
import itertools
import logging
import subprocess
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torchvision
import wandb
from tqdm import trange

from ex2mcmc.gan_distribution import Distribution, DistributionRegistry
from ex2mcmc.metrics.compute_fid_tf import calculate_fid_given_paths
from ex2mcmc.metrics.inception_score import (
    MEAN_TRASFORM,
    N_GEN_IMAGES,
    STD_TRANSFORM,
    get_inception_score,
)
from ex2mcmc.models.rnvp import RNVP  # noqa: F401
from ex2mcmc.models.rnvp_minimal import MinimalRNVP
from ex2mcmc.models.utils import GANWrapper
from ex2mcmc.sample import Sampler
from ex2mcmc.utils.callbacks import CallbackRegistry
from ex2mcmc.utils.general_utils import DotConfig, IgnoreLabelDataset, random_seed
from experiments.tools.plot_results import plot_res


FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs="+")
    parser.add_argument("--group", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--step_size", type=float)

    parser.add_argument("--suffix", type=str)

    args = parser.parse_args()
    return args


def define_sampler(
    config: DotConfig,
    gan: GANWrapper,
    ref_dist: Distribution,
    save_dir: Path,
):
    sampler_callbacks = []
    callbacks = config.callbacks.sampler_callbacks
    if callbacks:
        for _, callback in callbacks.items():
            params = callback.params.dict
            # HACK
            if "save_dir" in params:
                params["save_dir"] = save_dir
            sampler_callbacks.append(CallbackRegistry.create(callback.name, **params))
    sampler = Sampler(
        gan.gen,
        ref_dist,
        **config.sample_params.params,
        callbacks=sampler_callbacks,
    )

    return sampler


def main(config: DotConfig, device: torch.device, group: str):
    suffix = f"_{config.suffix}" if config.suffix else ""
    dir_suffix = ""  # f"_{config.distribution.name}"

    # sample
    if config.sample_params.sample:
        if config.sample_params.sub_dir:
            save_dir = Path(
                config.sample_params.save_dir + dir_suffix,
                config.sample_params.sub_dir + suffix,
            )
        else:
            save_dir = Path(
                config.sample_params.save_dir + dir_suffix,
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + suffix,
            )
        save_dir = save_dir.with_name(
            f"{save_dir.name}_{config.sample_params.params.sampling}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        yaml.round_trip_dump(config.dict, Path(save_dir, config.file_name).open("w"))
        yaml.round_trip_dump(
            dict(gan_config=config.gan_config.dict),
            Path(save_dir, "gan_config.yml").open("w"),
        )
        print(save_dir)
        gan = GANWrapper(config.gan_config, device)  # , eval=False)
        ref_dist = DistributionRegistry.create(
            config.sample_params.distribution.name,
            gan=gan,
            batch_size=config.batch_size,
        )

        if config.seed is not None:
            random_seed(config.seed)
        total_sample_z = []
        total_sample_x = []
        total_labels = []

        labels = None
        if config.resume:
            latents_dir = Path(save_dir, "latents")
            lat_paths = sorted(
                latents_dir.glob("*.npy"), key=lambda x: int(x.stem.split("_")[-1])
            )
            latent_path = lat_paths[-1]
            start_step = int(latent_path.stem.split("_")[-1])
            start_step_id = start_step // config.sample_params.save_every
            config.sample_params.params.dict["n_steps"] = (
                config.sample_params.params.n_steps - start_step
            )
            start_latents = torch.from_numpy(np.load(latent_path))
            if Path(save_dir, "labels.npy").exists():
                labels = torch.from_numpy(np.load(Path(save_dir, "labels.npy")))
        else:
            start_latents = gan.prior.sample((config.sample_params.total_n,)).cpu()
            start_step_id = 0

        if labels is None:
            labels = torch.LongTensor(
                np.random.randint(
                    0,
                    9,  # dataset_info.get("n_classes", 10) - 1,
                    config.sample_params.total_n,
                )
            )

        sampler = define_sampler(config, gan, ref_dist, save_dir)

        for i, start, label in zip(
            range(0, config.sample_params.total_n, config.sample_params.batch_size),
            torch.split(start_latents, config.sample_params.batch_size),
            torch.split(labels, config.sample_params.batch_size),
        ):
            print(i)

            # if wandb.run is not None:
            #     run = wandb.run
            #     run.config.update({"group": f"{group}"})
            #     run.config.update({"name": f"{group}_{i}"}, allow_val_change=True)

            if config.get("flow", None):
                gan.gen.proposal = MinimalRNVP(
                    gan.gen.z_dim, device=device, hidden=32, num_blocks=4
                )
                if config.gan_config.dp:
                    gan.gen.proposal = torch.nn.DataParallel(gan.gen.proposal)
                    gan.gen.proposal.prior = gan.gen.prior
                    gan.gen.proposal.log_prob = gan.gen.proposal.module.log_prob
                    gan.gen.proposal.sample = gan.gen.proposal.module.sample
                    gan.gen.proposal.inverse = gan.gen.proposal.module.inverse

                opt = torch.optim.Adam(gan.gen.proposal.parameters(), 1e-3)
                for _ in trange(1000):
                    e = -gan.gen.proposal.log_prob(
                        gan.gen.prior.sample((config.batch_size,))
                    ).mean()
                    gan.gen.proposal.zero_grad()
                    e.backward()
                    opt.step()
                opt = torch.optim.Adam(
                    gan.gen.proposal.parameters(), **config.flow.opt_params
                )
                gan.gen.proposal.optim = opt
                gan.gen.proposal.train()
                gan.gen.proposal.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    opt, lambda it: int(it < config.flow.train_iters)
                )
            else:
                gan.gen.proposal = gan.gen.prior

            start = start.to(device)
            label = label.to(device)
            gan.set_label(label)

            zs, xs = sampler(start)
            sampler.reset()
            gan.gen.input = gan.gen.output = gan.dis.input = gan.dis.output = None

            zs = torch.stack(zs, 0).cpu()
            xs = torch.stack(xs, 0).cpu()
            print(zs.shape)
            total_sample_z.append(zs)
            total_sample_x.append(xs)
            total_labels.append(label.cpu())

        total_sample_z = torch.cat(total_sample_z, 1)[
            :, : config.sample_params.total_n
        ]  # (number_of_steps / every) x total_n x latent_dim
        total_sample_x = torch.cat(total_sample_x, 1)[
            :, : config.sample_params.total_n
        ]  # (number_of_steps / every) x total_n x img_size x img_size
        total_labels = torch.cat(total_labels, 0)[: config.sample_params.total_n]

        imgs_dir = Path(save_dir, "images")
        imgs_dir.mkdir(exist_ok=True)
        latents_dir = Path(save_dir, "latents")
        latents_dir.mkdir(exist_ok=True)
        for slice_id in range(total_sample_x.shape[0]):
            np.save(
                Path(
                    imgs_dir,
                    f"{(slice_id + start_step_id) * config.sample_params.save_every}.npy",
                ),
                total_sample_x[slice_id].cpu().numpy(),
            )
            np.save(
                Path(
                    latents_dir,
                    f"{(slice_id + start_step_id) * config.sample_params.save_every}.npy",
                ),
                total_sample_z[slice_id].cpu().numpy(),
            )

        np.save(
            Path(
                save_dir,
                "labels.npy",
            ),
            total_labels.cpu().numpy(),
        )

    # afterall
    results_dir = config.afterall_params.results_dir + dir_suffix
    if config.afterall_params.sub_dir == "latest":
        results_dir = filter(Path(results_dir).glob("*"))[-1]
    else:
        results_dir = Path(
            results_dir, config.afterall_params.sub_dir + suffix + f"_{config.sampling}"
        )

    print(results_dir)

    assert Path(results_dir).exists()

    if config.get("resume", False):
        start_step_id = len(np.loadtxt(Path(results_dir, "fid_values.txt")))  # - 1
    else:
        start_step_id = 0

    # if config.afterall_params.init_wandb:
    #     wandb.init(**config.wandb_init_params, group=group)
    #     wandb.run.config.update({"group": f"{group}"})

    if config.afterall_params.compute_is:
        transform = torchvision.transforms.Normalize(MEAN_TRASFORM, STD_TRANSFORM)
        model = torchvision.models.inception_v3(
            pretrained=True, transform_input=False
        ).to(device)
        model.eval()

        if config.resume:
            is_values = np.loadtxt(Path(results_dir, "is_values.txt")).tolist()
        else:
            is_values = []
        for step in range(
            start_step_id * config.every,
            config.n_steps - int(config.burn_in_steps) + 1,
            config.every,
        ):
            file = Path(
                results_dir,
                "images",
                f"{step}.npy",
            )
            print(file)

            images = np.load(file)
            dataset = transform(torch.from_numpy(images))
            print(dataset.shape)
            dataset = IgnoreLabelDataset(torch.utils.data.TensorDataset(dataset))

            inception_score_mean, inception_score_std, _ = get_inception_score(
                dataset,
                model,
                resize=True,
                device=device,
                batch_size=50,
                splits=max(1, len(images) // N_GEN_IMAGES),
            )

            print(f"Iter: {step}\t IS: {inception_score_mean}")
            # if wandb.run is not None:
            #     wandb.run.log({"step": step, "overall IS": inception_score_mean})

            is_values.append((inception_score_mean, inception_score_std))
            np.savetxt(
                Path(
                    results_dir,
                    "is_values.txt",
                ),
                is_values,
            )

    if config.callbacks.afterall_callbacks:
        gan = GANWrapper(config.gan_config, device, eval=True)

        label_file = Path(
            results_dir,
            "labels.npy",
        )
        try:
            label = np.load(label_file)
        except Exception:
            label = np.random.randint(0, 10 - 1, 10000)

        afterall_callbacks = []
        callbacks = config.callbacks.afterall_callbacks
        for _, callback in callbacks.items():
            params = callback.params.dict
            # HACK
            if "gan" in params:
                params["gan"] = gan
            if "save_dir" in params:
                params["save_dir"] = results_dir
            # if "modes" in params:
            #     params["modes"] = dataset_info["modes"]

            afterall_callbacks.append(CallbackRegistry.create(callback.name, **params))

        if config.resume:
            results = np.loadtxt(Path(results_dir, "callback_results.txt")).tolist()
        else:
            results = [[] for _ in afterall_callbacks]

        for step in range(
            start_step_id * config.every,
            config.n_steps - int(config.burn_in_steps) + 1,
            config.every,
        ):
            x_file = Path(
                results_dir,
                "images",
                f"{step}.npy",
            )
            z_file = Path(
                results_dir,
                "latents",
                f"{step}.npy",
            )

            print(x_file)

            images = np.load(x_file)
            zs = np.load(z_file)

            info = dict(imgs=images, zs=zs, step=step, label=label)

            for callback_id, callback in enumerate(afterall_callbacks):
                val = callback.invoke(info)
                if val is not None:
                    results[callback_id].append(val)
        results = np.array(results)

        np.savetxt(
            Path(
                results_dir,
                "callback_results.txt",
            ),
            results,
        )

    if config.afterall_params.compute_fid:
        # model = InceptionV3().to(device)
        # model.eval()

        if config.resume:
            fid_values = np.loadtxt(Path(results_dir, "fid_values.txt")).tolist()
        else:
            fid_values = []
        for step in range(
            start_step_id * config.every,
            config.n_steps - int(config.burn_in_steps) + 1,
            config.every,
        ):
            file = Path(
                results_dir,
                "images",
                f"{step}.npy",
            )
            print(file)
            images = np.load(file)
            dataset = torch.from_numpy(images)
            print(dataset.shape)
            dataset = IgnoreLabelDataset(torch.utils.data.TensorDataset(dataset))

            # tf version
            stat_path = Path(
                "stats",
                f"{config.gan_config.dataset.name}",
                f"fid_stats_{config.gan_config.dataset.name}.npz",
            )
            fid = calculate_fid_given_paths(
                (stat_path.as_posix(), file.as_posix()),
                inception_path="thirdparty/TTUR/inception_model",
            )

            # torch version
            # mu, sigma, _ = get_activation_statistics(
            #     dataset, model, batch_size=100, device=device, verbose=True
            # )
            # fid = calculate_frechet_distance(
            #     mu, sigma, stats["mu"], stats["sigma"]
            # )
            print(f"Iter: {step}\t Fid: {fid}")
            # if wandb.run is not None:
            #     wandb.run.log({"step": step, "overall FID": fid})

            fid_values.append(fid)
            np.savetxt(
                Path(
                    results_dir,
                    "fid_values.txt",
                ),
                fid_values,
            )
    if not config.afterall_params.get("save_chains", False):
        im_paths = sorted(
            Path(results_dir, "images").glob("*.npy"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        lat_paths = sorted(
            Path(results_dir, "latents").glob("*.npy"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        for file_path in im_paths[:-1]:
            file_path.unlink()
        for file_path in lat_paths[:-1]:
            file_path.unlink()
        if not config.afterall_params.get("save_last_slice", False):
            im_paths[-1].unlink()
            lat_paths[-1].unlink()
            Path(results_dir, "labels.npy").unlink()

    plot_res(
        results_dir, config.gan_config, np.arange(0, config.n_steps + 1, config.every)
    )


def reset_anchors(args: argparse.Namespace, params: yaml.YAMLObject):
    if args.step_size:
        params["step_size"] = yaml.scalarfloat.ScalarFloat(
            args.step_size,
            prec=1,
            width=10,
            anchor=params["step_size"].anchor.value,
        )
    if args.resume:
        if "resume" in params:
            params["resume"] = yaml.scalarstring.LiteralScalarString(
                args.resume,
                anchor=params["resume"].anchor.value,
            )
        else:
            params["resume"] = args.resume
    if args.suffix:
        params["suffix"] = yaml.scalarstring.LiteralScalarString(args.suffix)


if __name__ == "__main__":
    args = parse_arguments()
    print(args.configs)
    # params = yaml.round_trip_load(Path(args.configs[0]).open("r"))
    # reset_anchors(args, params)
    # print(yaml.round_trip_dump(params))

    proc = subprocess.Popen("/bin/bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate(
        ("cat - " + " ".join([f"{conf} <(echo)" for conf in args.configs])).encode(
            "utf-8"
        )
    )
    config = yaml.round_trip_load(out.decode("utf-8"))
    print(yaml.round_trip_dump(config))
    config = DotConfig(config)

    if args.seed is not None:
        config.seed = args.seed
    config.file_name = Path(args.configs[0]).name
    config.resume = args.resume

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    group = args.group if args.group else f"{Path(args.configs[0]).stem}"
    main(config, device, group)
