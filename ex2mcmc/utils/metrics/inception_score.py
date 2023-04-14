"""
Code is partially borrowed from repository https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py # noqa: E501
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import yaml
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from yaml import Loader

from ex2mcmc.utils.callbacks import Callback, CallbackRegistry


N_INCEPTION_CLASSES = 1000
MEAN_TRASFORM = [0.485, 0.456, 0.406]
STD_TRANSFORM = [0.229, 0.224, 0.225]
N_GEN_IMAGES = 100005000


@CallbackRegistry.register()
class InceptionScoreCallback(Callback):
    def __init__(
        self,
        invoke_every: int = 1,
        device: Union[str, int, torch.device] = "cuda",
        update_input: bool = True,
        dp: bool = False,
        batch_size: int = 128,
    ) -> None:
        self.device = device
        self.model = torchvision.models.inception.inception_v3(
            pretrained=True, transform_input=False
        ).to(device)
        if dp:
            self.model = nn.DataParallel(self.model)

        self.model.eval()
        self.transform = transforms.Normalize(mean=MEAN_TRASFORM, std=STD_TRANSFORM)
        self.update_input = update_input
        self.invoke_every = invoke_every
        self.batch_size = batch_size

    @torch.no_grad()
    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        score = None
        step = info.get("step", self.cnt)
        if step % self.invoke_every == 0:
            imgs = torch.from_numpy(info["imgs"]).to(self.device)
            imgs = self.transform(imgs)
            pis = []
            for batch in torch.split(imgs, self.batch_size):
                pis_ = batch_inception(batch, self.model, resize=True).detach().cpu()
                pis.append(pis_)
            pis = torch.cat(pis, 0)
            score = (
                (pis * (torch.log(pis) - torch.log(pis.mean(0)[None, :])))
                .sum(1)
                .mean(0)
            )
            score = torch.exp(score)

            if self.update_input:
                info["inception_score"] = score
            logger = logging.getLogger()
            logger.info(f"\nIS: {score}")
        self.cnt += 1
        return score


def batch_inception(
    imgs: torch.Tensor,
    inception_model: nn.Module,
    resize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = imgs.device
    up = nn.Upsample(size=(299, 299), mode="bilinear").to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, -1)

    preds = get_pred(imgs)
    return preds


def get_inception_score(
    imgs: Iterable,
    inception_model: Optional[nn.Module] = None,
    gen: Optional[nn.Module] = None,
    generate_from_latents: bool = False,
    cuda: bool = True,
    batch_size: int = 32,
    resize: bool = False,
    splits: int = 1,
    device: Union[torch.device, int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N >= batch_size

    # Set up dataloader
    if not isinstance(imgs, torch.utils.data.Dataset):
        imgs = torch.utils.data.TensorDataset(imgs)

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    if inception_model is None:
        inception_model = inception_v3(
            pretrained=True,
            transform_input=False,  # False
        ).to(device)
        inception_model.eval()

    up = nn.Upsample(size=(299, 299), mode="bilinear").to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, -1).data.cpu()

    # Get predictions
    preds = torch.zeros((N, N_INCEPTION_CLASSES))

    for i, batch in enumerate(dataloader):
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device)
        if generate_from_latents:
            batch = gen(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = torch.mean(part, 0)
        split_scores.append(
            torch.exp(
                (part * (torch.log(part) - torch.log(py[None, :])))
                .sum(1)
                .mean(0)
                # torch.mean(torch.kl_div(part, torch.log(py[None, :])).sum(1))
            )
        )

    return (
        torch.mean(torch.stack(split_scores, 0)),
        torch.std(torch.stack(split_scores, 0)),
        preds,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_config", type=str)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.gan_config:
        from ex2mcmc.models.utils import GANWrapper
        from ex2mcmc.utils.general_utils import DotConfig

        gan_config = DotConfig(
            yaml.load(Path(args.gan_config).open("r"), Loader)
        ).gan_config
        # gen, _ = load_gan(gan_config, device)
        gan = GANWrapper(gan_config, device=device, load_weights=True)
        gen = gan.gen

        n_imgs = 1000  # 1000
        imgs = gen(torch.randn(n_imgs, gen.z_dim).to(device))
        imgs = gen.inverse_transform(imgs)

        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        imgs = transform(imgs)

        print("Calculating Inception Score...")
        print(
            get_inception_score(
                imgs,
                cuda=True,
                batch_size=32,
                resize=True,
                splits=1,
            )[:-1]
        )
    else:

        class IgnoreLabelDataset(torch.utils.data.Dataset):
            def __init__(self, orig):
                self.orig = orig

            def __getitem__(self, index):
                return self.orig[index][0]

            def __len__(self):
                return len(self.orig)

        cifar = dset.CIFAR10(
            root="data/",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Scale(32),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        IgnoreLabelDataset(cifar)

        print("Calculating Inception Score...")
        print(
            get_inception_score(
                IgnoreLabelDataset(cifar),
                cuda=True,
                batch_size=32,
                resize=True,
                splits=10,
            )[:-1]
        )
