from pathlib import Path

import torch
from torch import nn

from ex2mcmc.utils.general_utils import DotConfig

from .base import MemoryModel, ModelRegistry


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class CondDataParallel(torch.nn.DataParallel):
    label = None
    cond = False

    def forward(self, *inputs, **kwargs):
        if self.cond:
            return super().forward(*inputs, **kwargs, label=self.label)
        else:
            return super().forward(*inputs, **kwargs)


class GANWrapper:
    def __init__(
        self, config: DotConfig, device: torch.device, load_weights=True, eval=True
    ):
        self.config = config
        self.device = device

        self.gen = ModelRegistry.create(
            config.generator.name, **config.generator.params
        ).to(device)
        self.dis = ModelRegistry.create(
            config.discriminator.name, **config.discriminator.params
        ).to(device)

        # for n, p in self.gen.named_parameters():
        #     if 'weight' in n:
        #         torch.nn.init.orthogonal_(p, gain=0.8)

        if load_weights:
            self.load_weights()
        # else:
        #     self.gen.apply(init_weights)
        #     self.dis.apply(init_weights)

        if config.dp:
            self.gen = CondDataParallel(self.gen)
            self.dis = CondDataParallel(self.dis)
            self.dis.transform = self.dis.module.transform
            self.dis.output_layer = self.dis.module.output_layer
            self.gen.inverse_transform = self.gen.module.inverse_transform
            self.gen.z_dim = self.gen.module.z_dim
            self.gen.sample_label = self.gen.module.sample_label
            if hasattr(self.gen.module, "label"):
                self.gen.cond = self.gen.module.label
            if hasattr(self.dis.module, "label"):
                self.dis.cond = self.dis.module.label
            if hasattr(self.dis.module, "penult_layer"):
                self.dis.penult_layer = self.dis.module.penult_layer
        self.dp = config.dp

        if eval:
            self.gen = MemoryModel(self.gen)
            self.dis = MemoryModel(self.dis)

            dis_attrs = ["transform", "output_layer", "label", "penult_layer"]
            self.dis.__dict__.update(
                {attr: self.dis.module.__dict__.get(attr) for attr in dis_attrs}
            )
            gen_attrs = ["inverse_transform", "z_dim", "label"]
            self.gen.__dict__.update(
                {attr: self.gen.module.__dict__.get(attr) for attr in gen_attrs}
            )
            self.gen.sample_label = self.gen.module.sample_label
            if hasattr(self.gen.module, "label"):
                self.gen.cond = self.gen.module.label
            if hasattr(self.dis.module, "label"):
                self.dis.cond = self.dis.module.label

            self.dis.transform = self.dis.module.transform
        print(f"Transform: {self.dis.transform}")

        if eval:
            self.eval()
        self.define_prior()
        self.label = None

    def load_weights(self):
        gen_path = Path(self.config.generator.ckpt_path)
        # if not gen_path.exists():
        #     subprocess.run(["dvc pull", gen_path.parent])

        state_dict = torch.load(gen_path, map_location=self.device)
        try:
            self.gen.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.gen.load_state_dict(state_dict, strict=True)

        dis_path = Path(self.config.discriminator.ckpt_path)
        # if not dis_path.exists():
        #     subprocess.run(["dvc pull", dis_path.parent])
        state_dict = torch.load(
            dis_path,
            map_location=self.device,
        )
        try:
            self.dis.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.dis.load_state_dict(state_dict, strict=True)

    def eval(self):
        for param in self.gen.parameters():
            param.requires_grad = False
        for param in self.dis.parameters():
            param.requires_grad = False
        self.gen.eval()
        self.dis.eval()

    def get_latent_code_dim(self):
        return self.gen.z_dim

    def define_prior(self):
        if self.config.prior == "normal":
            prior = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(self.gen.z_dim).to(self.device),
                torch.eye(self.gen.z_dim).to(self.device),
            )
            prior.project = lambda z: z
        elif self.config.prior == "uniform":
            prior = torch.distributions.uniform.Uniform(
                -torch.ones(self.gen.z_dim).to(self.device),
                torch.ones(self.gen.z_dim).to(self.device),
            )
            prior.project = lambda z: torch.clip(z, -1 + 1e-9, 1 - 1e-9)
            prior.log_prob = lambda z: torch.zeros(z.shape[0], device=z.device)
        else:
            raise KeyError
        self.gen.prior = prior

    @property
    def transform(self):
        return self.dis.transform

    @property
    def inverse_transform(self):
        return self.gen.inverse_transform

    @property
    def prior(self):
        return self.gen.prior

    def set_label(self, label):
        if self.config.dp:
            self.gen.label = label if self.gen.cond else None
            self.dis.label = label if self.dis.cond else None
            self.label = label if self.dis.cond else None
        else:
            self.gen.label = label
            self.dis.label = label
            self.label = label
