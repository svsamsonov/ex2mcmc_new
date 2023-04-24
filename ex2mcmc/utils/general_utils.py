import random
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


PROJECT_PATH = Path(__file__).parent.parent.parent
CHECKPOINT_DIR = Path(PROJECT_PATH, "checkpoints")
LOG_DIR = Path(PROJECT_PATH, "log")
IMGS_DIR = Path(PROJECT_PATH, "imgs")


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


def time_comp(fun):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fun(*args, **kwargs)
        tf = time.perf_counter()
        dt = tf - t0
        if kwargs.get("verbose"):
            print("Time elapsed: ", "{:.2f}".format(dt) + "s")

        return result

    return wrapper


def time_comp_cls(fun):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fun(*args, **kwargs)
        tf = time.perf_counter()
        dt = tf - t0
        if getattr(args[0], "verbose", False):
            print("Time elapsed: ", "{:.2f}".format(dt) + "s")

        return result

    return wrapper


class DotConfig(Mapping):
    """
    Simple wrapper for config
    allowing access with dot notation
    """

    def __init__(self, yaml):
        self._dict = yaml

    def __getattr__(self, key):
        if key in self.__dict__:
            return super().__getattr__(key)
        if key in self._dict:
            value = self._dict[key]
            if isinstance(value, dict):
                return DotConfig(value)
            return value
        else:
            return None

    def items(self):
        return [(k, DotConfig(v)) for k, v in self._dict.items()]

    def keys(self):
        return self._dict.keys()

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return self._dict.__iter__()

    def __getitem__(self, key):
        return self._dict[key]

    @property
    def dict(self):
        return self._dict

    def __contains__(self, key):
        return key in self._dict

    def __setitem__(self, key, value):
        self._dict[key] = value

    # def __assign__(self, key, value):
    #     print('ho')
    #     self._dict[key] = value

    # def __setattr__(self, key, value):
    #     print(key)
    #     self._dict[key] = value
