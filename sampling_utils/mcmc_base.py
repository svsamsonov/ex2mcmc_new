from abc import ABC, abstractmethod

import torch


def increment_steps(call):
    def wrapper(self, start: torch.Tensor, target, proposal, *args, **kwargs):
        if "n_steps" in kwargs:
            n_steps = kwargs.get("n_steps")
        elif len(args) > 0:
            n_steps = args[0]
        elif "n_steps" in self.__dict__:
            n_steps = self.__dict__["n_steps"]
        else:
            raise TypeError("Missing argument 'n_steps'")
        out = call(self, start, target, proposal, n_steps, *args, **kwargs)
        self._steps_done += n_steps
        return out

    return wrapper


def adapt_stepsize_dec(call):
    def wrapper(self, *args, **kwargs):
        adapt_stepsize = kwargs.get("adapt_stepsize")

        if (
            self.adapt_stepsize is True and adapt_stepsize is not False
        ) or adapt_stepsize is True:
            adapt_stepsize = True
        else:
            adapt_stepsize = False

        if adapt_stepsize and self._steps_done > 0:
            grad_step = self.grad_step
            noise_scale = (2 * self.grad_step) ** 0.5

            kwargs["grad_step"] = grad_step
            kwargs["noise_scale"] = noise_scale

        out = call(self, *args, **kwargs)
        grad_step = out[-1]

        if adapt_stepsize:
            self.grad_step = grad_step

        return out

    return wrapper


class AbstractMCMC(ABC):
    _steps_done = 0

    def __init__(self, verbose=True, **kwargs):
        self._steps_done = 0
        self.verbose = verbose

    @abstractmethod
    @increment_steps
    @adapt_stepsize_dec
    def __call__(
        self,
        start: torch.Tensor,
        target,
        proposal,
        n_steps: int,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    @property
    def steps_done(self):
        return self._steps_done
