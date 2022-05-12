import collections
import functools
import random
import typing as t
from contextlib import AbstractContextManager, contextmanager
from itertools import repeat

import numpy as np
import torch
from torch import Tensor
from torch import multiprocessing as mp
from torch import nn

from rising.random.abstract import AbstractParameter

__all__ = ["ntuple", "single", "pair", "triple", "quadruple", "fix_seed_cxm", "nullcxm"]

T = t.TypeVar("T")


class nullcxm(AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *excinfo):
        pass


def ntuple(n: int) -> t.Callable[[t.Union[T, t.Sequence[T]]], t.Sequence[T]]:
    def parse(x: t.Union[T, t.Sequence[T]]) -> t.Sequence[T]:
        if isinstance(x, AbstractParameter):
            return nn.ModuleList([x])
        if isinstance(x, (Tensor, np.ndarray, str)):
            return tuple(repeat(x, n))
        if isinstance(x, collections.Iterable):
            item_list = tuple(x)
            if len(item_list) == n:
                return item_list
            if len(item_list) == 1:
                return tuple(repeat(item_list[0], n))
            if len(list(x)) != n:
                raise RuntimeError(f"Iterable shape inconsistent, n = {n}, given {len(list(x))}")

        return tuple(repeat(x, n))

    return parse


single = ntuple(1)
pair = ntuple(2)
triple = ntuple(3)
quadruple = ntuple(4)


class fixed_torch_seed:
    """
    fixed random seed for torch module
    """

    def __init__(self, seed: int = 10, cuda: bool = True) -> None:
        super().__init__()
        self.seed = seed
        self.cuda_flag = cuda and torch.cuda.is_available()

    def __enter__(self):
        seed = self.seed
        self.__pre_state = torch.get_rng_state()
        self.__pre_cuda_state_all = None
        if self.cuda_flag:
            self.__pre_cuda_state_all = torch.cuda.get_rng_state_all()

        torch.manual_seed(seed)
        if self.cuda_flag:
            torch.cuda.manual_seed_all(seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.__pre_state)
        if self.cuda_flag:
            torch.cuda.set_rng_state_all(self.__pre_cuda_state_all)

    def __call__(self, func):
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            with self:
                gen = func(*args, **kwargs)
            return gen

        return generator_context


class fixed_random_seed:
    """
    fixed random seed for random module
    """

    def __init__(self, seed: int = 10, **kwargs) -> None:
        super().__init__()
        self.seed = seed

    def __enter__(self):
        seed = self.seed
        self.__pre_state = random.getstate()
        random.seed(seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.__pre_state)

    def __call__(self, func):
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            with self:
                gen = func(*args, **kwargs)
            return gen

        return generator_context


def on_main_process():
    return mp.current_process().name == "MainProcess"


@contextmanager
def fix_seed_cxm(seed: int = 10):
    cuda = on_main_process() and torch.cuda.is_available()
    with fixed_torch_seed(seed=seed, cuda=cuda), fixed_random_seed(seed=seed):
        yield
