import torch
import functools
from torch import Tensor
from typing import Union


class fix_random_seed_ctx:
    def __init__(self, seed: Union[Tensor, int]) -> None:
        super().__init__()
        self._seed = int(seed)

    def __enter__(self):
        self._prev_seed = torch.random.get_rng_state()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.random.set_rng_state(self._prev_seed)  # noqa

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapped_func
