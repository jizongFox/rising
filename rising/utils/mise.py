import collections
from itertools import repeat

import numpy as np
from torch import Tensor
import typing as t

T = t.TypeVar("T")


def ntuple(n: int) -> t.Callable[[t.Union[T, t.Sequence[T]]], t.Sequence[T]]:
    def parse(x: t.Union[T, t.Sequence[T]]) -> t.Sequence[T]:
        if isinstance(x, (Tensor, np.ndarray, str)):
            return tuple(repeat(x, n))
        if isinstance(x, collections.abc.Iterable):
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
