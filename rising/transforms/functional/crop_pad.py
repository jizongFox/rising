from typing import Sequence, Union

import torch

from rising.transforms.functional import random_crop


def pad_random_crop(
    data: torch.Tensor, size: Union[int, Sequence[int]], pad_size=Union[int, Sequence[int]], pad_value=0
):
    ndim = data.dim() - 2
    if isinstance(size, (float, int)):
        size = [size] * ndim
    if isinstance(pad_size, (float, int)):
        pad_size = [pad_size] * ndim
    assert len(size) == len(pad_size) == ndim

    from rising.transforms import Pad

    data = Pad(pad_size=[x + y for x, y in zip(size, pad_size)], pad_value=pad_value, keys="data")(data=data)["data"]
    return random_crop(
        data,
        size=size,
        dist=0,
    )
