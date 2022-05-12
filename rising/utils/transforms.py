import typing
from typing import Iterable, Optional, Sequence, Union

import torch
from torch import nn

if typing.TYPE_CHECKING:
    from rising.transforms.abstract import AbstractTransform
    from rising.transforms.compose import Compose


def iter_transform(
    transforms: Union["Compose", "AbstractTransform", Sequence["AbstractTransform"], nn.ModuleList]
) -> Iterable["AbstractTransform"]:
    from rising.transforms.abstract import AbstractTransform
    from rising.transforms.compose import Compose

    if isinstance(transforms, AbstractTransform) and not isinstance(transforms, Compose):
        yield transforms
    elif isinstance(transforms, Compose):
        yield from iter_transform(transforms.transforms)
    elif isinstance(transforms, Sequence):
        for x in transforms:
            yield from iter_transform(x)
    elif isinstance(transforms, nn.ModuleList):
        for x in transforms:
            yield from iter_transform(x)
    else:
        raise TypeError(transforms)


def get_keys_from_transforms(transforms) -> Sequence[str]:
    _keys = [transform.keys for transform in iter_transform(transforms) if hasattr(transform, "keys")]
    keys = tuple(set([item for sublist in _keys for item in sublist]))
    return keys


def get_dtype_from_transforms(transforms) -> Optional[torch.dtype]:
    """
    this gives the dtype that the gpu transform should be converted, if ToDtypeTransform is given.
    """
    from rising.transforms import ToDtype

    for transform in iter_transform(transforms):
        if isinstance(transform, ToDtype):
            dtype = transform.dtype
            return dtype
