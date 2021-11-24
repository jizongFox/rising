from typing import Sequence, Union

from rising.random import AbstractParameter
from rising.transforms.abstract import BaseTransform, BaseTransformMixin, PerSampleTransformMixin
from rising.transforms.functional.crop import center_crop, random_crop

__all__ = ["CenterCrop", "RandomCrop"]

from rising.transforms.functional.crop_pad import pad_random_crop


class CenterCrop(BaseTransformMixin, BaseTransform):
    def __init__(
        self, *, size: Union[int, Sequence, AbstractParameter], keys: Sequence = ("data",), grad: bool = False, **kwargs
    ):
        """
        Args:
            size: size of crop
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=center_crop, keys=keys, grad=grad, property_names=("size",), size=size, **kwargs)


class RandomCrop(PerSampleTransformMixin, BaseTransform):
    """
    todo: to enhance this function with padding function.
    """

    def __init__(
        self,
        *,
        size: Union[int, Sequence, AbstractParameter],
        dist: Union[int, Sequence, AbstractParameter] = 0,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            size: size of crop
            dist: minimum distance to border. By default zero
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(
            augment_fn=random_crop,
            keys=keys,
            grad=grad,
            size=size,
            dist=dist,
            seeded=True,
            augment_fn_names=("size", "dist"),
            **kwargs
        )


class PadRandomCrop(PerSampleTransformMixin, BaseTransform):
    def __init__(
        self,
        size: Union[int, Sequence, AbstractParameter],
        pad_size: Union[int, Sequence[int]] = 0,
        pad_value: int = 0,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            size: random crop size
            pad: int, sequence[int], padding image to size+pad
        """
        super(PadRandomCrop, self).__init__(
            augment_fn=pad_random_crop,
            keys=keys,
            grad=grad,
            size=size,
            pad_size=pad_size,
            pad_value=pad_value,
            augment_fn_names=("size", "pad_size", "pad_value"),
        )
