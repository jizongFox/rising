from typing import Sequence, Union

from rising.transforms.abstract import BaseTransform, BaseTransformMixin, PerSampleTransformMixin
from rising.transforms.functional.crop import center_crop, random_crop
from rising.transforms.functional.crop_pad import pad_random_crop

__all__ = ["CenterCrop", "RandomCrop", "PadRandomCrop"]


class CenterCrop(BaseTransformMixin, BaseTransform):
    """
    CenterCrop input image given size.
    """

    def __init__(self, *, size: Union[int, Sequence], keys: Sequence = ("data",), grad: bool = False, **kwargs):
        """
        Args:
            size: size of crop
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=center_crop, keys=keys, grad=grad, augment_fn_names=("size",), size=size, **kwargs)


class RandomCrop(BaseTransformMixin, BaseTransform):
    """
    RandomCrop images given size and dist (distance to the border)
    """

    def __init__(
        self,
        *,
        size: Union[int, Sequence],
        dist: Union[int, Sequence] = 0,
        keys: Sequence = ("data",),
        grad: bool = False,
    ):
        """
        Args:
            size: size of crop
            dist: minimum distance to border. By default zero
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
        """
        super().__init__(
            augment_fn=random_crop,
            keys=keys,
            grad=grad,
            size=size,
            dist=dist,
            seeded=True,
            augment_fn_names=("size", "dist"),
        )


class PadRandomCrop(PerSampleTransformMixin, BaseTransform):
    """
    This operation pad the image and crop to the desired size.
    """

    def __init__(
        self,
        size: Union[int, Sequence],
        pad_size: Union[int, Sequence[int]] = 0,
        pad_value: Union[int, float, Sequence[int], Sequence[float]] = 0,
        keys: Sequence = ("data",),
        grad: bool = False,
    ):
        """
        Args:
            size: random crop size
            pad_size: int, sequence[int], padding image to size+pad
            pad_value: int, float or a list of them.  the value to pad
        """
        super(PadRandomCrop, self).__init__(
            augment_fn=pad_random_crop,
            keys=keys,
            grad=grad,
            size=size,
            pad_size=pad_size,
            pad_value=pad_value,
            augment_fn_names=("size", "pad_size", "pad_value"),
            paired_kw_names=("pad_value",),
        )
