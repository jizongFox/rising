from typing import Sequence, Tuple, Union

import torch

from rising.random.abstract import AbstractParameter
from rising.transforms.abstract import BaseTransform, ItemSeq
from rising.transforms.functional.sitk import itk2tensor, itk_clip, itk_resample

SpacingParamType = Union[
    int, Sequence[int], float, Sequence[float], torch.Tensor, AbstractParameter, Sequence[AbstractParameter]
]
SpacingTypeOrTuple = Union[SpacingParamType, Tuple[SpacingParamType, SpacingParamType, SpacingParamType]]

IntNumType = Union[int, AbstractParameter]


class _ITKTransform:
    """
    this mixin indicates if the transform is Tensor-based, use to not shuffle in Compose.
    """

    pass


class SITKResample(_ITKTransform, BaseTransform):
    """
    simpleitk resampling class
    """

    def __init__(
        self,
        spacing: SpacingParamType,
        *,
        pad_value: Union[int, float],
        keys: Sequence = ("data",),
        interpolation: ItemSeq[str] = "nearest",
    ):
        """
        resample simpleitk image given new spacing and padding value
        Args:
            spacing: float or tuple of three floats, indicating spacing for each dimension.
            pad_value: padding values
            interpolation: str or sequence of str to indicate the interpolation for different keys.
        """
        super().__init__(
            augment_fn=itk_resample,
            keys=keys,
            grad=False,
            spacing=spacing,
            pad_value=pad_value,
        )
        self.interpolation_mode = self.tuple_generator(interpolation)

    def forward(self, **data) -> dict:
        for key, interpolation in zip(self.keys, self.interpolation_mode):
            data[key] = self.augment_fn(
                data[key],
                spacing=self.spacing,
                interpolation=interpolation,
                pad_value=self.pad_value,
                **self.kwargs,
            )

        return data


class SITKWindows(_ITKTransform, BaseTransform):
    """
    simpleitk windows class
    """

    def __init__(self, low: IntNumType, high: IntNumType, *, keys: Sequence[str] = ("data",), **kwargs):
        super().__init__(augment_fn=itk_clip, keys=keys, grad=False, property_names=("low", "high"), low=low, high=high, **kwargs)

    def forward(self, **data) -> dict:
        kwargs = {}
        for k in self.property_names:
            kwargs[k] = getattr(self, k)

        kwargs.update(self.kwargs)

        # to make sure that in the sampling, there is case where `high` is lower than `low`.
        if kwargs["low"] > kwargs["high"]:
            kwargs["low"], kwargs["high"] = kwargs["high"], kwargs["low"]

        for _key in self.keys:
            data[_key] = self.augment_fn(data[_key], *self.args, **kwargs)
        return data


class SITK2Tensor(_ITKTransform, BaseTransform):
    def __init__(
        self,
        *,
        keys: Sequence = ("data",),
        dtype: ItemSeq[torch.dtype] = torch.float,
        insert_dim: int = None,
        grad: bool = False,
        **kwargs,
    ):
        """
        Convert sitk image to Tensor
        Args:
            dtype: tensor's dtype
            insert_dim: type: int, if you need to expand the tensor given specific dimension, default None,
        """
        super().__init__(augment_fn=itk2tensor, keys=keys, grad=grad, **kwargs)
        self.dtype = self.tuple_generator(dtype)
        self.insert_dim = insert_dim

    def forward(self, **data) -> dict:
        for key, dtype in zip(self.keys, self.dtype):
            data[key] = self.augment_fn(data[key], dtype=dtype)
            if self.insert_dim is not None:
                data[key] = data[key].unsqueeze(self.insert_dim)
        return data
