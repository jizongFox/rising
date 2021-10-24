from typing import Sequence, Tuple, Union, Type

import torch

from rising.random.abstract import AbstractParameter
from rising.transforms.abstract import item_or_sequence, BaseTransform
from rising.transforms_ext.functional.sitk import itk_resample, itk_clip, itk2tensor

SpacingParamType = Union[
    int, Sequence[int], float, Sequence[float], torch.Tensor, AbstractParameter, Sequence[AbstractParameter]
]
SpacingTypeOrTuple = Union[SpacingParamType, Tuple[SpacingParamType, SpacingParamType, SpacingParamType]]

IntNumType = Union[int, AbstractParameter]


class SITKResample(BaseTransform):
    """
        simpleitk resampling class
    """

    def __init__(self, spacing: SpacingParamType, *, pad_value: Union[int, float], keys: Sequence = ("data",),
                 interpolation: item_or_sequence[str] = "nearest", **kwargs):
        """
        resample simpleitk image given new spacing and padding value
        Args:
            spacing: float or tuple of three floats, indicating spacing for each dimension.
            pad_value: padding values
            interpolation: str or sequence of str to indicate the interpolation for different keys.
        """
        super().__init__(augment_fn=itk_resample, keys=keys, grad=False, spacing=spacing, pad_value=pad_value,
                         property_names=("spacing", "pad_value"), **kwargs)
        self.interpolation_mode = self._tuple_generator(interpolation)

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


class SITKWindows(BaseTransform):
    """
    simpleitk windows class
    """

    def __init__(self, low: IntNumType, high: IntNumType, *, keys: Sequence[str] = ("data",), **kwargs):
        super().__init__(itk_clip, keys=keys, grad=False, **kwargs)
        self.register_sampler("low", low)
        self.register_sampler("high", high)


class SITK2Tensor(BaseTransform):

    def __init__(self, *, keys: Sequence = ("data",), dtype: item_or_sequence[Type[torch.float]] = torch.float,
                 grad: bool = False, **kwargs):
        super().__init__(itk2tensor, keys=keys, grad=grad, **kwargs)
        self.dtype = self._tuple_generator(dtype)

    def forward(self, **data) -> dict:
        for key, dtype in zip(self.keys, self.dtype):
            data[key] = self.augment_fn(
                data[key],
                dtype=dtype
            )
        return data
