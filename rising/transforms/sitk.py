from typing import Sequence, Tuple, Union

import torch

from rising.random.abstract import AbstractParameter
from rising.transforms.abstract import item_or_sequence, BaseTransform
from rising.transforms.functional.sitk import itk_resample, itk_clip, itk2tensor

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
        super().__init__(itk_clip, keys=keys, grad=False, property_names=("low", "high"), low=low, high=high, **kwargs)

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


class SITK2Tensor(BaseTransform):

    def __init__(self, *, keys: Sequence = ("data",), dtype: item_or_sequence = torch.float,
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
