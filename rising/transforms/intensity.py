from typing import Optional, Sequence, Union

from rising.random import AbstractParameter
from rising.transforms.abstract import (
    BaseTransform,
    BaseTransformMixin,
    PerChannelTransformMixin,
    PerSampleTransformMixin,
    TYPE_item_seq,
    augment_callable,
)
from rising.transforms.functional.intensity import (
    add_noise,
    add_value,
    augment_rician_noise,
    bezier_3rd_order,
    clamp,
    gamma_correction,
    norm_mean_std,
    norm_min_max,
    norm_min_max_percentile,
    norm_range,
    norm_zero_mean_unit_std,
    random_inversion,
    scale_by_value,
)

__all__ = [
    "Clamp",
    "NormRange",
    "NormPercentile",
    "NormMinMax",
    "NormZeroMeanUnitStd",
    "NormMeanStd",
    "_Noise",
    "GaussianNoise",
    "ExponentialNoise",
    "GammaCorrection",
    "_RandomValuePerChannel",
    "RandomAddValue",
    "RandomScaleValue",
    "RandomBezierTransform",
    "InvertAmplitude",
    "RicianNoiseTransform",
]


class Clamp(BaseTransformMixin, BaseTransform):
    """Apply augment_fn to keys"""

    def __init__(
        self,
        min: TYPE_item_seq[Union[float, AbstractParameter]],
        max: TYPE_item_seq[Union[float, AbstractParameter]],
        keys: Sequence = ("data",),
        grad: bool = False,
    ):
        """
        Args:
            min: minimal value
            max: maximal value
            keys: the keys corresponding to the values to clamp
            grad: enable gradient computation inside transformation
        """
        super().__init__(
            augment_fn=clamp,
            keys=keys,
            grad=grad,
            paired_kw_names=("min", "max"),
            augment_fn_names=("min", "max"),
            min=min,
            max=max,
        )


class NormRange(PerSampleTransformMixin, BaseTransform):
    def __init__(
        self,
        min: TYPE_item_seq[Union[float, AbstractParameter]],
        max: TYPE_item_seq[Union[float, AbstractParameter]],
        keys: Sequence = ("data",),
        per_channel: bool = True,
        per_sample=True,
        grad: bool = False,
    ):
        """
        Args:
            min: minimal value
            max: maximal value
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
        """
        super().__init__(
            augment_fn=norm_range,
            keys=keys,
            grad=grad,
            paired_kw_names=("min", "max"),
            augment_fn_names=("min", "max", "per_channel"),
            min=min,
            max=max,
            per_sample=per_sample,
            per_channel=per_channel,
        )


class NormPercentile(PerSampleTransformMixin, BaseTransform):
    """clamp the distribution based on percentile and normalize it between 0 and 1"""

    def __init__(
        self,
        min: TYPE_item_seq[Union[float, AbstractParameter]],
        max: TYPE_item_seq[Union[float, AbstractParameter]],
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        per_channel: bool = True,
        per_sample=True,
    ):
        """
        Args:
            min: min percentile, between 0 and 1
            max: max percentile, between 0 and 1
            per_channel: if normalize per channel
            per_sample: if normalize per sample or per batch
        """
        super().__init__(
            augment_fn=norm_min_max_percentile,
            keys=keys,
            grad=grad,
            min=min,
            max=max,
            per_channel=per_channel,
            paired_kw_names=("min", "max"),
            augment_fn_names=("min", "max", "per_channel"),
            per_sample=per_sample,
        )


class NormMinMax(PerSampleTransformMixin, BaseTransform):
    """Norm to [0, 1]"""

    def __init__(
        self,
        keys: Sequence = ("data",),
        per_channel: bool = True,
        per_sample=True,
        grad: bool = False,
        eps: Optional[float] = 1e-8,
    ):
        """
        Args:
            keys: keys to normalize
            per_channel: normalize per channel
            per_sample: normalize per sample or per batch
            grad: enable gradient computation inside transformation
            eps: small constant for numerical stability.
                If None, no factor constant will be added
        """
        super().__init__(
            augment_fn=norm_min_max,
            keys=keys,
            grad=grad,
            per_channel=per_channel,
            per_sample=per_sample,
            eps=eps,
            augment_fn_names=(
                "per_channel",
                "eps",
            ),
        )


class NormZeroMeanUnitStd(PerSampleTransformMixin, BaseTransform):
    """Normalize mean to zero and std to one"""

    def __init__(
        self,
        keys: Sequence = ("data",),
        per_channel: bool = True,
        per_sample=True,
        grad: bool = False,
        eps: Optional[float] = 1e-8,
    ):
        """
        Args:
            keys: keys to normalize
            per_channel: normalize per channel
            per_sample: normalize per sample or per batch.
            grad: enable gradient computation inside transformation
            eps: small constant for numerical stability.
                If None, no factor constant will be added
        """
        super().__init__(
            augment_fn=norm_zero_mean_unit_std,
            keys=keys,
            grad=grad,
            per_channel=per_channel,
            augment_fn_names=("eps", "per_channel"),
            eps=eps,
            per_sample=per_sample,
        )


class NormMeanStd(PerSampleTransformMixin, BaseTransform):
    """Normalize mean and std with provided values"""

    def __init__(
        self,
        mean: TYPE_item_seq[Union[float, Sequence[float]]],
        std: TYPE_item_seq[Union[float, Sequence[float]]],
        keys: Sequence[str] = ("data",),
        per_channel: bool = True,
        per_sample=True,
        grad: bool = False,
        **kwargs,
    ):
        """
        Args:
            mean: used for mean normalization
            std: used for std normalization
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(
            augment_fn=norm_mean_std,
            keys=keys,
            grad=grad,
            mean=mean,
            std=std,
            per_channel=per_channel,
            paired_kw_names=(
                "mean",
                "std",
            ),
            augment_fn_names=("mean", "std", "per_channel"),
            per_sample=per_sample,
            **kwargs,
        )


class GammaCorrection(BaseTransformMixin, BaseTransform):
    """Apply Gamma correction"""

    def __init__(self, gamma: Union[float, AbstractParameter], keys: Sequence = ("data",), grad: bool = False):
        """
        Args:
            gamma: define gamma
            keys: keys to normalize
            grad: enable gradient computation inside transformation
        """
        super().__init__(
            augment_fn=gamma_correction,
            keys=keys,
            grad=grad,
            augment_fn_names=("gamma",),
            gamma=gamma,
        )


class _Noise(PerChannelTransformMixin, BaseTransform):
    """
    Add noise to data

    .. warning:: This transform will apply different noise patterns to different keys.
    """

    def __init__(
        self, noise_type: str, per_channel: bool = False, keys: Sequence = ("data",), grad: bool = False, **kwargs
    ):
        """
        Args:
            noise_type: supports all inplace functions of a
                :class:`torch.Tensor`
            per_channel: enable transformation per channel
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to noise function

        See Also:
            :func:`torch.Tensor.normal_`, :func:`torch.Tensor.exponential_`
        """
        super().__init__(
            augment_fn=add_noise, per_channel=per_channel, keys=keys, grad=grad, noise_type=noise_type, **kwargs
        )


class ExponentialNoise(_Noise):
    """
    Add exponential noise to data

    .. warning:: This transform will apply different noise patterns to different keys.
    """

    def __init__(self, lambd: float, keys: Sequence = ("data",), grad: bool = False):
        """
        Args:
            lambd: lambda of exponential distribution
            keys: keys to normalize
            grad: enable gradient computation inside transformation
        """
        super().__init__(
            noise_type="exponential_",
            lambd=lambd,
            keys=keys,
            grad=grad,
            augment_fn_names=("noise_type", "lambd"),
        )


class GaussianNoise(_Noise):
    """
    Add gaussian noise to data

    .. warning:: This transform will apply different noise patterns to different keys.
    """

    def __init__(self, mean: float, std: float, keys: Sequence = ("data",), grad: bool = False):
        """
        Args:
            mean: mean of normal distribution
            std: std of normal distribution
            keys: keys to normalize
            grad: enable gradient computation inside transformation
        """
        super().__init__(
            noise_type="normal_",
            mean=mean,
            std=std,
            keys=keys,
            grad=grad,
            augment_fn_names=("noise_type", "mean", "std"),
        )


class _RandomValuePerChannel(PerChannelTransformMixin, BaseTransform):
    """
    Apply augmentations which take random values as input by keyword :attr:`value`

    .. warning:: This transform will apply different values to different keys.
    """

    def __init__(
        self,
        augment_fn: augment_callable,
        augment_fn_names: Sequence[str],
        random_sampler: AbstractParameter,
        per_channel: bool = False,
        keys: Sequence[str] = ("data",),
        grad: bool = False,
    ):
        """
        Args:
            augment_fn: augmentation function
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
        """
        super().__init__(
            augment_fn=augment_fn,
            augment_fn_names=augment_fn_names,
            per_channel=per_channel,
            keys=keys,
            grad=grad,
        )
        self.register_sampler("value", random_sampler)


class RandomAddValue(_RandomValuePerChannel):
    """
    Increase values additively

    .. warning:: This transform will apply different values to different keys.
    """

    def __init__(
        self,
        random_sampler: AbstractParameter,
        per_channel: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
    ):
        """
        Args:
            random_sampler: specify values to add
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
        """
        super().__init__(
            augment_fn=add_value,
            augment_fn_names=("value",),
            random_sampler=random_sampler,
            per_channel=per_channel,
            keys=keys,
            grad=grad,
        )


class RandomScaleValue(_RandomValuePerChannel):
    """
    Scale Values

    .. warning:: This transform will apply different values to different keys.
    """

    def __init__(
        self,
        random_sampler: AbstractParameter,
        per_channel: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs,
    ):
        """
        Args:
            random_sampler: specify values to add
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(
            augment_fn=scale_by_value,
            augment_fn_names=("value",),
            random_sampler=random_sampler,
            per_channel=per_channel,
            keys=keys,
            grad=grad,
        )


class RandomBezierTransform(BaseTransformMixin, BaseTransform):
    """Apply a random 3rd order bezier spline to the intensity values, as proposed in Models Genesis."""

    def __init__(
        self,
        maxv: float = 1.0,
        minv: float = 0.0,
        keys: Sequence = ("data",),
    ):
        super().__init__(
            augment_fn=bezier_3rd_order,
            keys=keys,
            grad=False,
            maxv=maxv,
            augment_fn_names=("maxv", "minv"),
            minv=minv,
        )


class InvertAmplitude(BaseTransformMixin, BaseTransform):
    """
    Inverts the amplitude with probability p according to the following formula:
    out = maxv + minv - data
    """

    def __init__(
        self,
        prob: float = 0.5,
        maxv: float = 1.0,
        minv: float = 0.0,
        keys: Sequence = ("data",),
    ):
        super().__init__(
            augment_fn=random_inversion,
            keys=keys,
            grad=False,
            prob_inversion=prob,
            maxv=maxv,
            minv=minv,
            augment_fn_names=("prob_inversion", "maxv", "minv"),
        )


class RicianNoiseTransform(PerSampleTransformMixin, BaseTransform):
    def __init__(
        self,
        *,
        keys: Sequence[str],
        grad: bool = False,
        std: Union[float, AbstractParameter],
        per_sample=True,
        p: float = 1,
        keep_range: bool = True,
    ):
        """Adds rician noise with the given std.
        The Noise of MRI data tends to have a rician distribution: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2254141/
        Args:
            std :  Union[float, AbstractParameter], samples std of Gaussian distribution used to calculate
            per_sample: if apply the noise per sample
            p: the probability of applying the transform.
            keep_range: if keep range of the image.
        CAREFUL: This transform will modify the value range of your data!

        adapted from batchgenerators:
        https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/noise_transforms.py
        """

        super().__init__(
            augment_fn=augment_rician_noise,
            p=p,
            keys=keys,
            grad=grad,
            augment_fn_names=("std", "keep_range"),
            per_sample=per_sample,
        )
        self.register_sampler("std", std)
        self.keep_range = keep_range
