# from __future__ import annotations
from itertools import combinations
from typing import Callable, Optional, Sequence, Union

from torch.multiprocessing import Value

from rising.random import AbstractParameter, DiscreteParameter
from rising.transforms.abstract import BaseTransform, BaseTransformMixin, PerSampleTransformMixin, ItemSeq
from rising.transforms.functional import mirror, resize_native, rot90

__all__ = ["Mirror", "Rot90", "ResizeNative", "Zoom", "ProgressiveResize", "SizeStepScheduler"]

from rising.constants import FInterpolation

scheduler_type = Callable[[int], Union[int, Sequence[int]]]


class Mirror(PerSampleTransformMixin, BaseTransform):
    """Random mirror transform"""

    def __init__(
        self,
        *,
        dims=ItemSeq[Union[int, DiscreteParameter]],
        p_sample: float = 0.5,
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        per_sample: bool = True,
    ):
        """
        Args:
            dims: dimensions to apply random mirror
            p_sample: the probability of applying the mirror (0<=p_sample<=1), default=0.5
            per_sample: if applied per sample.
            keys: attributes to be applied.
        """
        super().__init__(
            augment_fn=mirror,
            keys=keys,
            grad=grad,
            augment_fn_names=("dims",),
            per_sample=per_sample,
            dims=dims,
            p=p_sample,
            seeded=True,
        )


class Rot90(PerSampleTransformMixin, BaseTransform):
    """Rotate 90 degree around dims"""

    def __init__(
        self,
        dims: ItemSeq[Union[Sequence[int], DiscreteParameter]],
        keys: Sequence[str] = ("data",),
        num_rots: DiscreteParameter = DiscreteParameter((0, 1, 2, 3)),
        p_sample: float = 0.5,
        per_sample: bool = True,
        grad: bool = False,
    ):
        """
        Args:
            dims: dims/axis ro rotate. If more than two dims are
                provided, 2 dimensions are randomly chosen at each call
            keys: keys which should be rotated
            num_rots: possible values for number of rotations
            prob: probability for rotation
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to superclass

        See Also:
            :func:`torch.Tensor.rot90`
        """
        if not isinstance(dims, DiscreteParameter):
            if len(dims) >= 2:
                dims = list(combinations(dims, 2))
            else:
                raise RuntimeError(f"dims must be at least 2 dims, given {dims}.")
            dims = DiscreteParameter(dims)
        super().__init__(
            augment_fn=rot90,
            keys=keys,
            grad=grad,
            augment_fn_names=("dims", "k"),
            per_sample=per_sample,
            dims=dims,
            p=p_sample,
            seeded=True,
            k=num_rots,
        )


class ResizeNative(BaseTransformMixin, BaseTransform):
    """Resize data to given size"""

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        mode: ItemSeq[FInterpolation] = FInterpolation.nearest,
        align_corners: ItemSeq[bool] = None,
        preserve_range: bool = False,
        keys: Sequence[str] = ("data",),
        grad: bool = False,
    ):
        """
        Args:
            size: spatial output size (excluding batch size and
                number of channels)
            mode: one of ``nearest``, ``linear``, ``bilinear``, ``bicubic``,
                ``trilinear``, ``area`` (for more inforamtion see
                :func:`torch.nn.functional.interpolate`) or their sequence, for different keys.
            align_corners: input and output tensors are aligned by the center \
                points of their corners pixels, preserving the values at the
                corner pixels. Input can be sequence, for different keys.
            preserve_range: output tensor has same range as input tensor
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(
            augment_fn=resize_native,
            keys=keys,
            grad=grad,
            size=size,
            mode=mode,
            align_corners=align_corners,
            preserve_range=preserve_range,
            p=1,
            augment_fn_names=("size", "mode", "align_corners", "preserve_range"),
            per_sample=False,
            paired_kw_names=("mode", "align_corners"),
        )
        assert self.p == 1.0


class Zoom(BaseTransformMixin, BaseTransform):
    """Apply augment_fn to keys. By default the scaling factor is sampled
    from a uniform distribution with the range specified by
    :attr:`random_args`
    """

    def __init__(
        self,
        scale_factor: Union[Sequence, AbstractParameter] = (0.75, 1.25),
        mode: ItemSeq[FInterpolation] = FInterpolation.nearest,
        align_corners: ItemSeq[bool] = None,
        preserve_range: bool = False,
        keys: Sequence[str] = ("data",),
        grad: bool = False,
    ):
        """
        Args:
            scale_factor: positional arguments passed for random function.
                If Sequence[Sequence] is provided, a random value for each item
                in the outer Sequence is generated. This can be used to set
                different ranges for different axis.
            mode: one of `nearest`, `linear`, `bilinear`,
                `bicubic`, `trilinear`, `area` (for more
                inforamtion see :func:`torch.nn.functional.interpolate`)
            align_corners: input and output tensors are aligned by the center
                points of their corners pixels, preserving the values at the
                corner pixels.
            preserve_range: output tensor has same range as input tensor
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn

        See Also:
            :func:`random.uniform`, :func:`torch.nn.functional.interpolate`
        """
        super().__init__(
            augment_fn=resize_native,
            keys=keys,
            grad=grad,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            preserve_range=preserve_range,
            p=1,
            augment_fn_names=("scale_factor", "mode", "align_corners", "preserve_range"),
            per_sample=False,
            paired_kw_names=("mode", "align_corners"),
        )
        assert self.p == 1.0


class ProgressiveResize(ResizeNative):
    """Resize data to sizes specified by scheduler"""

    def __init__(
        self,
        scheduler: scheduler_type,
        mode: ItemSeq[FInterpolation] = FInterpolation.nearest,
        align_corners: ItemSeq[Optional[bool]] = None,
        preserve_range: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
    ):
        """
        Args:
            scheduler: scheduler which determined the current size.
                The scheduler is called with the current iteration of the
                transform
            mode: one of ``nearest``, ``linear``, ``bilinear``, ``bicubic``,
                    ``trilinear``, ``area`` (for more inforamtion see
                    :func:`torch.nn.functional.interpolate`)
            align_corners: input and output tensors are aligned by the center
                points of their corners pixels, preserving the values at the
                corner pixels.
            preserve_range: output tensor has same range as input tensor
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn

        Warnings:
            When this transformations is used in combination with
            multiprocessing, the step counter is not perfectly synchronized
            between multiple processes.
            As a result the step count my jump between values
            in a range of the number of processes used.
        """
        super().__init__(
            size=0,
            mode=mode,
            align_corners=align_corners,
            preserve_range=preserve_range,
            keys=keys,
            grad=grad,
        )
        self.scheduler = scheduler
        self._step = Value("i", 0)

    def reset_step(self) -> ResizeNative:
        """
        Reset step to 0

        Returns:
            ResizeNative: returns self to allow chaining
        """
        with self._step.get_lock():
            self._step.value = 0
        return self

    def increment(self) -> ResizeNative:
        """
        Increment step by 1

        Returns:
            ResizeNative: returns self to allow chaining
        """
        with self._step.get_lock():
            self._step.value += 1
        return self

    @property
    def step(self) -> int:
        """
        Current step

        Returns:
            int: number of steps
        """
        return self._step.value

    def forward(self, **data) -> dict:
        """
        Resize data

        Args:
            **data: input batch

        Returns:
            dict: augmented batch
        """
        self.size = self.scheduler(self.step)
        self.increment()
        return super().forward(**data)


class SizeStepScheduler:
    """Scheduler return size when milestone is reached"""

    def __init__(self, milestones: Sequence[int], sizes: Union[Sequence[int], Sequence[Sequence[int]]]):
        """
        Args:
            milestones: contains number of iterations where size should be changed
            sizes: sizes corresponding to milestones
        """
        if len(milestones) != len(sizes) - 1:
            raise TypeError("Sizes must include initial size and thus " "has one element more than miltstones.")
        self.targets = sorted(zip((0, *milestones), sizes), key=lambda x: x[0], reverse=True)

    def __call__(self, step) -> Union[int, Sequence[int], Sequence[Sequence[int]]]:
        """
        Return size with regard to milestones

        Args:
            step: current step

        Returns:
            Union[int, Sequence[int], Sequence[Sequence[int]]]: current size
        """
        for t in self.targets:
            if step >= t[0]:
                return t[1]
        return self.targets[-1][1]
