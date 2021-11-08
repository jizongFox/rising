"""
Provides the Augmentations and Transforms used by the
:class:`rising.loading.DataLoader`.

Implementations include:

* Transformation Base Classes
* Composed Transforms
* Affine Transforms
* Channel Transforms
* Cropping Transforms
* Device Transforms
* Format Transforms
* Intensity Transforms
* Kernel Transforms
* Spatial Transforms
* Tensor Transforms
* Utility Transforms
* Painting Transforms
"""

from rising.transforms.abstract import (
    BaseTransform,
    BaseTransformMixin,
    PerChannelTransformMixin,
    PerSampleTransformMixin,
    _AbstractTransform,
)
from rising.transforms.affine import Affine, BaseAffine, Resize, Rotate, Scale, StackedAffine, Translate
from rising.transforms.channel import ArgMax, OneHot
from rising.transforms.compose import Compose, DropoutCompose, OneOf
from rising.transforms.crop import CenterCrop, RandomCrop
from rising.transforms.format import FilterKeys, MapToSeq, PopKeys, RenameKeys, SeqToMap
from rising.transforms.grid import (
    CenterCropGrid,
    ElasticDistortion,
    GridTransform,
    RadialDistortion,
    RandomCropGrid,
    StackedGridTransform,
)
from rising.transforms.intensity import (
    Clamp,
    ExponentialNoise,
    GammaCorrection,
    GaussianNoise,
    InvertAmplitude,
    Noise,
    NormMeanStd,
    NormMinMax,
    NormPercentile,
    NormRange,
    NormZeroMeanUnitStd,
    RandomAddValue,
    RandomBezierTransform,
    RandomScaleValue,
    RandomValuePerChannel,
)
from rising.transforms.kernel import GaussianSmoothing, KernelTransform
from rising.transforms.pad import Pad
from rising.transforms.painting import LocalPixelShuffle, RandomInOrOutpainting, RandomInpainting, RandomOutpainting
from rising.transforms.sitk import SITK2Tensor, SITKResample, SITKWindows
from rising.transforms.spatial import Mirror, ProgressiveResize, ResizeNative, Rot90, SizeStepScheduler, Zoom
from rising.transforms.tensor import (
    Permute,
    TensorInsertDim,
    TensorOp,
    TensorRemoveDim,
    ToDevice,
    ToDeviceDtype,
    ToDtype,
    ToTensor,
)
from rising.transforms.utility import BoxToSeg, DoNothing, InstanceToSemantic, SegToBox
