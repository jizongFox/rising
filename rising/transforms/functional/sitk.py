from typing import Union, Tuple

import SimpleITK as sitk
import numpy as np
import torch

from rising.utils import check_scalar


def itk_resample(image: sitk.Image, spacing: Union[float, Tuple[float, float, float]], *,
                 interpolation: str = "nearest", pad_value: int) -> sitk.Image:
    """
    resample sitk image given spacing, pad value and interpolation.

    Args:
        image: sitk image
        spacing: new spacing, either a scalar or a tuple of three scalars.
        interpolation: interpolation method, "linear" or "nearest".
        pad_value: pad value for out of space pixels.

    Returns:
        torch.Tensor: affine params in correct shape
    """
    if check_scalar(spacing):
        spacing: Tuple[float, float, float] = (spacing, spacing, spacing)  # noqa

    ori_spacing = image.GetSpacing()
    ori_size = image.GetSize()
    new_size = (round(ori_size[0] * (ori_spacing[0] / spacing[0])),
                round(ori_size[1] * (ori_spacing[1] / spacing[1])),
                round(ori_size[2] * (ori_spacing[2] / spacing[2])))
    interp = {"linear": sitk.sitkLinear, "nearest": sitk.sitkNearestNeighbor, "cosine": sitk.sitkCosineWindowedSinc}[
        interpolation]
    return sitk.Resample(image, new_size, sitk.Transform(), interp, image.GetOrigin(), spacing, image.GetDirection(),
                         pad_value, image.GetPixelID())


def itk_clip(image: sitk.Image, low: int, high: int) -> sitk.Image:
    """
    clamp sitk image given low and high values, used to windows CT images.
    Args:
        image: sitk image
        low: low threshold to clip, type: int
        high: high threshold to clip, type: int
    Returns:
        sitk.Image
    """
    assert low < high, (low, high)
    return sitk.Clamp(image, sitk.sitkInt16, int(low), int(high))


def itk2tensor(image: sitk.Image, *, dtype=torch.float):
    np_array = sitk.GetArrayFromImage(image).astype(float, copy=False)
    return torch.from_numpy(np_array).to(dtype)[None, ...]
