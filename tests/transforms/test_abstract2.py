import time
import unittest
from typing import Sequence, Union

import torch

from rising.transforms import BaseTransform, BaseTransformMixin, PerSampleTransformMixin
from rising.transforms.abstract import item_or_seq
from rising.transforms.functional import resize_native


class ResizeNative(BaseTransformMixin, BaseTransform):
    """Resize data to given size"""

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        mode: item_or_seq[str] = "nearest",
        align_corners: item_or_seq[bool] = None,
        preserve_range: bool = False,
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        **kwargs
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
            paired_kw_names=("mode", "align_corners"),
            augment_fn_names=("size", "mode", "align_corners"),
            **kwargs
        )


class TestBaseTransform(unittest.TestCase):
    def test_naive(self):
        image = torch.randn(3, 10, 224, 224, 224, device="cuda", dtype=torch.float16)
        target = torch.randint(0, 5, (3, 1, 224, 224, 224), device="cuda", dtype=torch.float16)
        cur_time = time.time()
        transform = ResizeNative(
            size=(16, 16, 16), mode=("trilinear", "nearest"), align_corners=None, keys=("data", "target")
        )
        transform(data=image, target=target)
        print(time.time() - cur_time)
