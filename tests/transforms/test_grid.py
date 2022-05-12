import typing as t
import unittest

import numpy as np
import torch
from PIL import Image
from deepclustering2.viewer import multi_slice_viewer_debug

from rising.transforms.grid import ElasticDistortion, RadialDistortion
from tests.utils.mise import gpu_timeit

T = t.TypeVar("T")
item_or_seq = t.Union[T, t.Sequence[T]]
float_or_seq = item_or_seq[int]


class TestGridCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.device = "cuda"
        self.dtype = torch.float16
        self._image = Image.open("/home/jizong/Workspace/rising/notebooks/MedNIST/BreastMRI/000000.jpeg").convert("L")
        self._image = torch.from_numpy(np.asarray(self._image), ).float()[None, None, ...]. \
            repeat(100, 1, 1, 1).to(self.device).to(self.dtype)
        self._target = (self._image > 0.5).to(self.dtype)

    def test_elastic_transform(self):
        transform = ElasticDistortion(std=2, alpha=0.1, keys=("data", "target"),
                                      interpolation_mode=("bilinear", "nearest"), per_sample=False).to(self.device).to(
            self.dtype)
        with gpu_timeit():
            for _ in range(100):
                output, target = transform(data=self._image, target=self._target).values()

    def test_radial_distortion(self):
        transform = RadialDistortion(scale=(0, 0, 0.3), keys=("data", "target"),
                                     interpolation_mode=("bilinear", "nearest")).to(self.device).to(self.dtype)

        with gpu_timeit():
            for _ in range(100):
                output, target = transform(data=self._image, target=self._target).values()
                multi_slice_viewer_debug([self._image.squeeze().float(), output.float().squeeze()],
                                         self._target.squeeze().float(), target.squeeze().float(), block=True,
                                         no_contour=True)
