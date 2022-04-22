import random
import unittest

import SimpleITK as sitk
import torch
from matplotlib import pyplot as plt

from rising.constants import FInterpolation
from rising.loading import DataLoader
from rising.random import DiscreteParameter, UniformParameter
from rising.transforms import Mirror, ProgressiveResize, ResizeNative, Rot90, SizeStepScheduler, Zoom, \
    ResizeNativeCentreCrop
from tests.realtime_viewer import multi_slice_viewer_debug


class TestSpatialTransforms(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        self.batch_dict = {
            "data": self.load_nii_data("../../tests/data/patient004_frame01.nii.gz"),
            "label": self.load_nii_data("../../tests/data/patient004_frame01_gt.nii.gz"),
        }

    def load_nii_data(self, path):
        return torch.from_numpy(
            sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(float, copy=False)
        ).unsqueeze(1)

    def test_mirror_transform(self):
        trafo = Mirror(dims=DiscreteParameter((0, 1, (0, 1))), p_sample=0.5, keys=("data", "label"))
        outp = trafo(**self.batch_dict)
        image1, target1 = self.batch_dict.values()
        image2, target2 = outp.values()
        multi_slice_viewer_debug(image1.squeeze(), target1.squeeze())
        multi_slice_viewer_debug(image2.squeeze(), target2.squeeze(), block=True)

    def test_rot90_transform(self):
        trafo = Rot90(dims=[0, 1], num_rots=DiscreteParameter((2,)), p_sample=0.5, keys=("data", "label"))
        outp = trafo(**self.batch_dict)
        image1, target1 = self.batch_dict.values()
        image2, target2 = outp.values()
        multi_slice_viewer_debug(image1.squeeze(), target1.squeeze())
        multi_slice_viewer_debug(image2.squeeze(), target2.squeeze(), block=True)

        trafo = Rot90(dims=[0, 1], num_rots=DiscreteParameter((2,)), p_sample=1, keys=("data", "label"))
        outp = trafo(**self.batch_dict)
        image1, target1 = self.batch_dict.values()
        image2, target2 = outp.values()
        multi_slice_viewer_debug(image1.squeeze(), target1.squeeze())
        multi_slice_viewer_debug(image2.squeeze(), target2.squeeze(), block=True)

    def test_resize_transform(self):
        trafo = ResizeNative(
            (128, 256),
            keys=(
                "data",
                "label",
            ),
            mode=(FInterpolation.bilinear, FInterpolation.nearest),
            align_corners=(False, None),
        )
        out = trafo(**self.batch_dict)
        image1, target1 = self.batch_dict.values()
        image2, target2 = out.values()
        multi_slice_viewer_debug(image1.squeeze(), target1.squeeze())
        multi_slice_viewer_debug(image2.squeeze(), target2.squeeze(), block=True, no_contour=True)

    def test_zoom_transform(self):
        _range = (1.5, 2.0)
        # scale_factor = UniformParameter(*_range)()

        trafo = Zoom(scale_factor=[UniformParameter(*_range), UniformParameter(*_range)], keys=("data", "label"))

        out = trafo(**self.batch_dict)
        image1, target1 = self.batch_dict.values()
        image2, target2 = out.values()
        multi_slice_viewer_debug(image1.squeeze(), target1.squeeze(), block=False, no_contour=True)
        multi_slice_viewer_debug(image2.squeeze(), target2.squeeze(), block=True, no_contour=True)

    def test_progressive_resize(self):
        image1, target1 = self.batch_dict.values()
        multi_slice_viewer_debug(image1.squeeze(), target1.squeeze(), no_contour=True)

        sizes = [1, 3, 6]
        scheduler = SizeStepScheduler([1, 2], [112, 224, 336])
        trafo = ProgressiveResize(scheduler, keys=("data", "label"))
        for i in range(3):
            outp = trafo(**self.batch_dict)
            image2, target2 = outp.values()
            multi_slice_viewer_debug(image2.squeeze(), target2.squeeze(), block=False, no_contour=True)

        plt.show()

    def test_size_step_scheduler(self):
        scheduler = SizeStepScheduler([10, 20], [16, 32, 64])
        self.assertEqual(scheduler(-1), 16)
        self.assertEqual(scheduler(0), 16)
        self.assertEqual(scheduler(5), 16)
        self.assertEqual(scheduler(11), 32)
        self.assertEqual(scheduler(21), 64)

    def test_size_step_scheduler_error(self):
        with self.assertRaises(TypeError):
            scheduler = SizeStepScheduler([10, 20], [32, 64])

    def test_progressive_resize_integration(self):
        sizes = [1, 3, 6]
        scheduler = SizeStepScheduler([1, 2], [1, 3, 6])
        trafo = ProgressiveResize(scheduler)

        dset = [self.batch_dict] * 10
        loader = DataLoader(dset, num_workers=4, batch_transforms=trafo)

        data_shape = [tuple(i["data"].shape) for i in loader]

        self.assertIn((1, 10, 1, 1, 1), data_shape)
        self.assertIn((1, 10, 3, 3, 3), data_shape)
        self.assertIn((1, 10, 6, 6, 6), data_shape)

    def test_resize_native_center_crop(self):
        trafo = ResizeNativeCentreCrop(size=(1000, 2000), margin=(10, 15), keys=("data", "label"),
                                       mode=(FInterpolation.bilinear, FInterpolation.nearest))
        outp = trafo(**self.batch_dict)
        image1, target1 = self.batch_dict.values()
        image2, target2 = outp.values()
        multi_slice_viewer_debug(image1.squeeze(), target1.squeeze())
        multi_slice_viewer_debug(image2.squeeze(), target2.squeeze(), block=True)


if __name__ == "__main__":
    unittest.main()
