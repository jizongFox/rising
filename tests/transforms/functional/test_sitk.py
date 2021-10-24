from rising.transforms.functional.sitk import itk_resample, itk_clip
import SimpleITK as sitk
import unittest
import numpy as np
from deepclustering2.viewer import multi_slice_viewer_debug


class SITKTestCase(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._image_path = "../../data/patient004_frame01.nii.gz"
        self._mask_path = "../../data/patient004_frame01_gt.nii.gz"
        self._sitk_image = sitk.ReadImage(self._image_path)
        self._sitk_mask = sitk.ReadImage(self._mask_path)

    def test_resampling(self):
        resampled_image = itk_resample(self._sitk_image, spacing=(0.1, 0.1, 10), interpolation="linear", pad_value=0)
        resampled_mask = itk_resample(self._sitk_mask, spacing=(0.1, 0.1, 10), interpolation="nearest", pad_value=0)
        assert np.allclose(np.unique(sitk.GetArrayFromImage(resampled_mask)), np.array([0, 1, 2, 3]))
        assert resampled_image.GetSpacing() == (0.1, 0.1, 10)

    def test_clip(self):
        clipped_image = itk_clip(self._sitk_image, 10, 600)
        multi_slice_viewer_debug(sitk.GetArrayFromImage(clipped_image).squeeze(), block=True)
