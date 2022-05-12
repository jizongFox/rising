import random
import unittest
from math import isclose

import torch

from rising.random import DiscreteParameter
from rising.transforms import (
    Clamp,
    ExponentialNoise,
    GammaCorrection,
    GaussianNoise,
    NormMeanStd,
    NormMinMax,
    NormPercentile,
    NormRange,
    NormZeroMeanUnitStd,
    RandomAddValue,
    RandomScaleValue,
)
from tests.transforms import chech_data_preservation


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        self.batch_dict = {
            "data": torch.arange(1, 10).reshape(1, 1, 3, 3).float(),
            "seg": torch.rand(1, 1, 3, 3),
            "label": torch.arange(3),
        }

    def test_clamp_transform(self):
        trafo = Clamp(0, 1)

        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue((outp["data"] == torch.ones_like(outp["data"])).all())

    def test_norm_range_transform(self):
        trafo = NormRange(0.1, 0.2, per_channel=False)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = NormRange(0.1, 0.2, per_channel=True)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue(isclose(outp["data"].min().item(), 0.1, abs_tol=1e-6))
        self.assertTrue(isclose(outp["data"].max().item(), 0.2, abs_tol=1e-6))

    def test_norm_percentile_transform(self):
        trafo = NormPercentile(0.001, 0.99, per_channel=False)
        # out = trafo(**self.batch_dict)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = NormPercentile(0.001, 0.99, per_channel=True)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue(
            isclose(outp["data"].min().item(), torch.quantile(self.batch_dict["data"], 0.001), abs_tol=1e-6)
        )
        self.assertTrue(isclose(outp["data"].max().item(), torch.quantile(self.batch_dict["data"], 0.99), abs_tol=1e-6))

    def test_norm_min_max_transform(self):
        trafo = NormMinMax(per_channel=False)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = NormMinMax(per_channel=True)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue(isclose(outp["data"].min().item(), 0.0, abs_tol=1e-6))
        self.assertTrue(isclose(outp["data"].max().item(), 1.0, abs_tol=1e-6))

    def test_norm_zero_mean_transform(self):
        trafo = NormZeroMeanUnitStd(per_channel=False)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = NormZeroMeanUnitStd(per_channel=True)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue(isclose(outp["data"].mean().item(), 0.0, abs_tol=1e-6))
        self.assertTrue(isclose(outp["data"].std().item(), 1.0, abs_tol=1e-6))

    def test_norm_std_transform(self):
        mean = self.batch_dict["data"].mean().item()
        std = self.batch_dict["data"].std().item()
        trafo = NormMeanStd(mean=mean, std=std, per_channel=False)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = NormMeanStd(mean=mean, std=std, per_channel=True)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue(isclose(outp["data"].mean().item(), 0.0, abs_tol=1e-6))
        self.assertTrue(isclose(outp["data"].std().item(), 1.0, abs_tol=1e-6))

    def test_expoential_noise_transform(self):
        trafo = ExponentialNoise(lambd=0.0001)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))
        self.check_noise_distance(trafo)

    def test_gaussian_noise_transform(self):
        trafo = GaussianNoise(mean=75, std=1)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))
        self.check_noise_distance(trafo)

    def check_noise_distance(self, trafo, min_diff=50):
        outp = trafo(**self.batch_dict)
        comp_diff = (outp["data"] - self.batch_dict["data"]).mean().item()
        self.assertTrue(comp_diff > min_diff)

    def test_random_add_value(self):
        trafo = RandomAddValue(DiscreteParameter((2,)))
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        expected_out = self.batch_dict["data"] + 2.0
        self.assertTrue((outp["data"] == expected_out).all())

    def test_random_scale_value(self):
        trafo = RandomScaleValue(DiscreteParameter((2,)))
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        expected_out = self.batch_dict["data"] * 2.0
        self.assertTrue((outp["data"] == expected_out).all())

    def test_gamma_transform_scalar(self):
        trafo = GammaCorrection(gamma=2)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = GammaCorrection(gamma=2)
        outp = trafo(**self.batch_dict)
        expected_out = self.batch_dict["data"].pow(2)
        self.assertTrue((outp["data"] == expected_out).all())


if __name__ == "__main__":
    unittest.main()
