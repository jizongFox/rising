import unittest

import torch

from rising.random import NormalParameter, UniformParameter


class TestContinuous(unittest.TestCase):
    def test_uniform(self):
        self.check_distribution(UniformParameter(0, 2), torch.distributions.Uniform(0, 2))
        self.check_constant(UniformParameter(0, 0), 0)
        with self.assertRaises(AssertionError):
            self.check_constant(UniformParameter(0.0, 0.0), 1.0)
        with self.assertRaises(RuntimeError):
            self.check_constant(UniformParameter(0.0, 0.0), 0)

    def test_normal(self):
        self.check_distribution(NormalParameter(0, 2), torch.distributions.Normal(0, 2))
        self.check_constant(NormalParameter(0, 0), 0)
        with self.assertRaises(AssertionError):
            self.check_constant(NormalParameter(0.0, 0.0), 1.0)
        with self.assertRaises(RuntimeError):
            self.check_constant(NormalParameter(0.0, 0.0), 0)

    def check_distribution(self, param, dist, size=(10,)):
        state = torch.random.get_rng_state()
        res_param = param(size)

        torch.random.set_rng_state(state)
        res_dist = dist.sample(size)
        self.assertTrue(res_dist.allclose(res_param))

    def check_constant(self, param, constant, size=(10,)):
        res_params = param(size)
        assert res_params.shape == torch.Size(size)
        self.assertTrue(res_params.allclose(torch.ones(*size, dtype=torch.long) * constant))


if __name__ == "__main__":
    unittest.main()
