import unittest

import torch

from rising.random import DiscreteCombinationsParameter, DiscreteParameter
from rising.random.discrete import combinations_all


class TestDiscrete(unittest.TestCase):
    def test_discrete_error(self):
        with self.assertRaises(ValueError) as e:
            param = DiscreteParameter((1.0, 2.0), replacement=False, weights=("yes", 0.7))
        expected_msg = "weights and cum_weights should only be specified if replacement is set to True!"
        assert e.exception.args[0] == expected_msg

    def test_discrete_parameter(self):
        param = DiscreteParameter((1,), replacement=True)
        sample = param(size=(100, 100))
        assert sample.allclose(torch.ones_like(sample) * 1)

    def test_discrete_parameter2(self):
        param = DiscreteParameter((1, 2), replacement=True)
        sample = param()
        assert sample in (1, 2)

    def test_discrete_combinations_parameter(self):
        param = DiscreteCombinationsParameter((1,))
        sample = param()
        self.assertEqual(sample, 1)

    def test_combination_all(self):
        combs = combinations_all((0, 1))
        self.assertIn((0,), combs)
        self.assertIn((1,), combs)
        self.assertIn((0, 1), combs)
        self.assertEqual(len(combs), 3)


if __name__ == "__main__":
    unittest.main()
