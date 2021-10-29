import unittest

import torch

from rising.transforms.functional.pad import pad


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.ones(1, 1, 10, 10)
        self.complex_data = torch.ones(10, 3, 100, 100, 100)

    def test_constant_pad(self):
        for p in range(1, 4):
            padded = pad(self.data, pad_size=p, mode="constant", value=1.23)
            expected = torch.ones(1, 1, 10 + p * 2, 10 + p * 2)
            expected[:, :, 0:p, :] = 1.23
            expected[:, :, :, 0:p] = 1.23
            expected[:, :, -p:, :] = 1.23
            expected[:, :, :, -p:] = 1.23
            self.assertTrue((padded == expected).all())

    def test_complex_pad(self):
        for p in range(1, 4):
            padded = pad(self.complex_data, pad_size=p, mode="circular", )
            expected = torch.ones(10, 3, 100 + p * 2, 100 + p * 2, 100 + p * 2)
            self.assertTrue((padded == expected).all())


if __name__ == "__main__":
    unittest.main()
