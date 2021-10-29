from rising.transforms import Pad
import unittest
import torch


class TestPad(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        image = torch.zeros(1, 1, 10, 10)
        self._data = {"data": image, "seg": image.clone()}

    def test_pad_no_pad(self):
        for pad_size in range(5, 10):
            transform = Pad(padding_size=pad_size, pad_value=(0, -1), keys=("data", "seg"))
            cropped = transform(**self._data)
            expected = torch.zeros(1, 1, 10, 10)
            assert torch.allclose(cropped["data"], expected)
            assert torch.allclose(cropped["seg"], expected)

    def test_pad(self):
        for pad_size in range(11, 15):
            transform = Pad(padding_size=pad_size, pad_value=(0, -1), keys=("data", "seg"))
            cropped = transform(**self._data)
            expected = torch.zeros(1, 1, pad_size, pad_size)
            expected_mask = torch.zeros_like(expected)
            expected_mask[:, :, 0:(pad_size - 10) // 2, :] = -1
            expected_mask[:, :, :, 0:(pad_size - 10) // 2] = -1
            expected_mask[:, :, :, -(pad_size - (pad_size - 10) // 2 - 10):] = -1
            expected_mask[:, :, -(pad_size - (pad_size - 10) // 2 - 10):, :] = -1
            assert torch.allclose(cropped["data"], expected)
            assert torch.allclose(cropped["seg"], expected_mask)
