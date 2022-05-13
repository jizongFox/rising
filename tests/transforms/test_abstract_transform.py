import unittest
from typing import Sequence
from unittest.mock import Mock, call

import torch

from rising.random import DiscreteParameter
from rising.transforms import AbstractTransform, BaseTransform, PerChannelTransformMixin, PerSampleTransformMixin
from rising.transforms.abstract import BaseTransform2


class AddTransform(PerSampleTransformMixin, BaseTransform):
    def __init__(
        self,
        *,
        dims=[1],
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        property_names: Sequence[str] = (),
        key_associate_kwargs_names: Sequence[str] = ("dims",),
    ):
        super(AddTransform, self).__init__(
            augment_fn=sum_dim,
            keys=keys,
            grad=grad,
            property_names=property_names,
            key_associate_kwargs_names=key_associate_kwargs_names,
            per_sample=True,
            dims=dims,
            augment_fn_names=("dims",),
        )

    def forward(self, **data) -> dict:
        return super().forward(**data)


def sum_dim(data, dims=None, **kwargs):
    if dims is not None:
        dims = [d + 2 for d in dims]
        return torch.sum(data, dims, **kwargs)
    else:
        return data


class TestAbstractTransform(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.batch_dict = {
            "data": torch.rand(1, 1, 32, 32, requires_grad=True),
            "seg": torch.rand(1, 1, 32, 32),
            "label": torch.arange(3),
        }

    def test_abstract_transform(self):
        trafo = AbstractTransform(grad=False, internal0=True)
        self.assertTrue(trafo.internal0)
        with self.assertRaises(NotImplementedError):
            trafo(**self.batch_dict)

    def test_abstract_transform_no_grad(self):
        trafo = AddTransform(grad=False)
        output = trafo(**self.batch_dict)

        self.assertIsNone(output["data"]._grad_fn)
        with self.assertRaises(RuntimeError):
            output["data"].mean().backward()

    def test_abstract_transform_grad(self):
        trafo = AddTransform(grad=True)
        output = trafo(**self.batch_dict)
        self.assertIsNotNone(output["data"]._grad_fn)
        output["data"].mean().backward()
        self.assertIsNotNone(self.batch_dict["data"].grad)

    def test_base_transform(self):
        trafo = BaseTransform(augment_fn=lambda x: x + 50, keys=("data", "seg"))
        output = trafo(**self.batch_dict)

        diff_data = (output["data"] - self.batch_dict["data"]).mean().item()
        diff_seg = (output["seg"] - self.batch_dict["seg"]).mean().item()
        diff_label = (output["label"] - self.batch_dict["label"]).float().mean().item()

        self.assertEqual(diff_data, 50)
        self.assertEqual(diff_seg, 50)
        self.assertEqual(diff_label, 0)

    def test_per_sample_transform(self):
        mock = Mock(return_value=0)

        def augment_fn(inp, *args, **kwargs):
            return mock(inp)

        trafo = PerSampleTransformMixin(augment_fn=augment_fn, keys=("label",))
        output = trafo(**self.batch_dict)
        calls = [
            call(torch.tensor([0])),
            call(torch.tensor([1])),
            call(torch.tensor([2])),
        ]
        mock.assert_has_calls(calls)

    def test_per_channel_transform_per_channel_true(self):
        mock = Mock(return_value=0)

        def augment_fn(inp, *args, **kwargs):
            return mock(inp)

        trafo = PerChannelTransformMixin(augment_fn, per_channel=True, keys=("label",))
        self.batch_dict["label"] = self.batch_dict["label"][None]
        output = trafo(**self.batch_dict)
        calls = [
            call(torch.tensor([0])),
            call(torch.tensor([1])),
            call(torch.tensor([2])),
        ]
        mock.assert_has_calls(calls)

    def test_per_channel_transform_per_channel_false(self):
        mock = Mock(return_value=0)

        def augment_fn(inp, *args, **kwargs):
            return mock(inp)

        trafo = PerChannelTransformMixin(augment_fn, per_channel=False, keys=("label",))
        self.batch_dict["label"] = self.batch_dict["label"][None]
        output = trafo(**self.batch_dict)
        mock.assert_called_once()


class AddTransform2(PerSampleTransformMixin, BaseTransform2):
    def __init__(
        self,
        *,
        dims=1,
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        property_names: Sequence[str] = (),
    ):
        super(AddTransform2, self).__init__(
            augment_fn=sum_dim,
            keys=keys,
            grad=grad,
            property_names=property_names,
            key_associate_kwargs_names=("dims",),
            per_sample=True,
            dims=dims,
            augment_fn_names=("dims",),
        )

    def forward(self, **data) -> dict:
        return super().forward(**data)


class TestAbstractTransform2(unittest.TestCase):
    """
    this is a new interface for transforms that takes the input arguments as a iterable of the given keys
    """

    def setUp(self) -> None:
        # torch.manual_seed(0)
        self.batch_dict = {
            "data": torch.rand(1, 1, 2, 2, 2, 2, 2, 2, requires_grad=True),
            "seg": torch.rand(
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
            ),
            "label": torch.arange(3),
        }

    def test_abstract_transform_bind_paired_kwargs(self):
        trafo = AddTransform2(grad=False)
        output = trafo(**self.batch_dict)

        self.assertIsNone(output["data"]._grad_fn)
        with self.assertRaises(RuntimeError):
            output["data"].mean().backward()

    def test_paired_kwargs(self, *args, **kwargs):
        # trafo = AddTransform2(grad=False, dims=1)
        # param = trafo.get_paired_kwargs("data")
        # self.assertEqual(param, {"dims": 1})
        trafo = AddTransform2(grad=False, dims=[1, DiscreteParameter(list(range(34)))], keys=("data", "seg"))
        print(trafo.get_paired_kwargs())

        trafo = AddTransform2(grad=False, dims=[DiscreteParameter(list(range(34)))] * 2, keys=("data", "seg"))
        print(trafo.get_paired_kwargs())
        # trafo(**self.batch_dict)
        # self.assertEqual((param["dims"] in [1, 2,]), True)


if __name__ == "__main__":
    unittest.main()
