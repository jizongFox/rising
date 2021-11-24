from typing import Sequence, Union

import torch

from rising.transforms.abstract import BaseTransform, BaseTransformMixin, ITEM_or_SEQ
from rising.transforms.functional.pad import pad as _pad
from rising.utils.mise import ntuple


class Pad(BaseTransformMixin, BaseTransform):
    def __init__(
        self,
        *,
        pad_size: ITEM_or_SEQ[int],
        mode: str = "constant",
        pad_value: ITEM_or_SEQ[float],
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        padding_size should be the same for all the keys.
        pad_value can be different for different keys
        """
        super().__init__(
            augment_fn=_pad,
            keys=keys,
            grad=grad,
            paired_kw_names=("value", "mode"),
            augment_fn_names=("pad_size", "mode", "value"),
            pad_size=pad_size,
            mode=mode,
            value=pad_value,
            **kwargs
        )

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        seed = int(torch.randint(0, int(1e16), (1,)))

        for _key in self.keys:
            with self.random_cxm(seed):
                kwargs = {k: getattr(self, k) for k in self._augment_fn_names if k not in self._paired_kw_names}
                kwargs.update(self.get_pair_kwargs(_key))
                input_shape = data[_key].shape[2:]
                pad_size = self.pad_parameters(input_shape, self.pad_size, ndim=(data[_key].dim() - 2))
                kwargs.update({"pad_size": pad_size})

                if torch.rand(1).item() < self.p:
                    data[_key] = self.augment_fn(data[_key], **kwargs)
        return data

    @staticmethod
    def pad_parameters(input_shape: Union[int, Sequence[int]], resize_shape: Union[int, Sequence[int]], *, ndim: int):
        input_shape = ntuple(ndim)(input_shape)
        resize_shape = ntuple(ndim)(resize_shape)
        shape_difference = (max(0, r - i) for r, i in zip(resize_shape, input_shape))
        return [sub for y in [(x // 2, x - (x // 2)) for x in shape_difference] for sub in y][::-1]
