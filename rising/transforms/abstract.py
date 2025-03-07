from abc import ABC, abstractmethod

try:
    from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union, final
except ImportError:
    from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union
    from typing_extensions import final

import torch
from torch import nn

from rising.random import AbstractParameter, DiscreteParameter
from rising.utils.mise import fix_seed_cxm, ntuple, nullcxm

__all__ = [
    "_AbstractTransform",
    "ItemSeq",
    "BaseTransform",
    "PerChannelTransformMixin",
    "PerSampleTransformMixin",
    "BaseTransformMixin",
]

T = TypeVar("T")
ItemSeq = Union[T, Sequence[T]]

augment_callable = Callable[..., Any]
augment_axis_callable = Callable[[torch.Tensor, Union[float, Sequence]], Any]


class _AbstractTransform(nn.Module):
    """Base class for all transforms"""

    def __init__(self, *, grad: bool = False, **kwargs):
        """
        Args:
            grad: enable gradient computation inside transformation
        """
        super().__init__()
        self.grad = grad
        self._registered_samplers: List[str] = []
        for key, item in kwargs.items():
            setattr(self, key, item)

    def register_sampler(self, name: str, sampler: Union[Sequence, AbstractParameter], *args, **kwargs):
        """
        Registers a parameter sampler to the transform.
        Internally a property is created to forward calls to the attribute to
        calls of the sampler.

        Args:
            name : the property name
            sampler : the sampler. Will be wrapped to a sampler always returning
                the same element if not already a sampler
            *args : additional positional arguments (will be forwarded to
                sampler call)
            **kwargs : additional keyword arguments (will be forwarded to
                sampler call)
        """
        if name in self._registered_samplers:
            raise ValueError(f"{name} has been registered as sampler.")
        self._registered_samplers.append(name)

        if not isinstance(sampler, (tuple, list)):
            sampler = [sampler]

        new_sampler = []
        for _sampler in sampler:
            if not isinstance(_sampler, AbstractParameter):
                _sampler = DiscreteParameter([_sampler], replacement=True)
            new_sampler.append(_sampler)
        sampler = new_sampler

        def sample(self):
            """
            Sample random values
            """
            sample_result = tuple([_sampler(*args, **kwargs) for _sampler in sampler])

            if len(sample_result) == 1:
                return sample_result[0]
            return sample_result

        if hasattr(self, name):
            delattr(self, name)
        setattr(self, name, property(sample))

    def need_sampler(self, value) -> bool:
        if isinstance(value, AbstractParameter):
            return True
        if isinstance(value, (list, tuple)):
            return any([self.need_sampler(x) for x in value])
        if isinstance(value, dict):
            raise NotImplementedError(value)
        else:
            return False

    @final
    def __getattribute__(self, item) -> Any:
        """
        Automatically dereference registered samplers

        Args:
            item: name of attribute

        Returns:
            Any: attribute
        """
        res = super().__getattribute__(item)
        if isinstance(res, property) and item in self._registered_samplers:
            # by first checking the type we reduce the lookup
            # time for all non property objects
            return res.__get__(self)
        else:
            return res

    @final
    def __call__(self, *args, **kwargs) -> Any:
        """
        Call super class with correct torch context

        Args:
            *args: forwarded positional arguments
            **kwargs: forwarded keyword arguments

        Returns:
            Any: transformed data

        """
        if self.grad:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()

        with context:
            return super().__call__(*args, **kwargs)

    @abstractmethod
    def forward(self, **data) -> dict:
        """
        Implement transform functionality here

        Args:
            **data: dict with data

        Returns:
            dict: dict with transformed data
        """
        raise NotImplementedError


class BaseTransform(_AbstractTransform, ABC):
    """
    Transform to apply a functional interface to given keys

    .. warning:: This transform should not be used
        with functions which have randomness build in because it will
        result in different augmentations per key.
    Modifications:
    Three kinds of attributes can be found here.
        1. The attribute which can be sampled (include AbstractParameters, Sequence[Abstract], int, float, Tensor etc.)
            These attributes can be sampled when calling __get__ magic function. stored in _registered_samplers.
        2. The attribute which is specific to each key, such as interpolation, mode, etc. These keys must be get
            per key. stored in _registered_key_pairs.
        3. The attribute which is normal.


    Since we have modified the seed manager with context manager,
    we can release the requirement of seeding.

    We need to pass a list of attributes to be passed to augment_fn ???

    """

    def __init__(
        self,
        *,
        augment_fn: augment_callable,
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        paired_kw_names: Sequence[str] = (),
        augment_fn_names: Sequence[str] = (),
        per_sample: bool = True,
        **kwargs,
    ):
        """
        Args:
            augment_fn: function for augmentation
                    Modification made here: all augment_fu accept data under form of BCHW(D) form.
            *args: positional arguments passed to augment_fn
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            property_names: a tuple containing all the properties to call
                during forward pass
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(grad=grad, **kwargs)
        sampler_vals = {k: v for k, v in kwargs.items() if self.need_sampler(v)}

        self._paired_kw_names: List[str] = []  # hidden list
        self._augment_fn_names = augment_fn_names  # kwargs passed to the augment_fn
        self.paired_kw_names = paired_kw_names

        self.augment_fn = augment_fn

        assert isinstance(keys, Sequence), keys
        self.keys = keys
        self.tuple_generator = ntuple(len(self.keys))

        self.per_sample = per_sample

        for kwarg_name in self.paired_kw_names:
            self.register_paired_attribute(kwarg_name, getattr(self, kwarg_name))

        for name, val in sampler_vals.items():
            self.register_sampler(name, val)  # lazy sampling values

    def sample_for_batch(self, name: str, batch_size: int) -> Optional[Union[Any, Sequence[Any]]]:
        """
        Sample elements for batch

        Args:
            name: name of parameter
            batch_size: batch size

        Returns:
            Optional[Union[Any, Sequence[Any]]]: sampled elements
        """
        elem = getattr(self, name)
        if elem is not None and self.per_sample:
            return [elem] + [getattr(self, name) for _ in range(batch_size - 1)]
        else:
            return elem  # either a single scalar value or None

    def register_paired_attribute(self, name: str, value: ItemSeq[T]):
        if name in self._paired_kw_names:
            raise ValueError(f"{name} has been registered in self._pair_kwarg_names")
        if name not in self._augment_fn_names:
            raise ValueError(f"{name} must be provided in `augment_fn_names`")
        self._paired_kw_names.append(name)
        setattr(self, name, self.tuple_generator(value))

    def get_pair_kwargs(self, key: str) -> Dict[str, Any]:
        assert key in self.keys, key
        index = self.keys.index(key)
        return {k: getattr(self, k)[index] for k in self._paired_kw_names}

    @abstractmethod
    def forward(self, **data) -> dict:
        """
        implementation override by mixin
        """
        raise NotImplementedError


class _BaseMixin(ABC):
    """
    base mixin class to perform the forward function.
    """

    per_sample: bool  # we have per_sample by default.
    get_pair_kwargs: Callable[[str], Dict[str, Any]]
    kwargs: Dict[str, Any]
    keys: Sequence[str]
    augment_fn: Callable  # all augment_fn take BCHWD as data input.

    _augment_fn_names: Sequence[str]
    _paired_kw_names: List[str]


class BaseTransformMixin(_BaseMixin):
    """
    this mixin call augment_fun and put all batch data into the data.
    you don't care about per_sample option.
    """

    def __init__(self, *, seeded: bool = True, p: float = 1, **kwargs) -> None:
        """
        Args:
            seeded: bool, default False. if the transformation need to fix the random seed for each key
            p: float: probability of applying augment_fn per batch
        """
        super().__init__(**kwargs)
        self.seeded = seeded
        assert 0 <= p <= 1, p
        self.p = p

    def forward(self, **data) -> Dict[str, Any]:
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

                if torch.rand(1).item() < self.p:
                    data[_key] = self.augment_fn(data[_key], **kwargs)
        return data

    @property
    def random_cxm(self):
        """random seed control context manager, if self.seeded."""
        return fix_seed_cxm if self.seeded else nullcxm


class PerSampleTransformMixin(BaseTransformMixin):
    """
    Transfer to a mixed in to override the forward of `BasesTransform`.

    1. random draw a seed as int
    3. fixed the seed + i as the beginning of the sample 1CBHWD

    """

    def __init__(self, *, p: float = 1, **kwargs):
        """
        Args:
            p: probability of applying the transform per sample.
        """
        super(PerSampleTransformMixin, self).__init__(**kwargs)
        assert 0 <= p <= 1, p
        self.p = p

    def forward(self, **data) -> dict:
        """
        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        if not self.per_sample:
            return super(PerSampleTransformMixin, self).forward(**data)

        seed = int(torch.randint(0, int(1e16), (1,)))

        for key in self.keys:
            batch_size = data[key].shape[0]
            out = []
            for b in range(batch_size):
                with self.random_cxm(seed + b):
                    kwargs = {k: getattr(self, k) for k in self._augment_fn_names if k not in self._paired_kw_names}
                    kwargs.update(self.get_pair_kwargs(key))

                    if torch.rand(1).item() < self.p:
                        out.append(self.augment_fn(data[key][b][None, ...], **kwargs))
                    else:
                        out.append(data[key][b][None, ...])

            data[key] = torch.cat(out, dim=0)
        return data


class PerChannelTransformMixin(BaseTransformMixin):
    """
    Transfer to a mixed in to override the forward of `BasesTransform`.

    This mixin gives augment_fn without per_channel attribute a chance to perform channel-wise operation.

    Apply transformation per channel (but still to whole batch)

    .. warning:: This transform should not be used
        with functions which have randomness build in because it will
        result in different augmentations per channel and key.
    """

    def __init__(self, *, per_channel: bool, p: float = 1, **kwargs):
        """
        Args:
            per_channel:bool parameter to perform per_channel operation
            kwargs: base parameters
        """
        super().__init__(**kwargs)
        self.per_channel = per_channel
        self.p = p

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        if not self.per_channel:
            return super().forward(**data)

        seed = int(torch.randint(0, int(1e16), (1,)))

        for key in self.keys:
            out = []
            channel_dim = data[key].shape[1]
            for c in range(channel_dim):
                with self.random_cxm(seed + c):
                    kwargs = {k: getattr(self, k) for k in self._augment_fn_names if k not in self._paired_kw_names}
                    kwargs.update(self.get_pair_kwargs(key))
                    if torch.rand(1).item() < self.p:
                        out.append(self.augment_fn(data[key][:, c].unsqueeze(1), **kwargs))
                    else:
                        out.append(data[key][:, c].unsqueeze(1))
            data[key] = torch.cat(out, dim=1)

        return data


class PerSamplePerChannelTransformMixin(BaseTransformMixin):
    def __init__(self, *, per_channel: bool, p_channel: float = 1, per_sample: bool, p_sample: float = 1, **kwargs):
        """
        Args:
            per_channel:bool parameter to perform per_channel operation
            kwargs: base parameters
        """
        super().__init__(**kwargs)
        self.per_channel = per_channel
        self.p_channel = p_channel

        self.per_sample = per_sample
        self.p_sample = p_sample

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        if not self.per_channel:
            self.p = self.p_sample
            return PerSampleTransformMixin.forward(self, **data)
        if not self.per_sample:
            self.p = self.p_channel
            return PerChannelTransformMixin.forward(self, **data)

        seed = int(torch.randint(0, int(1e16), (1,)))

        for key in self.keys:
            batch_size, channel_dim = data[key].shape[0:2]
            for b in range(batch_size):
                cur_data = data[key]
                processed_batch = []
                for c in range(channel_dim):
                    with self.random_cxm(seed + b + c):
                        kwargs = {k: getattr(self, k) for k in self._augment_fn_names if k not in self._paired_kw_names}
                        kwargs.update(self.get_pair_kwargs(key))
                        if torch.rand(1).item() < self.p:
                            processed_batch.append(self.augment_fn(cur_data[b, c][None, None, ...], **kwargs))
                        else:
                            processed_batch.append(data[key][:, c][None, None, ...])
                data[key][b] = torch.cat(processed_batch, dim=1)[0]

        return data
