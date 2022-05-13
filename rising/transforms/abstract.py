from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union, cast

from torch.nn import ModuleList

from rising.utils.checktype import to_scalar
from rising.utils.shape import check_tensor_dim

try:
    from typing import final
except ImportError:
    from typing_extensions import final

import torch
from torch import Tensor, nn

from rising.random import AbstractParameter, DiscreteParameter
from rising.utils.mise import fix_seed_cxm, ntuple, nullcxm

__all__ = [
    "AbstractTransform",
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

ParameterType = Union[AbstractParameter, int, float, Tensor]
ParameterTypeBoard = ItemSeq[ParameterType]


class _Abstract(nn.Module):
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


class AbstractTransform(_Abstract, ABC):
    """Base class for all transforms"""

    def register_sampler(self, name: str, sampler: ParameterTypeBoard, *args, same_per_call: bool = False, **kwargs):
        """
        Registers a parameter sampler to the transform.
        Internally a property is created to forward calls to the attribute to
        calls of the sampler.

        All passed sampler would be set to AbstractParameter and to save into base class

        Args:
            name : the property name
            sampler : the sampler. Will be wrapped to a sampler always returning
                the same element if not already a sampler
            *args : additional positional arguments (will be forwarded to
                sampler call)
            same_per_call : if True, the sampler will be called per sample, instead, it would use the first sampling
                of the list, useful to create the same parameter for different key
            **kwargs : additional keyword arguments (will be forwarded to
                sampler call)
        """
        if name in self._registered_samplers:
            raise ValueError(f"{name} has been registered as sampler.")
        self._registered_samplers.append(name)

        __added_dim = False
        if not isinstance(sampler, (tuple, list, ModuleList)):
            assert isinstance(sampler, (AbstractParameter, str, int, float, torch.Tensor)), sampler
            sampler = [sampler]
            __added_dim = True

        new_sampler = []
        for _sampler in sampler:
            if not isinstance(_sampler, AbstractParameter):
                _sampler = DiscreteParameter([_sampler], replacement=True)
            new_sampler.append(_sampler)

        sampler = cast(List[AbstractParameter], new_sampler)

        def sample_from_sequence(self):
            """
            Sample random values
            """
            seed = to_scalar(torch.randint(0, 2**32 - 1, (1,), dtype=torch.int64))
            if not same_per_call:
                with self.random_cxm(seed):
                    sample_result = tuple([_sampler(*args, **kwargs) for _sampler in sampler])
            # sample from AbstractParameter
            else:
                with self.random_cxm(seed):
                    sample_result = tuple([sampler[0](*args, **kwargs)] * len(self))
            # same from the first element

            if __added_dim:
                return sample_result[0]
            return sample_result

        if hasattr(self, name):
            delattr(self, name)
        setattr(self, name, property(sample_from_sequence))

    def need_sampler(self, value) -> bool:
        if isinstance(value, AbstractParameter):
            return True
        if isinstance(value, (list, tuple)):
            return any([self.need_sampler(x) for x in value])
        if isinstance(value, dict):
            raise NotImplementedError(value)
        else:
            return False


class BaseTransform(AbstractTransform, ABC):
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
        self.augment_fn = augment_fn

        self._augment_fn_names = augment_fn_names  # kwargs passed to the augment_fn

        assert isinstance(keys, Sequence), keys
        self.keys = keys
        self.tuple_generator = ntuple(self.num_keys)

        self.per_sample = per_sample

        # todo: to simplify this part, to make every kwargs to arg_function to be iterable and index-able
        for name in augment_fn_names:
            sequenced_param = self.register_paired_attribute(name, getattr(self, name))
            self.register_sampler(name, sequenced_param, same_per_call=self.same_per_call(getattr(self, name)))

    def same_per_call(self, sampler) -> bool:
        if isinstance(sampler, AbstractParameter) and len(self.keys) > 1:
            return True
        if (isinstance(sampler, Sequence) and not isinstance(sampler, str)) and len(sampler) == 1:
            return self.same_per_call(sampler[0])
        return False

    def sample_for_batch(self, name: str, batch_size: int) -> Optional[Sequence[Any]]:
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
        if elem is None:
            return None
        return [elem] * batch_size

    def register_paired_attribute(self, name: str, value: ItemSeq[T]):
        if name not in self._augment_fn_names:
            raise ValueError(f"{name} must be provided in `augment_fn_names`")
        return self.tuple_generator(value)

    def _get_sequenced_kwargs(self, /, key: str) -> Dict[str, Any]:
        assert key in self.keys, key
        index = self.keys.index(key)
        return {k: getattr(self, k)[index] for k in self._augment_fn_names}

    def get_sequenced_kwargs(self, key: str = None) -> Dict[str, Any]:
        if key is not None:
            return self._get_sequenced_kwargs(key=key)
        result = {}
        seed = to_scalar(torch.randint(0, 2**32 - 1, (1,), dtype=torch.int64))
        for key in self.keys:
            with self.random_cxm(seed):
                result.update({key: self._get_sequenced_kwargs(key=key)})
        return result

    @abstractmethod
    def forward(self, **data) -> dict:
        """
        implementation override by mixin
        """
        raise NotImplementedError

    @property
    def num_keys(self) -> int:
        return len(self.keys)

    def __len__(self):
        return self.num_keys

    @property
    def seeded(self) -> bool:
        if self.num_keys >= 2:
            return True
        return False

    @property
    def random_cxm(self):
        """random seed control context manager, if self.seeded."""
        return fix_seed_cxm if self.seeded else nullcxm


if TYPE_CHECKING:
    _MIXIN_BASE = BaseTransform
else:
    _MIXIN_BASE = ABC

tensor_dim_check = False


class BaseTransformMixin(_MIXIN_BASE):
    """
    this mixin call augment_fun and put all batch data into the data.
    you don't care about per_sample option.
    """

    def __init__(self, *, p: float = 1, tensor_dim_check: bool = tensor_dim_check, **kwargs) -> None:
        """
        Args:
            p: float: probability of applying augment_fn per batch
            tensor_dim_check: bool, default False. if the transformation need to check the tensor dimension
        """
        super().__init__(**kwargs)
        assert 0 <= p <= 1, p
        self.p = p
        self._tensor_dim_check = tensor_dim_check

    def forward(self, **data) -> Dict[str, Any]:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        seed = to_scalar(torch.randint(0, int(1e16), (1,)))

        for key in self.keys:
            with self.random_cxm(seed):

                kwargs = self.get_sequenced_kwargs(key)

                if to_scalar(torch.rand(1)) < self.p:
                    if self._tensor_dim_check:
                        assert check_tensor_dim(data[key])

                    data[key] = self.augment_fn(data[key], **kwargs)
        return data


class PerSampleTransformMixin(BaseTransformMixin):
    """
    Transfer to a mixed in to override the forward of `BasesTransform`.

    1. random draw a seed as int
    3. fixed the seed + i as the beginning of the sample 1CBHWD

    """

    def __init__(self, *, p: float = 1, tensor_dim_check: bool = tensor_dim_check, **kwargs):
        """
        Args:
            p: probability of applying the transform per sample.
        """
        super(PerSampleTransformMixin, self).__init__(p=p, tensor_dim_check=tensor_dim_check, **kwargs)

    def forward(self, **data) -> dict:
        """
        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        if not self.per_sample:
            return super(PerSampleTransformMixin, self).forward(**data)

        seed = to_scalar(torch.randint(0, int(1e16), (1,)))

        for key in self.keys:
            batch_size = data[key].shape[0]
            out = []
            for b in range(batch_size):
                with self.random_cxm(seed + b):
                    kwargs = self.get_sequenced_kwargs(key)
                    cur_input = data[key][b][None, ...]
                    if self._tensor_dim_check:
                        assert check_tensor_dim(cur_input)

                    if to_scalar(torch.rand(1)) < self.p:
                        out.append(self.augment_fn(cur_input, **kwargs))
                    else:
                        out.append(cur_input)

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

    def __init__(
        self, *, p: float = 1, per_channel: bool = True, tensor_dim_check: bool = tensor_dim_check, **kwargs
    ) -> None:
        """
        Args:
            p: probability of applying the transform per sample.
            per_channel: whether to apply the transform per channel.
            tensor_dim_check: whether to check the tensor dimension.
        """
        super().__init__(p=p, tensor_dim_check=tensor_dim_check, **kwargs)
        self.per_channel = per_channel

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

        seed = to_scalar(torch.randint(0, int(1e16), (1,)))

        for key in self.keys:
            out = []
            channel_dim = data[key].shape[1]
            for c in range(channel_dim):
                with self.random_cxm(seed + c):
                    kwargs = self.get_sequenced_kwargs(key)
                    cur_input = data[key][:, c].unsqueeze(1)
                    if self._tensor_dim_check:
                        assert check_tensor_dim(cur_input)

                    if to_scalar(torch.rand(1)) < self.p:
                        out.append(self.augment_fn(cur_input, **kwargs))
                    else:
                        out.append(cur_input)
            data[key] = torch.cat(out, dim=1)

        return data


class PerSamplePerChannelTransformMixin(BaseTransformMixin):
    def __init__(
        self,
        *,
        per_channel: bool = True,
        p_channel: float = 1,
        per_sample: bool = True,
        p: float = 1,
        check_tensor_dim: bool = tensor_dim_check,
        **kwargs,
    ) -> None:
        """
        Args:
            per_channel:bool parameter to perform per_channel operation
            kwargs: base parameters
        """
        super().__init__(**kwargs, p=p, check_tensor_dim=check_tensor_dim)
        self.per_channel = per_channel
        self.p_channel = p_channel

        self.per_sample = per_sample
        self.p_sample = p

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
                        kwargs = self.get_sequenced_kwargs(key)
                        cur_input = cur_data[b, c][None, None, ...]
                        if self._tensor_dim_check:
                            assert check_tensor_dim(cur_input)
                        if to_scalar(torch.rand(1)) < self.p:
                            processed_batch.append(self.augment_fn(cur_input, **kwargs))
                        else:
                            processed_batch.append(cur_input)
                data[key][b] = torch.cat(processed_batch, dim=1)[0]

        return data
