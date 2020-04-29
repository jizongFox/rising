from .abstract import AbstractTransform
from typing import Union, Sequence, Callable, Tuple

from rising.transforms.functional.utility import pop_keys, filter_keys

__all__ = ["MapToSeq", "SeqToMap", "PopKeys", "FilterKeys"]


class MapToSeq(AbstractTransform):
    def __init__(self, *keys, grad: bool = False, **kwargs):
        """
        Convert dict to sequence

        Parameters
        ----------
        keys: tuple
            keys which are mapped into sequence.
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            additional keyword arguments passed to superclass
        """
        super().__init__(grad=grad, **kwargs)
        if isinstance(keys[0], (list, tuple)):
            keys = keys[0]
        self.keys = keys

    def forward(self, **data) -> tuple:
        """
        Convert input

        Parameters
        ----------
        data: dict
            input dict

        Returns
        -------
        tuple
            mapped data
        """
        return tuple(data[_k] for _k in self.keys)


class SeqToMap(AbstractTransform):
    def __init__(self, *keys, grad: bool = False, **kwargs):
        """
        Convert sequence to dict

        Parameters
        ----------
        keys: tuple
            keys which are mapped into dict.
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            additional keyword arguments passed to superclass
        """
        super().__init__(grad=grad, **kwargs)
        if isinstance(keys[0], (list, tuple)):
            keys = keys[0]
        self.keys = keys

    def forward(self, *data, **kwargs) -> dict:
        """
        Convert input

        Parameters
        ----------
        data: tuple
            input tuple

        Returns
        -------
        dict
            mapped data
        """
        return {_key: data[_idx] for _idx, _key in enumerate(self.keys)}


class PopKeys(AbstractTransform):
    def __init__(self, keys: Union[Callable, Sequence], return_popped: bool = False):
        """
        Pops keys from a given data dict

        Parameters
        ----------
        keys : Callable or Sequence of Strings
            if callable it must return a boolean for each key indicating whether it should be popped from the dict.
            if sequence of strings, the strings shall be the keys to be popped
        return_popped : bool
            whether to also return the popped values (default: False)

        """
        super().__init__(grad=False)
        self.keys = keys
        self.return_popped = return_popped

    def forward(self, **data) -> Union[dict, Tuple[dict, dict]]:
        return pop_keys(data=data, keys=self.keys, return_popped=self.return_popped)


class FilterKeys(AbstractTransform):
    def __init__(self, keys: Union[Callable, Sequence], return_popped: bool = False):
        """
        Filters keys from a given data dict

        Parameters
        ----------
        keys : Callable or Sequence of Strings
            if callable it must return a boolean for each key indicating whether it should be retained in the dict.
            if sequence of strings, the strings shall be the keys to be retained
        return_popped : bool
            whether to also return the popped values (default: False)

        """
        super().__init__(grad=False)
        self.keys = keys
        self.return_popped = return_popped

    def forward(self, **data) -> Union[dict, Tuple[dict, dict]]:
        return filter_keys(data=data, keys=self.keys, return_popped=self.return_popped)
