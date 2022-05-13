from functools import partial
from itertools import combinations
from random import choices as sample_with_replacement
from random import sample as sample_without_replacement
from typing import List, Optional, Sequence, TypeVar, Union

import torch

from rising.random.abstract import AbstractParameter

__all__ = ["DiscreteParameter", "DiscreteCombinationsParameter"]

T = TypeVar("T")


def combinations_all(data: Sequence) -> List:
    """
    Return all combinations of all length for given sequence

    Args:
        data: sequence to get combinations of

    Returns:
        List: all combinations
    """
    comb = []
    for r in range(1, len(data) + 1):
        comb.extend(combinations(data, r=r))
    return comb


class DiscreteParameter(AbstractParameter):
    """
    Samples parameters from a discrete population with or without
    replacement
    """

    def __init__(
        self,
        population: Sequence[T],
        replacement: bool = False,
        weights: Sequence[T] = None,
        cum_weights: Sequence[T] = None,
    ):
        """
        Args:
            population : the parameter population to sample from
            replacement : whether to sample with replacement
            weights : relative sampling weights
            cum_weights : cumulative sampling weights
        """
        super().__init__()
        if replacement:
            sample_fn = partial(sample_with_replacement, weights=weights, cum_weights=cum_weights)
        elif weights is not None or cum_weights is not None:
            raise ValueError("weights and cum_weights should only be specified if " "replacement is set to True!")
        else:
            sample_fn = sample_without_replacement

        self.sample_fn = sample_fn
        self.population = population

    def sample(self, n_samples: int) -> List[T]:
        """
        Samples from the discrete internal population

        Args:
            n_samples : the number of elements to sample

        Returns:
            list: the sampled values

        """
        return self.sample_fn(population=self.population, k=n_samples)

    def forward(
        self,
        size: Optional[Union[Sequence, torch.Size]] = None,
        device: Union[torch.device, str] = None,
        dtype: Union[torch.dtype, str] = None,
        tensor_like: torch.Tensor = None,
    ) -> Union[None, list, torch.Tensor]:
        """
        Forward function (will also be called if the module is called).
        Calculates the number of samples from the given shape, performs the
        sampling and converts it back to the correct shape.

        Args:
            size: the size of the sampled values. If None, it samples one value
                without reshaping
            device : the device the result value should be set to, if it is a tensor
            dtype : the dtype, the result value should be casted to, if it is a tensor
            tensor_like: the tensor, having the correct dtype and device.
                The result will be pushed onto this device and casted to this
                dtype if this is specified.

        Returns:
            list or torch.Tensor: the sampled values

        Notes:
            if the parameter ``tensor_like`` is given,
            it overwrites the parameters ``dtype`` and ``device``
        """
        if size is None:
            size = (1,)
        n_samples = self._get_n_samples(size)
        samples = self.sample(n_samples)

        if any([s is None for s in samples]):
            return None

        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples).flatten()

        if isinstance(samples, torch.Tensor):
            if tensor_like is not None:
                samples = samples.to(tensor_like)
            else:
                samples = samples.to(device=device, dtype=dtype)
        return samples


class DiscreteCombinationsParameter(DiscreteParameter):
    """
    Sample parameters from an extended population which consists of all
    possible combinations of the given population
    """

    def __init__(self, population: Sequence[T], replacement: bool = False):
        """
        Args:
            population : population to build combination of
            replacement : whether to sample with replacement
        """
        population = combinations_all(population)
        super().__init__(population=population, replacement=replacement)


if __name__ == "__main__":
    print(DiscreteParameter(population=[1, 2, 3, 4, 5], replacement=True))
