from typing import Union, cast

import torch
from torch.distributions import Distribution as TorchDistribution

from rising.random.abstract import AbstractParameter, ConstantParameter
from rising.utils import check_scalar

__all__ = ["ContinuousParameter", "NormalParameter", "UniformParameter"]


class ContinuousParameter(AbstractParameter):
    """Class to perform parameter sampling from torch distributions"""

    def __init__(self, distribution: TorchDistribution):
        """
        Args:
            distribution : the distribution to sample from
        """
        super().__init__()
        self.dist = distribution

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples from the internal distribution

        Args:
            n_samples : the number of elements to sample

        Returns
            torch.Tensor: samples
        """

        # input should be a tuple or torch.Size tuple.
        return self.dist.sample((n_samples,))


class NormalParameter(ContinuousParameter):
    """
    Samples Parameters from a normal distribution.
    For details have a look at :class:`torch.distributions.Normal`
    if sigma is 0, return a ConstantParameter
    """

    def __init__(self, mu: Union[float, torch.Tensor], sigma: Union[float, torch.Tensor]):
        """
        Args:
            mu : the distributions mean
            sigma : the distributions standard deviation
        """
        assert check_scalar(mu) and check_scalar(sigma)
        if sigma == 0:
            dist = cast(torch.distributions.Distribution, ConstantParameter(constant=mu))
        else:
            dist = torch.distributions.Normal(mu, sigma)
        super().__init__(dist)


class UniformParameter(ContinuousParameter):
    """
    Samples Parameters from a uniform distribution.
    For details have a look at :class:`torch.distributions.Uniform`
    if `low`==`high` , return a ConstantParameter
    """

    def __init__(self, low: Union[float, int, torch.Tensor], high: Union[float, int, torch.Tensor]):
        """
        Args:
            low : the lower range (inclusive)
            high : the higher range (exclusive)
        """
        assert check_scalar(low) and check_scalar(high)

        if low == high:
            dist = cast(torch.distributions.Distribution, ConstantParameter(low))
        elif low < high:
            dist = torch.distributions.Uniform(low=low, high=high)
        else:
            raise ValueError("low must be smaller than high, given: low={} and high={}".format(low, high))
        super().__init__(dist)
