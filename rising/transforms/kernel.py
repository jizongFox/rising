import math
from typing import Callable, Sequence

import torch
from torch.nn import functional as F

from rising.utils import check_scalar
from rising.utils.mise import ntuple

from .abstract import AbstractTransform, item_or_seq

__all__ = ["KernelTransform", "GaussianSmoothing"]


class KernelTransform(AbstractTransform):
    """
    Baseclass for kernel based transformations (kernel is applied to
    each channel individually)
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: item_or_seq[int],
        dim: int = 2,
        stride: item_or_seq[int] = 1,
        padding: item_or_seq[int] = 0,
        padding_mode: str = "zero",
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            in_channels: number of input channels
            kernel_size: size of kernel
            dim: number of spatial dimensions
            stride: stride of convolution
            padding: padding size for input
            padding_mode: padding mode for input. Supports all modes
                from :func:`torch.functional.pad` except ``circular``
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to superclass

        See Also:
            :func:`torch.functional.pad`
        """
        super().__init__(grad=grad, **kwargs)
        self.keys = keys
        self._tuple_generator = ntuple(len(self.keys))

        self.in_channels = in_channels

        if check_scalar(kernel_size):
            kernel_size = [kernel_size] * dim
        self.kernel_size = kernel_size

        if check_scalar(stride):
            stride = [stride] * dim
        self.stride = stride

        if check_scalar(padding):
            padding = [padding] * dim * 2
        self.padding = padding

        self.padding_mode = self._tuple_generator(padding_mode)

        kernel = self.create_kernel()
        self.register_buffer("weight", kernel)
        self.groups = in_channels
        self.conv = self.get_conv(dim)

    @staticmethod
    def get_conv(dim) -> Callable:
        """
        Select convolution with regard to dimension

        Args:
            dim: spatial dimension of data

        Returns:
            Callable: the suitable convolutional function
        """
        if dim == 1:
            return F.conv1d
        elif dim == 2:
            return F.conv2d
        elif dim == 3:
            return F.conv3d
        else:
            raise TypeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def create_kernel(self) -> torch.Tensor:
        """
        Create kernel for convolution
        """
        raise NotImplementedError

    def forward(self, **data) -> dict:
        """
        Apply kernel to selected keys

        Args:
            data: input data

        Returns:
            dict: dict with transformed data
        """
        dtype, device = data[self.keys[0]].dtype, data[self.keys[0]].device
        self.to(dtype)
        for key, padding_mode in zip(self.keys, self.padding_mode):
            inp_pad = F.pad(data[key], self.padding, mode=padding_mode)

            data[key] = self.conv(inp_pad, weight=self.weight, groups=self.groups, stride=self.stride)
        return data


class GaussianSmoothing(KernelTransform):
    """
    Perform Gaussian Smoothing.
    Filtering is performed separately for each channel in the input using a
    depthwise convolution.
    This code is adapted from:
    'https://discuss.pytorch.org/t/is-there-anyway-to-do-'
    'gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10'
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: item_or_seq[int],
        std: item_or_seq[float],
        dim: int = 2,
        stride: item_or_seq[int] = 1,
        padding: item_or_seq[int] = 0,
        padding_mode: item_or_seq[str] = "constant",
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            in_channels: number of input channels
            kernel_size: size of kernel
            std: standard deviation of gaussian
            dim: number of spatial dimensions
            stride: stride of convolution
            padding: padding size for input
            padding_mode: padding mode for input. Supports all modes from
                :func:`torch.functional.pad` except ``circular``
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to superclass

        See Also:
            :func:`torch.functional.pad`
        """
        if check_scalar(std):
            std = [std] * dim
        self.std = std
        super().__init__(
            in_channels=in_channels,
            kernel_size=kernel_size,
            dim=dim,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            keys=keys,
            grad=grad,
            **kwargs
        )

    def create_kernel(self) -> torch.Tensor:
        """
        Create gaussian blur kernel
        """
        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in self.kernel_size])

        for size, std, mgrid in zip(self.kernel_size, self.std, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / kernel.sum()

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(self.in_channels, *[1] * (kernel.dim() - 1))
        kernel.requires_grad = False
        return kernel.contiguous()
