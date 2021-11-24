from abc import abstractmethod
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from rising.random.utils import fix_random_seed_ctx
from rising.transforms.abstract import TYPE_item_seq, _AbstractTransform
from rising.transforms.functional import center_crop, random_crop
from rising.transforms.kernel import GaussianSmoothing
from rising.utils.affine import get_batched_eye, matrix_to_homogeneous
from rising.utils.mise import ntuple

__all__ = [
    "GridTransform",
    "StackedGridTransform",
    "CenterCropGrid",
    "RandomCropGrid",
    "ElasticDistortion",
    "RadialDistortion",
]


class GridTransform(_AbstractTransform):
    """
    Abstract class for grid transformation.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("data",),
        interpolation_mode: TYPE_item_seq[str] = "bilinear",
        padding_mode: TYPE_item_seq[str] = "zeros",
        align_corners: TYPE_item_seq[bool] = False,
        grad: bool = False,
        **kwargs,
    ):
        super().__init__(grad=grad)
        self.keys = keys
        self._tuple_generator = ntuple(len(self.keys))
        self.interpolation_mode: Sequence[str] = self._tuple_generator(interpolation_mode)
        self.padding_mode: Sequence[str] = self._tuple_generator(padding_mode)
        self.align_corners: Sequence[bool] = self._tuple_generator(align_corners)
        self.kwargs = kwargs

        self.grid: Optional[Dict[str, Tensor]] = None

    def forward(self, **data) -> dict:
        device, dtype = data[self.keys[0]].device, data[self.keys[0]].dtype

        if self.grid is None:
            self.grid = self.create_grid(data, device=device, dtype=dtype)

        self.grid = self.augment_grid(self.grid, device=device, dtype=dtype)

        for key, interpol, padding_mode, align_corners in zip(
            self.keys, self.interpolation_mode, self.padding_mode, self.align_corners
        ):
            data[key] = F.grid_sample(
                data[key], self.grid[key], mode=interpol, padding_mode=padding_mode, align_corners=align_corners
            )
        self.grid = None
        return data

    @abstractmethod
    def augment_grid(self, grid: Dict[str, Tensor], *, device, dtype) -> Dict[str, Tensor]:
        """
        this functions modifies the grid
        """
        raise NotImplementedError

    def create_grid(
        self, data: Dict[str, Tensor], matrix: Tensor = None, *, device: torch.device, dtype: torch.dtype
    ) -> Dict[str, Tensor]:
        grid = {}
        for key, align_corners in zip(self.keys, self.align_corners):
            cur_data = data[key]
            batch_size = cur_data.shape[0]
            ndim = cur_data.dim() - 2
            if matrix is None:
                matrix = get_batched_eye(batchsize=batch_size, ndim=ndim, device=device, dtype=dtype)
                matrix = matrix_to_homogeneous(matrix)[:, :-1]

            grid[key] = F.affine_grid(matrix, size=list(cur_data.shape), align_corners=align_corners)
        return grid

    def __add__(self, other):
        if not isinstance(other, GridTransform):
            raise ValueError("Concatenation is only supported for grid transforms.")
        return StackedGridTransform(self, other)

    def __radd__(self, other):
        if not isinstance(other, GridTransform):
            raise ValueError("Concatenation is only supported for grid transforms.")
        return StackedGridTransform(other, self)


class StackedGridTransform(GridTransform):
    def __init__(self, *transforms: Union[GridTransform, Sequence[GridTransform]]):
        super().__init__(keys=None, interpolation_mode=None, padding_mode=None, align_corners=None)
        if isinstance(transforms, (tuple, list)):
            if isinstance(transforms[0], (tuple, list)):
                transforms = transforms[0]
        self.transforms = transforms

    def create_grid(
        self, data: Dict[str, Tensor], matrix: Tensor = None, *, device: torch.device, dtype: torch.dtype
    ) -> Dict[str, Tensor]:
        return self.transforms[0].create_grid(data=data, matrix=matrix, device=device, dtype=dtype)

    def augment_grid(self, grid: Dict[str, Tensor], *, device, dtype) -> Dict[str, Tensor]:
        for transform in self.transforms:
            grid = transform.augment_grid(grid, device=device, dtype=dtype)
        return grid


class CenterCropGrid(GridTransform):
    def __init__(
        self,
        *,
        size: Union[int, Sequence[int]],
        keys: Sequence[str] = ("data",),
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        grad: bool = False,
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            grad=grad,
            **kwargs,
        )
        self.size = size

    def augment_grid(self, grid: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        return {key: center_crop(cur_grid, size=self.size, grid_crop=True) for key, cur_grid in grid.items()}


class RandomCropGrid(GridTransform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        dist: Union[int, Sequence[int]] = 0,
        keys: Sequence[str] = ("data",),
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        grad: bool = False,
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            grad=grad,
            **kwargs,
        )
        self.size = size
        self.dist = dist

    def augment_grid(self, grid: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        return {key: random_crop(item, size=self.size, dist=self.dist, grid_crop=True) for key, item in grid.items()}


class ElasticDistortion(GridTransform):
    """
    ElasticDistortion transformation
    """

    def __init__(
        self,
        std: Union[float, Sequence[float]],
        alpha: float,
        dim: int = 2,
        keys: Sequence[str] = ("data",),
        interpolation_mode: TYPE_item_seq[str] = "bilinear",
        padding_mode: TYPE_item_seq[str] = "zeros",
        align_corners: TYPE_item_seq[bool] = False,
        grad: bool = False,
        per_sample: bool = True,
        **kwargs,
    ):
        """
        std: std of the gaussian smooth
        """
        super().__init__(
            keys=keys,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            grad=grad,
            **kwargs,
        )
        self.std = std
        self.alpha = alpha
        self.per_sample = per_sample
        self.gaussian = GaussianSmoothing(in_channels=1, kernel_size=7, std=self.std, dim=dim, stride=1, padding=3)

    def augment_grid(self, grid: Dict[Tuple, Tensor], *, device, dtype) -> Dict[Tuple, Tensor]:
        seed = torch.randint(0, int(1e6), size=(1,))

        def get_perturb_grid(batch_size: int = 1) -> Tensor:
            random_offsets = torch.rand(batch_size, 1, *grid[key].shape[1:-1], device=device, dtype=dtype) * 2 - 1
            return self.gaussian(data=random_offsets)["data"] * self.alpha

        for key in grid.keys():
            cur_data = grid[key]
            batch_size = cur_data.shape[0]
            with fix_random_seed_ctx(seed):
                if self.per_sample:
                    random_offsets = get_perturb_grid(batch_size)[:, 0, ..., None]
                else:
                    random_offsets = get_perturb_grid(1)[:, 0, ..., None]
            grid[key] += random_offsets
        return grid


class RadialDistortion(GridTransform):
    def __init__(
        self,
        scale: Tuple[float, float, float],
        keys: Sequence[str] = ("data",),
        interpolation_mode: TYPE_item_seq[str] = "bilinear",
        padding_mode: TYPE_item_seq[str] = "zeros",
        align_corners: TYPE_item_seq[bool] = False,
        grad: bool = False,
        **kwargs,
    ):
        super().__init__(
            keys=keys,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            grad=grad,
            **kwargs,
        )
        self.scale = scale

    def augment_grid(self, grid: Dict[Tuple, Tensor], **kwargs) -> Dict[Tuple, Tensor]:
        new_grid = {key: radial_distortion_grid(cur_grid, scale=self.scale) for key, cur_grid in grid.items()}
        return new_grid


def radial_distortion_grid(grid: Tensor, scale: Tuple[float, float, float]) -> Tensor:
    dist = torch.norm(grid, p=2, dim=-1, keepdim=True)
    dist = dist / dist.max()
    distortion = (scale[0] * dist.pow(3) + scale[1] * dist.pow(2) + scale[2] * dist) / 3
    return grid * (1 - distortion)
