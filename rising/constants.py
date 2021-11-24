from enum import Enum


class FInterpolation(Enum):
    bilinear = "bilinear"
    nearest = "nearest"
    trilinear = "trilinear"


class Interpolation(Enum):
    pass


class AffineInterpolation(Enum):
    linear = "bilinear"
    nearest = "nearest"
