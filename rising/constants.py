from enum import Enum


class Interpolation(Enum):
    pass


class AffineInterpolation(Enum):
    linear = "bilinear"
    nearest = "nearest"
