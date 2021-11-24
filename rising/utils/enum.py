from enum import Enum


class FInterpolation(Enum):
    bilinear = "bilinear"
    nearest = "nearest"
    trilinear = "trilinear"
