from typing import Any, Union

import numpy as np
import torch


def check_scalar(x: Union[Any, float, int]) -> bool:
    """
    Provide interface to check for scalars

    Args:
        x: object to check for scalar

    Returns:
        bool" True if input is scalar
    """
    if isinstance(x, (int, float)):
        return True
    elif isinstance(x, torch.Tensor) and x.numel() == 1:
        return True
    elif isinstance(x, np.ndarray) and x.size == 1:
        return True
    else:
        return False


def to_scalar(x: Union[Any, float, int]) -> Union[float, int]:
    """
    Provide interface to convert to scalar  if possible

    Args:
        x: object to convert to scalar
    """
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, torch.Tensor) and x.numel() == 1:
        return x.item()
    elif isinstance(x, np.ndarray) and x.size == 1:
        return x.item()
    else:
        raise ValueError("Cannot convert to scalar")
