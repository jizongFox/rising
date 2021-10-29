from typing import Sequence, Union

from torch import Tensor
from torch.nn import functional as F

from rising.utils import check_scalar
from rising.utils.mise import ntuple


def pad(data: Tensor, pad_size: Union[int, Sequence[int]], grid_pad=False, mode="constant", value: float = 0.0):
    n_dim = data.dim() - 2
    # padding parameters
    if check_scalar(pad_size):
        pad_size = ntuple(n_dim * 2)(pad_size)
    elif isinstance(pad_size, Sequence):
        pad_size = tuple(pad_size)
        if not (len(pad_size) == 0 or len(pad_size) != n_dim or len(pad_size) != n_dim * 2):
            raise TypeError(pad_size)
        if len(pad_size) == n_dim:
            pad_size = tuple((z for double in zip(pad_size, pad_size) for z in double))
        elif (len(pad_size)) == n_dim * 2:
            pad_size = pad_size
        else:
            raise RuntimeError(pad_size)
    assert isinstance(pad_size, tuple) and len(pad_size) == 2 * n_dim

    # todo: understand the grid_pad for affine distribution
    if grid_pad is False:
        return F.pad(data, pad=pad_size, mode=mode, value=value)
    else:
        raise NotImplementedError(grid_pad)


#
# if __name__ == '__main__':
#     from PIL import Image
#     import requests
#     from io import BytesIO
#     import numpy as np
#     import torch
#     from torch.nn import functional as F
#
#     url = "https://miro.medium.com/max/1400/0*1F2u74JQYI8sUuYg"
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content))
#     np_img = np.asarray(img).astype(np.float32)
#     t_img = torch.from_numpy(np_img).permute(2, 0, 1)
#     t_img = t_img[None, ...]
#     t_img_pad = F.pad(t_img, pad=(200, 200, 400, 400), mode="replicate")
