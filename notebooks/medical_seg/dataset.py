import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data.dataset import Dataset, T_co


class ACDCDataset(Dataset):
    def __init__(self, *, root: str, train: bool = True) -> None:
        self.root = root
        self.train = train
        self._root = Path(root, "train" if train else "val")

        self.images = filter(
            self.image_filter, [str(x.relative_to(self._root)) for x in Path(self._root).rglob("*.nii.gz")]
        )
        self.images = sorted(self.images)

    def __getitem__(self, index) -> T_co:
        image_path = str(self._root / self.images[index])
        gt_path = image_path.replace(".nii.gz", "_gt.nii.gz")
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(np.float32, copy=False)[None, ...]
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.float32, copy=False)[None, ...]
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(gt)}

    def __len__(self):
        return len(self.images)

    @staticmethod
    def image_filter(path: str):
        _match = re.compile(r"patient\d+_frame\d+.nii.gz").search(str(path))
        if _match is None:
            return False
        return True
