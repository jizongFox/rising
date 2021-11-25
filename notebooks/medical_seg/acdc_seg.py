from tqdm import tqdm

from notebooks.medical_seg.dataset import ACDCDataset, InfiniteRandomSampler
from rising import transforms
from rising.constants import FInterpolation
from rising.loading import DataLoader, default_transform_call
from rising.random import UniformParameter

tra_dataset = ACDCDataset(root="/home/jizong/Workspace/rising/notebooks/medical_seg/data", train=True)

seq_tra_augment = transforms.Compose(
    transforms.NormPercentile(keys=("image",), min=0.02, max=0.98),
    transforms.ResizeNative(
        size=(10, 224, 224),
        mode=(FInterpolation.trilinear, FInterpolation.nearest),
        preserve_range=True,
        keys=("image", "label"),
    ),
    # transforms.PadRandomCrop(size=(10, 224, 224), pad_size=2, pad_value=(0, 0), keys=("image", "label")),
)

batch_augment = transforms.Compose(
    # transforms.ToDtype(keys=("image", "label"), dtype=torch.half),
    # transforms.ToDevice(keys=("image", "label"), device=torch.device("cuda")),
    transforms.GammaCorrection(gamma=UniformParameter(0.8, 2), keys=("image",)),
    transforms.RicianNoiseTransform(keys=("image",), std=0.05, keep_range=False),
    transforms.BaseAffine(
        scale=(1, UniformParameter(0.5, 2), UniformParameter(0.9, 1.2)),
        rotation=(0, UniformParameter(-10, 10), UniformParameter(-10, 10)),
        degree=True,
        p=1,
        per_sample=True,
        interpolation_mode=("bilinear", "nearest"),
        keys=("image", "label"),
    ),
    transforms.RandomCrop(size=(8, 192, 168), keys=("image", "label")),
    transforms.ElasticDistortion(
        std=20,
        alpha=0.2,
        dim=3,
        keys=("image", "label"),
        interpolation_mode=("bilinear", "nearest"),
    ),
    transform_call=default_transform_call,
)

tra_loader = DataLoader(
    tra_dataset,
    sampler=InfiniteRandomSampler(tra_dataset, shuffle=False),
    batch_size=6,
    sample_transforms=seq_tra_augment,
    gpu_transforms=batch_augment,
    pseudo_batch_dim=True,
    num_workers=16,
)

for data in tqdm(tra_loader):
    image, label = data["image"], data["label"]
    # from tests.realtime_viewer import multi_slice_viewer_debug
    #
    # for img, lab in zip(image, label):
    #     multi_slice_viewer_debug(img.squeeze(), lab.squeeze(), block=False, no_contour=True)
    # plt.show()
