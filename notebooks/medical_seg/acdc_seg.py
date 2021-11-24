from tqdm import tqdm
from unet_3d import UNet

from notebooks.medical_seg.dataset import ACDCDataset
from rising import transforms
from rising.loading import DataLoader, default_transform_call
from rising.random import UniformParameter

tra_dataset = ACDCDataset(root="/home/jizong/Workspace/rising/notebooks/medical_seg/data", train=True)
val_dataset = ACDCDataset(root="/home/jizong/Workspace/rising/notebooks/medical_seg/data", train=False)

seq_tra_augment = transforms.Compose(
    transforms.NormPercentile(keys=("image",), min=0.01, max=0.99),
    transforms.Pad(keys=("image", "label"), pad_value=0.0, pad_size=(16, 256, 256)),
    transforms.RandomCrop(
        keys=("image", "label"),
        size=(
            16,
            256,
            256,
        ),
    ),
)

seq_val_augment = transforms.Compose(
    transforms.NormPercentile(keys=("image", "label"), min=0.01, max=0.99),
    transforms.Pad(keys=("image", "label"), pad_value=0.0, pad_size=(256, 256, 15)),
    # transforms
    # transforms.CenterCrop(size=(192, 168, 10), keys=("image", "label")),
    transform_call=default_transform_call,
)

batch_augment = transforms.Compose(
    # transforms.ToDtype(keys=("image", "label"), dtype=torch.half),
    # transforms.ToDevice(keys=("image", "label"), device=torch.device("cuda")),
    transforms.GammaCorrection(gamma=UniformParameter(0.8, 2), keys=("image",)),
    transforms.BaseAffine(
        scale=(UniformParameter(0.9, 1.2), UniformParameter(0.9, 1.2), 1),
        rotation=(0, UniformParameter(-50, 50), UniformParameter(-50, 50)),
        translation=0,
        degree=True,
        p=0.8,
        interpolation_mode=("bilinear", "nearest"),
        keys=("image", "label"),
    ),
    transforms.RandomCrop(size=(16, 192, 168), dist=0, keys=("image", "label")),
    transforms.ElasticDistortion(
        std=20,
        alpha=0.1,
        dim=3,
        keys=("image", "label"),
        interpolation_mode=("bilinear", "nearest"),
    ),
    transform_call=default_transform_call,
)

tra_loader = DataLoader(
    tra_dataset,
    batch_size=8,
    shuffle=True,
    sample_transforms=seq_tra_augment,
    gpu_transforms=batch_augment,
    pseudo_batch_dim=True,
    num_workers=4,
)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, sample_transforms=seq_val_augment,
#                         gpu_transforms=batch_augment, pseudo_batch_dim=True, num_workers=4)
model = UNet(in_dim=1, out_dim=4, num_filters=4)
model.cuda()

for data in tqdm(tra_loader):
    image, label = data["image"], data["label"]
    # image, label = image.to(torch.float), label.to(torch.float)
    # prediction = model(image)
    #
    # breakpoint()
