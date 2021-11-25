from tqdm import tqdm
from unet_3d import UNet

from notebooks.medical_seg.dataset import ACDCDataset
from rising import transforms
from rising.loading import DataLoader, default_transform_call
from rising.random import UniformParameter

tra_dataset = ACDCDataset(root="/home/jizong/Workspace/rising/notebooks/medical_seg/data", train=True)
val_dataset = ACDCDataset(root="/home/jizong/Workspace/rising/notebooks/medical_seg/data", train=False)

seq_tra_augment = transforms.Compose(
    transforms.NormPercentile(keys=("image",), min=0.02, max=0.98),
    transforms.PadRandomCrop(size=(15, 224, 224), pad_size=2, pad_value=(0, 0), keys=("image", "label")),
)

batch_augment = transforms.Compose(
    # transforms.ToDtype(keys=("image", "label"), dtype=torch.half),
    # transforms.ToDevice(keys=("image", "label"), device=torch.device("cuda")),
    transforms.GammaCorrection(gamma=UniformParameter(0.8, 2), keys=("image",)),
    transforms.BaseAffine(
        scale=(1, UniformParameter(0.8, 1.2), UniformParameter(0.9, 1.2)),
        rotation=(0, UniformParameter(-10, 10), UniformParameter(-10, 50)),
        translation=0,
        degree=True,
        p=0.5,
        interpolation_mode=("bilinear", "nearest"),
        keys=("image", "label"),
    ),
    transforms.RandomCrop(size=(10, 192, 168), keys=("image", "label")),
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
    batch_size=6,
    shuffle=True,
    sample_transforms=seq_tra_augment,
    gpu_transforms=batch_augment,
    pseudo_batch_dim=True,
    num_workers=0,
)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, sample_transforms=seq_val_augment,
#                         gpu_transforms=batch_augment, pseudo_batch_dim=True, num_workers=4)
model = UNet(in_dim=1, out_dim=4, num_filters=4)
model.cuda()

for data in tqdm(tra_loader):
    image, label = data["image"], data["label"]
    from tests.realtime_viewer import multi_slice_viewer_debug

    multi_slice_viewer_debug([*image.squeeze()], *label.squeeze(), block=True, no_contour=True)
    # image, label = image.to(torch.float), label.to(torch.float)
    # prediction = model(image)
    #
    # breakpoint()
