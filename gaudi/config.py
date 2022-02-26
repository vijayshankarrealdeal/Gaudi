import torch
from torch import nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from habana_frameworks.torch.utils.library_loader import load_habana_module

###
load_habana_module()
###
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "./gaudi/checkpoint/gen.pth.tar"
CHECKPOINT_DISC = "./gaudi/checkpoint/disc.pth.tar"
DEVICE = torch.device("hpu")
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
BATCH_SIZE = 16
NUM_WORKERS = 2
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)