import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class StereoSRDataset(Dataset):
    def __init__(self, data_root: str, scale: int = 4, patch_size: int = 64,
                 augment: bool = True, is_train: bool = True):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment and is_train
        self.is_train = is_train

        hr_left_dir = os.path.join(data_root, "hr", "left")
        hr_right_dir = os.path.join(data_root, "hr", "right")

        if os.path.exists(hr_left_dir):
            left_paths = sorted(glob.glob(os.path.join(hr_left_dir, "*")))
            right_paths = sorted(glob.glob(os.path.join(hr_right_dir, "*")))
        else:
            left_paths = sorted(glob.glob(os.path.join(data_root, "left", "*")))
            right_paths = sorted(glob.glob(os.path.join(data_root, "right", "*")))
        if not left_paths:
            all_imgs = sorted(glob.glob(os.path.join(data_root, "*")))
            mid = len(all_imgs) // 2
            left_paths = all_imgs[:mid]
            right_paths = all_imgs[mid:]

        assert len(left_paths) == len(right_paths), \
            f"Mismatched pair counts: {len(left_paths)} left vs {len(right_paths)} right"
        assert len(left_paths) > 0, f"No images found in {data_root}"

        self.pairs = list(zip(left_paths, right_paths))

        lr_dir = os.path.join(data_root, f"lr_x{scale}", "left")
        self.has_lr = os.path.exists(lr_dir)
        if self.has_lr:
            self.lr_left_paths = sorted(glob.glob(os.path.join(
                data_root, f"lr_x{scale}", "left", "*")))
            self.lr_right_paths = sorted(glob.glob(os.path.join(
                data_root, f"lr_x{scale}", "right", "*")))

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img)

    def _random_crop(self, *images, patch_size: int):
        _, h, w = images[0].shape
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            images = [TF.pad(img, [0, 0, pad_w, pad_h], padding_mode="reflect")
                      for img in images]
            _, h, w = images[0].shape

        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        return [img[:, top:top+patch_size, left:left+patch_size] for img in images]

    def _augment(self, hr_l, hr_r, lr_l, lr_r):
        if random.random() > 0.5:
            hr_l = TF.hflip(hr_l)
            hr_r = TF.hflip(hr_r)
            lr_l = TF.hflip(lr_l)
            lr_r = TF.hflip(lr_r)
            hr_l, hr_r = hr_r, hr_l
            lr_l, lr_r = lr_r, lr_l
        if random.random() > 0.5:
            hr_l = TF.vflip(hr_l)
            hr_r = TF.vflip(hr_r)
            lr_l = TF.vflip(lr_l)
            lr_r = TF.vflip(lr_r)
        if random.random() > 0.5:
            angle = random.choice([90, -90])
            hr_l = TF.rotate(hr_l, angle)
            hr_r = TF.rotate(hr_r, angle)
            lr_l = TF.rotate(lr_l, angle)
            lr_r = TF.rotate(lr_r, angle)
        if random.random() > 0.5:
            perm = torch.randperm(3)
            hr_l = hr_l[perm]
            hr_r = hr_r[perm]
            lr_l = lr_l[perm]
            lr_r = lr_r[perm]

        return hr_l, hr_r, lr_l, lr_r

    def __getitem__(self, idx):
        left_path, right_path = self.pairs[idx]
        hr_left = self._load_image(left_path)
        hr_right = self._load_image(right_path)
        if self.has_lr:
            lr_left = self._load_image(self.lr_left_paths[idx])
            lr_right = self._load_image(self.lr_right_paths[idx])
        else:
            _, h, w = hr_left.shape
            lr_h, lr_w = h // self.scale, w // self.scale
            lr_left = TF.resize(hr_left, [lr_h, lr_w],
                                interpolation=TF.InterpolationMode.BICUBIC,
                                antialias=True)
            lr_right = TF.resize(hr_right, [lr_h, lr_w],
                                 interpolation=TF.InterpolationMode.BICUBIC,
                                 antialias=True)

        if self.is_train:
            hr_patch_size = self.patch_size * self.scale
            hr_left, hr_right = self._random_crop(hr_left, hr_right,
                                                   patch_size=hr_patch_size)
            lr_left = TF.resize(hr_left, [self.patch_size, self.patch_size],
                                interpolation=TF.InterpolationMode.BICUBIC,
                                antialias=True)
            lr_right = TF.resize(hr_right, [self.patch_size, self.patch_size],
                                 interpolation=TF.InterpolationMode.BICUBIC,
                                 antialias=True)
            if self.augment:
                hr_left, hr_right, lr_left, lr_right = self._augment(
                    hr_left, hr_right, lr_left, lr_right)
        lr_left = lr_left.clamp(0, 1)
        lr_right = lr_right.clamp(0, 1)
        hr_left = hr_left.clamp(0, 1)
        hr_right = hr_right.clamp(0, 1)

        return {
            "lr_left": lr_left,
            "lr_right": lr_right,
            "hr_left": hr_left,
            "hr_right": hr_right,
            "filename": os.path.basename(left_path),
        }


class StereoMixup:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, batch):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        B = batch["lr_left"].size(0)
        idx = torch.randperm(B)

        for key in ["lr_left", "lr_right", "hr_left", "hr_right"]:
            batch[key] = lam * batch[key] + (1 - lam) * batch[key][idx]

        return batch


def build_dataset(args, split="train"):
    if split == "train":
        return StereoSRDataset(
            args.train_data, scale=args.scale, patch_size=args.patch_size,
            augment=args.augment, is_train=True
        )
    elif split == "val":
        return StereoSRDataset(
            args.val_data, scale=args.scale, patch_size=0,
            augment=False, is_train=False
        )
    elif split == "test":
        return StereoSRDataset(
            args.test_data, scale=args.scale, patch_size=0,
            augment=False, is_train=False
        )
