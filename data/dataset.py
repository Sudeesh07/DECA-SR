"""
Dataset loader for Stereo SR dataset.

Supports two layouts:

1. Training patches (pre-cropped):
   patches_x{scale}/{scene_id}/hr0.png, hr1.png, lr0.png, lr1.png

2. Test sets:
   test/{DatasetName}/hr/{scene_id}/hr0.png, hr1.png
   test/{DatasetName}/lr_x{scale}/{scene_id}/lr0.png, lr1.png
"""

import glob
import os
import random
from typing import Dict, List, Optional, Tuple
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def _is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class StereoSRDataset(Dataset):
    """
    Stereo SR dataset supporting:

    Training layout (pre-extracted patches):
      root/patches_x{scale}/000001/hr0.png, hr1.png, lr0.png, lr1.png
      root/patches_x{scale}/000002/hr0.png, hr1.png, lr0.png, lr1.png
      ...

    Test layout:
      root/test/{source}/hr/{scene}/hr0.png, hr1.png
      root/test/{source}/lr_x{scale}/{scene}/lr0.png, lr1.png

    Args:
        data_root: Path to the StereoSR root directory
        scale: Super-resolution scale factor (2 or 4)
        patch_size: Crop size for training (in LR space). Ignored for test.
        augment: Whether to apply data augmentation (training only)
        is_train: If True, loads from patches_x{scale}. If False, loads from test.
        test_source: Which test set to use, e.g. "Flickr1024", "KITTI2012", etc.
                     Required when is_train=False.
    """

    def __init__(
        self,
        data_root: str,
        scale: int = 4,
        patch_size: int = 64,
        augment: bool = True,
        is_train: bool = True,
        test_source: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment and is_train
        self.is_train = is_train
        self.test_source = test_source

        if is_train:
            self.samples = self._discover_train_patches(data_root, scale)
        else:
            if test_source is None:
                raise ValueError("test_source must be specified for test/val split")
            self.samples = self._discover_test_pairs(data_root, scale, test_source)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No stereo pairs found. root={data_root}, scale={scale}, "
                f"is_train={is_train}, test_source={test_source}"
            )

    @staticmethod
    def _discover_train_patches(data_root: str, scale: int) -> List[Dict[str, str]]:
        """
        Finds all training patch folders.

        Expected: root/patches_x{scale}/{scene_id}/hr0.png, hr1.png, lr0.png, lr1.png
        """
        patches_dir = os.path.join(data_root, f"patches_x{scale}")
        if not os.path.isdir(patches_dir):
            raise RuntimeError(f"Training patches directory not found: {patches_dir}")

        scene_dirs = sorted(
            d for d in glob.glob(os.path.join(patches_dir, "*"))
            if os.path.isdir(d)
        )

        samples = []
        for scene in scene_dirs:
            # Look for hr0/hr1/lr0/lr1 with any common image extension
            hr0 = lr0 = hr1 = lr1 = None
            for f in os.listdir(scene):
                name = os.path.splitext(f)[0].lower()
                path = os.path.join(scene, f)
                if not _is_image(path):
                    continue
                if name == "hr0":
                    hr0 = path
                elif name == "hr1":
                    hr1 = path
                elif name == "lr0":
                    lr0 = path
                elif name == "lr1":
                    lr1 = path

            if hr0 and hr1 and lr0 and lr1:
                samples.append({
                    "hr_left": hr0,
                    "hr_right": hr1,
                    "lr_left": lr0,
                    "lr_right": lr1,
                    "scene": os.path.basename(scene),
                })
            else:
                missing = []
                if not hr0: missing.append("hr0")
                if not hr1: missing.append("hr1")
                if not lr0: missing.append("lr0")
                if not lr1: missing.append("lr1")
                # Silently skip incomplete folders but warn if many are missing
                pass

        return samples

    @staticmethod
    def _discover_test_pairs(
        data_root: str, scale: int, source: str
    ) -> List[Dict[str, str]]:
        """
        Finds all test pairs for a given source dataset.

        Expected:
          root/test/{source}/hr/{scene}/hr0.png, hr1.png
          root/test/{source}/lr_x{scale}/{scene}/lr0.png, lr1.png
        """
        hr_root = os.path.join(data_root, "test", source, "hr")
        lr_root = os.path.join(data_root, "test", source, f"lr_x{scale}")

        if not os.path.isdir(hr_root):
            raise RuntimeError(f"Test HR directory not found: {hr_root}")
        if not os.path.isdir(lr_root):
            raise RuntimeError(f"Test LR directory not found: {lr_root}")

        # Find scene folders inside hr/
        scene_dirs = sorted(
            d for d in glob.glob(os.path.join(hr_root, "*"))
            if os.path.isdir(d)
        )

        samples = []
        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)

            # Find HR images (hr0, hr1) with any extension
            hr0 = hr1 = None
            for f in os.listdir(scene_dir):
                name = os.path.splitext(f)[0].lower()
                path = os.path.join(scene_dir, f)
                if not _is_image(path):
                    continue
                if name == "hr0":
                    hr0 = path
                elif name == "hr1":
                    hr1 = path

            # Find LR images (lr0, lr1)
            lr_scene_dir = os.path.join(lr_root, scene_name)
            lr0 = lr1 = None
            if os.path.isdir(lr_scene_dir):
                for f in os.listdir(lr_scene_dir):
                    name = os.path.splitext(f)[0].lower()
                    path = os.path.join(lr_scene_dir, f)
                    if not _is_image(path):
                        continue
                    if name == "lr0":
                        lr0 = path
                    elif name == "lr1":
                        lr1 = path

            if hr0 and hr1:
                sample = {
                    "hr_left": hr0,
                    "hr_right": hr1,
                    "scene": scene_name,
                }
                if lr0 and lr1:
                    sample["lr_left"] = lr0
                    sample["lr_right"] = lr1
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_image(path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img)

    @staticmethod
    def _crop(img: torch.Tensor, top: int, left: int, h: int, w: int) -> torch.Tensor:
        return img[:, top : top + h, left : left + w]

    @staticmethod
    def _safe_pad(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        _, h, w = img.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            img = TF.pad(img, [0, 0, pad_w, pad_h], padding_mode="reflect")
        return img

    def _paired_random_crop(
        self,
        hr_left: torch.Tensor,
        hr_right: torch.Tensor,
        lr_left: torch.Tensor,
        lr_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random crop in LR space, mapped to HR space for strict alignment."""
        _, lr_h, lr_w = lr_left.shape

        lr_left = self._safe_pad(lr_left, self.patch_size, self.patch_size)
        lr_right = self._safe_pad(lr_right, self.patch_size, self.patch_size)
        _, lr_h, lr_w = lr_left.shape

        top_lr = random.randint(0, lr_h - self.patch_size)
        left_lr = random.randint(0, lr_w - self.patch_size)

        hr_patch = self.patch_size * self.scale
        top_hr = top_lr * self.scale
        left_hr = left_lr * self.scale

        hr_left = self._safe_pad(hr_left, top_hr + hr_patch, left_hr + hr_patch)
        hr_right = self._safe_pad(hr_right, top_hr + hr_patch, left_hr + hr_patch)

        lr_left = self._crop(lr_left, top_lr, left_lr, self.patch_size, self.patch_size)
        lr_right = self._crop(lr_right, top_lr, left_lr, self.patch_size, self.patch_size)
        hr_left = self._crop(hr_left, top_hr, left_hr, hr_patch, hr_patch)
        hr_right = self._crop(hr_right, top_hr, left_hr, hr_patch, hr_patch)

        return hr_left, hr_right, lr_left, lr_right

    @staticmethod
    def _augment(
        hr_l: torch.Tensor,
        hr_r: torch.Tensor,
        lr_l: torch.Tensor,
        lr_r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stereo-safe augmentations:
        - Horizontal flip + view swap (preserves epipolar geometry)
        - Vertical flip (disparities stay horizontal)
        - Channel shuffle (color augmentation)
        - No 90° rotations (would break horizontal epipolar assumption)
        """
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
            perm = torch.randperm(3)
            hr_l = hr_l[perm]
            hr_r = hr_r[perm]
            lr_l = lr_l[perm]
            lr_r = lr_r[perm]

        return hr_l, hr_r, lr_l, lr_r

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        sample = self.samples[idx]

        hr_left = self._load_image(sample["hr_left"])
        hr_right = self._load_image(sample["hr_right"])
        lr_left = self._load_image(sample["lr_left"])
        lr_right = self._load_image(sample["lr_right"])
        has_lr = "lr_left" in sample and "lr_right" in sample



        if self.is_train:
            # Training patches already have LR images, but may need re-cropping
            # if patch_size differs from the stored patch size
            if lr_left is not None:
                hr_left, hr_right, lr_left, lr_right = self._paired_random_crop(
                    hr_left, hr_right, lr_left, lr_right
                )
            if self.augment:
                hr_left, hr_right, lr_left, lr_right = self._augment(
                    hr_left, hr_right, lr_left, lr_right
                )
        else:
            # Test: use provided LR or generate by downsampling
            if lr_left is None or lr_right is None:
                _, h, w = hr_left.shape
                lr_h = h // self.scale
                lr_w = w // self.scale
                # Crop HR to be exactly divisible by scale
                hr_left = hr_left[:, : lr_h * self.scale, : lr_w * self.scale]
                hr_right = hr_right[:, : lr_h * self.scale, : lr_w * self.scale]
                lr_left = TF.resize(
                    hr_left,
                    [lr_h, lr_w],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True,
                )
                lr_right = TF.resize(
                    hr_right,
                    [lr_h, lr_w],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True,
                )

        lr_left = lr_left.clamp(0, 1)
        lr_right = lr_right.clamp(0, 1)
        hr_left = hr_left.clamp(0, 1)
        hr_right = hr_right.clamp(0, 1)

        return {
            "lr_left": lr_left,
            "lr_right": lr_right,
            "hr_left": hr_left,
            "hr_right": hr_right,
            "filename": f"{sample['scene']}.png",
        }


class StereoMixup:
    """Mixup augmentation that preserves stereo consistency."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        b = batch["lr_left"].size(0)
        idx = torch.randperm(b)
        for key in ["lr_left", "lr_right", "hr_left", "hr_right"]:
            batch[key] = lam * batch[key] + (1 - lam) * batch[key][idx]
        return batch


def build_dataset(args, split: str = "train") -> StereoSRDataset:
    """
    Build dataset from args namespace.

    Expected args attributes:
        args.data_root:   Path to StereoSR root
        args.scale:       SR scale factor (2 or 4)
        args.patch_size:  Training patch size in LR space
        args.augment:     Whether to augment (bool)
        args.test_source: Test dataset name, e.g. "Flickr1024" (for val/test)
    """
    if split == "train":
        return StereoSRDataset(
            data_root=args.data_root,
            scale=args.scale,
            patch_size=args.patch_size,
            augment=args.augment,
            is_train=True,
        )
    elif split in ("val", "test"):
        return StereoSRDataset(
            data_root=args.data_root,
            scale=args.scale,
            augment=False,
            is_train=False,
            test_source=args.test_source,
        )
    else:
        raise ValueError(f"Unknown split: {split}")