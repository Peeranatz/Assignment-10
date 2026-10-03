"""
dataset.py: High-performance Dataset pipeline for PSU Reservoir Lane Segmentation
Pre-caches rasterized masks and resized images in memory for high-speed GPU training.
Author: Peeranat Chunhok (Student ID: 6710110295)
Course: 241-353 Artificial Intelligence Ecosystem Module
Department: Artificial Intelligence Engineering, Faculty of Engineering, Prince of Songkla University
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PSULaneDataset(Dataset):
    """
    PSU Reservoir Custom Lane Dataset.
    Loads raw RGB images and YOLO-seg polygon annotations (Class 0: Lane),
    rasterizes polygons into 2D binary segmentation masks,
    and applies realistic outdoor data augmentation.
    """
    def __init__(
        self,
        samples: List[Tuple[Path, Path]],
        img_size: Tuple[int, int] = (128, 128),
        is_train: bool = True,
        augment: bool = True,
        preload: bool = True
    ):
        self.samples = samples
        self.img_size = img_size  # (Height, Width)
        self.is_train = is_train
        self.augment = augment and is_train
        self.preload = preload
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Pre-cache images and rasterized masks for fast iteration
        self.cached_images = []
        self.cached_masks = []
        self.cached_stems = []

        if self.preload:
            target_h, target_w = self.img_size
            for img_path, lbl_path in self.samples:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                mask = self._rasterize_polygon(lbl_path, h, w)

                img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

                self.cached_images.append(img_resized)
                self.cached_masks.append(mask_resized)
                self.cached_stems.append(img_path.stem)

    def __len__(self) -> int:
        if self.preload:
            return len(self.cached_images)
        return len(self.samples)

    def _rasterize_polygon(self, label_path: Path, height: int, width: int) -> np.ndarray:
        """
        Converts Ultralytics YOLO-seg normalized polygon coordinates into a binary raster mask.
        Lane class is class 0 in the PSU Reservoir track dataset.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        if not label_path.exists():
            return mask

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # We segment Lane area (Class 0)
                    if cls_id == 0:
                        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                        coords[:, 0] *= width
                        coords[:, 1] *= height
                        pts = coords.astype(np.int32)
                        cv2.fillPoly(mask, [pts], 1)
        return mask

    def _apply_augmentation(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies domain-specific augmentations as required by Assignment 10:
        1. Small White Balance Shift (color channel temperature variance)
        2. Small Brightness & Contrast Shift (mimic sunlight and cloud shadow noise)
        3. Small Blur (motion / camera focus noise)
        4. Random Horizontal Flip (with corresponding mask flip)
        """
        img = img.copy()
        mask = mask.copy()

        # 1. Random White Balance shift
        if random.random() < 0.5:
            r_scale = random.uniform(0.90, 1.10)
            g_scale = random.uniform(0.90, 1.10)
            b_scale = random.uniform(0.90, 1.10)
            img = img.astype(np.float32)
            img[:, :, 0] = np.clip(img[:, :, 0] * b_scale, 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * g_scale, 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * r_scale, 0, 255)
            img = img.astype(np.uint8)

        # 2. Random Brightness & Contrast shift (Sunlight / Shadow noise)
        if random.random() < 0.6:
            alpha = random.uniform(0.85, 1.20)  # Contrast
            beta = random.uniform(-20, 20)      # Brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # 3. Random Gaussian Blur
        if random.random() < 0.30:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # 4. Random Horizontal Flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.preload:
            img = self.cached_images[idx]
            mask = self.cached_masks[idx]
            stem = self.cached_stems[idx]
        else:
            img_path, lbl_path = self.samples[idx]
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            mask = self._rasterize_polygon(lbl_path, h, w)
            target_h, target_w = self.img_size
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            stem = img_path.stem

        # Apply Data Augmentation if in training mode
        if self.augment:
            img, mask = self._apply_augmentation(img, mask)

        # Convert BGR to RGB and Normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - self.mean) / self.std

        # Convert to PyTorch tensors (CHW for image, 1HW for mask)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor, stem


def collect_dataset_samples(dataset_root: Path) -> List[Tuple[Path, Path]]:
    """
    Recursively scans and matches image (.jpg) and label (.txt) files in the PSU dataset.
    """
    samples = []
    
    img_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "images",
        dataset_root
    ]
    
    found_imgs = {}
    for d in img_dirs:
        if d.exists():
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                for p in d.glob(ext):
                    if p.name not in found_imgs:
                        found_imgs[p.stem] = p
                        
    lbl_dirs = [
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
        dataset_root / "labels",
        dataset_root.parent / "work4" / "sam2_generated_labels",
        dataset_root.parent / "work4" / "psu_reservoir_dataset" / "labels" / "train"
    ]
    
    found_lbls = {}
    for d in lbl_dirs:
        if d.exists():
            for p in d.glob("*.txt"):
                if p.stem not in found_lbls:
                    found_lbls[p.stem] = p
                    
    # Match pairs
    for stem, img_path in sorted(found_imgs.items()):
        lbl_path = found_lbls.get(stem)
        if lbl_path is None:
            candidate = img_path.parents[1] / "labels" / img_path.parent.name / f"{stem}.txt"
            if candidate.exists():
                lbl_path = candidate
            else:
                candidate2 = img_path.parent / f"{stem}.txt"
                if candidate2.exists():
                    lbl_path = candidate2
                    
        if lbl_path and lbl_path.exists():
            samples.append((img_path, lbl_path))

    return samples


def get_dataloaders(
    dataset_root: str,
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 16,
    train_ratio: float = 0.65,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Creates reproducible Train (65%) and Test (35%) DataLoaders with fast in-memory caching.
    """
    root = Path(dataset_root)
    all_samples = collect_dataset_samples(root)
    if not all_samples:
        raise FileNotFoundError(f"No valid image-label pairs found in {dataset_root}")

    # Deterministic split (65% Train : 35% Test)
    rng = random.Random(seed)
    shuffled_samples = list(all_samples)
    rng.shuffle(shuffled_samples)
    
    n_train = int(len(shuffled_samples) * train_ratio)
    train_samples = shuffled_samples[:n_train]
    test_samples = shuffled_samples[n_train:]

    train_dataset = PSULaneDataset(train_samples, img_size=img_size, is_train=True, augment=True, preload=True)
    test_dataset = PSULaneDataset(test_samples, img_size=img_size, is_train=False, augment=False, preload=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, test_loader, train_samples, test_samples


if __name__ == "__main__":
    test_dir = r"D:\work_AJfern\work4\psu_reservoir_dataset"
    print(f"Scanning and caching dataset from: {test_dir}")
    train_loader, test_loader, train_s, test_s = get_dataloaders(
        dataset_root=test_dir,
        img_size=(128, 128),
        batch_size=16,
        train_ratio=0.65,
        seed=42,
        num_workers=0
    )
    print(f"Total dataset matched: {len(train_s) + len(test_s)} samples")
    print(f"Train split (65%): {len(train_s)} samples ({len(train_loader)} batches)")
    print(f"Test split (35%): {len(test_s)} samples ({len(test_loader)} batches)")
