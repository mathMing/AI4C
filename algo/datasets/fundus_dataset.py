# -*- coding: utf-8 -*-
"""
fundus_dataset.py
=================
眼底图像 Dataset：

输入数据组织方式（自动检测）：
1. **目录结构**（默认）:
        data/
        ├── 0_No_DR/
        │   ├── img001.jpg
        │   └── ...
        ├── 1_Mild_NPDR/
        └── 4_Proliferative_DR/

2. **CSV 元数据**:
        data/images/xxx.jpg
        data/train.csv  -> id_code, diagnosis
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


class FundusDataset(Dataset):
    """眼底图像分类 Dataset。"""

    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[Callable] = None):
        """
        Args:
            samples: [(image_path, label), ...]
            transform: 图像 transform
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[FundusDataset] failed to open {path}: {e}")
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


# ------------------------- 目录结构扫描 ------------------------- #
def scan_dir_structure(root: Path) -> List[Tuple[str, int]]:
    """扫描 root/0_No_DR/*.jpg, root/1_Mild_NPDR/*.jpg, ...

    支持子目录名：0, 1, 2, 3, 4（不论是否带前缀）
    """
    samples = []
    if not root.exists():
        return samples

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        # 提取数字 label
        try:
            label = int(cls_dir.name.split("_")[0])
        except ValueError:
            continue
        if label not in CLASS_NAMES:
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                samples.append((str(img_path), label))
    return samples


def load_samples(data_root: Path) -> List[Tuple[str, int]]:
    """根据 data_root 自动检测数据格式。"""
    # 优先级 1：目录结构
    dir_samples = scan_dir_structure(data_root)
    if dir_samples:
        print(f"[FundusDataset] loaded {len(dir_samples)} samples from dir structure at {data_root}")
        return dir_samples

    # 优先级 2：CSV
    for csv_name in ("train.csv", "labels.csv", "metadata.csv"):
        csv_path = data_root / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # 推断列名
            id_col = "id_code" if "id_code" in df.columns else df.columns[0]
            label_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]
            samples = []
            img_dir = data_root / "images"
            for _, row in df.iterrows():
                img_path = img_dir / f"{row[id_col]}.jpg"
                if img_path.exists():
                    samples.append((str(img_path), int(row[label_col])))
            print(f"[FundusDataset] loaded {len(samples)} samples from {csv_path}")
            return samples

    print(f"[FundusDataset] no data found in {data_root}")
    return []


# ------------------------- DataLoader 构建 ------------------------- #
def build_dataloaders(
    data_root: Path,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, int]]:
    """构建 train/val/test DataLoader + 类别权重。"""
    samples = load_samples(data_root)
    if not samples:
        raise FileNotFoundError(f"no data found at {data_root}")

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(samples, [n_train, n_val, n_test], generator=generator)

    # wrap with transforms
    train_ds = FundusDataset(train_set, transform=train_transform)
    val_ds = FundusDataset(val_set, transform=val_transform)
    test_ds = FundusDataset(test_set, transform=val_transform)

    # 类别权重（按训练集计算）
    train_labels = [s[1] for s in train_set]
    class_counts = {c: train_labels.count(c) for c in CLASS_NAMES}
    total = sum(class_counts.values())
    class_weights = {c: total / (len(class_counts) * max(1, cnt)) for c, cnt in class_counts.items()}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights
