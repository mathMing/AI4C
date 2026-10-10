# -*- coding: utf-8 -*-
"""
fundus_augment.py
=================
眼底图像数据增强策略：

1. **颜色归一化**：眼底图常有色偏（不同设备/不同医院），做 CLAHE
2. **几何变换**：水平翻转、垂直翻转、旋转 ±15°、缩放 0.9-1.1
3. **随机擦除**：模拟遮挡（5-15% 区域）
4. **标准化**：ImageNet mean/std（与 ImageNet 预训练 ViT 对齐）

两类强度：
- mild：仅 flip + resize
- strong：完整套件
"""
from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageEnhance
from torchvision import transforms

FUNDUS_MEAN = (0.485, 0.456, 0.406)
FUNDUS_STD = (0.229, 0.224, 0.225)


# ------------------------- Pillow 自定义增强 ------------------------- #
class CLAHE:
    """对比度受限的自适应直方图均衡（眼底色偏修正）。"""

    def __init__(self, clip_limit: float = 2.0, tile: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile = tile

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            import cv2
            import numpy as np
            arr = np.array(img)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile)
            l2 = clahe.apply(l)
            lab2 = cv2.merge([l2, a, b])
            arr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
            return Image.fromarray(arr2)
        except Exception:
            return img


class RandomBrightnessContrast:
    """随机亮度 + 对比度（模拟不同眼底采集条件）。"""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: Image.Image) -> Image.Image:
        import random
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        return img


# ------------------------- build transforms ------------------------- #
def build_train_transform(img_size: int = 224, mode: str = "strong"):
    """构建训练 transform。"""
    if mode == "mild":
        return transforms.Compose([
            CLAHE(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(FUNDUS_MEAN, FUNDUS_STD),
        ])

    # strong (default)
    return transforms.Compose([
        CLAHE(),
        transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        RandomBrightnessContrast(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(FUNDUS_MEAN, FUNDUS_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def build_val_transform(img_size: int = 224):
    """验证 / 测试 transform。"""
    return transforms.Compose([
        CLAHE(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(FUNDUS_MEAN, FUNDUS_STD),
    ])
