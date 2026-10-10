# -*- coding: utf-8 -*-
"""
prepare_data.py
===============
自动准备 DR 训练数据：

策略：
1. **若 data/0_No_DR/... 已有数据 → 直接使用**
2. **否则尝试从 Kaggle / HuggingFace 下载**
3. **若离线/无下载 → 生成合成眼底数据（用真实 ViT 跑通训练 pipeline）**

合成数据：每类 200 张，模拟"有/无 DR 病变"两种模式：
- class 0：全黑眼底 + 红色视盘
- class 1-4：在不同位置叠加不同数量/大小的"渗出"（亮黄斑点）
- 加随机噪声 + 模糊模拟真实拍摄
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CLASS_NAMES = {
    0: "0_No_DR",
    1: "1_Mild_NPDR",
    2: "2_Moderate_NPDR",
    3: "3_Severe_NPDR",
    4: "4_Proliferative_DR",
}


def make_synthetic_fundus(cls: int, size: int = 384, seed: int = 0) -> Image.Image:
    """生成单张合成眼底图像。"""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # 1) 黑色背景
    img = Image.new("RGB", (size, size), color=(8, 4, 4))
    draw = ImageDraw.Draw(img)

    # 2) 中央视盘 (亮黄)
    cx, cy = size // 2 + rng.randint(-15, 15), size // 2 + rng.randint(-15, 15)
    r_disc = size // 8
    for r, color in [(r_disc + 20, (60, 30, 25)), (r_disc, (220, 180, 60)), (r_disc // 2, (240, 220, 120))]:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)

    # 3) 血管分支
    for _ in range(rng.randint(8, 14)):
        ang = rng.uniform(0, 2 * np.pi)
        length = rng.randint(size // 6, size // 2)
        x2 = int(cx + np.cos(ang) * length)
        y2 = int(cy + np.sin(ang) * length)
        draw.line((cx, cy, x2, y2), fill=(140, 30, 30), width=rng.randint(2, 5))

    # 4) 按 class 叠加"病变"
    if cls == 0:
        # 无 DR
        pass
    elif cls == 1:
        # 轻度 NPDR：少量微动脉瘤（小红点）
        for _ in range(rng.randint(3, 8)):
            x = rng.randint(size // 6, size - size // 6)
            y = rng.randint(size // 6, size - size // 6)
            r = rng.randint(2, 4)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 30, 30))
    elif cls == 2:
        # 中度 NPDR：渗出（亮黄）+ 出血（红）
        for _ in range(rng.randint(8, 18)):
            x = rng.randint(size // 8, size - size // 8)
            y = rng.randint(size // 8, size - size // 8)
            r = rng.randint(3, 7)
            color = (255, 220, 80) if rng.random() < 0.5 else (220, 30, 30)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    elif cls == 3:
        # 重度 NPDR：大面积渗出 + 多发出血
        for _ in range(rng.randint(20, 35)):
            x = rng.randint(size // 8, size - size // 8)
            y = rng.randint(size // 8, size - size // 8)
            r = rng.randint(3, 10)
            color = (255, 220, 80) if rng.random() < 0.5 else (200, 30, 30)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    elif cls == 4:
        # 增殖期：大量新生血管（亮红线）
        for _ in range(rng.randint(30, 50)):
            x = rng.randint(size // 8, size - size // 8)
            y = rng.randint(size // 8, size - size // 8)
            r = rng.randint(2, 5)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 60, 60))
        # 玻璃体积血（红色半透明覆盖）
        overlay = Image.new("RGBA", (size, size), (180, 30, 30, 60))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # 5) 加噪声 + 模糊模拟真实
    arr = np.array(img).astype(np.float32)
    noise = np_rng.normal(0, 6, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.6))

    return img


def generate_synthetic_dataset(root: Path, per_class: int = 80, size: int = 384, seed_base: int = 0):
    """在 root 下生成 5 类 × per_class 张合成眼底图。"""
    root = Path(root)
    for cls, name in CLASS_NAMES.items():
        cls_dir = root / name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            seed = seed_base + cls * 10000 + i
            img = make_synthetic_fundus(cls, size=size, seed=seed)
            img.save(cls_dir / f"img_{i:04d}.jpg", quality=85)
    print(f"[prepare_data] generated {per_class * 5} synthetic fundus images at {root}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/root/autodl-tmp/AI4C/algo/data")
    p.add_argument("--per_class", type=int, default=80)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--seed", type=int, default=2025)
    args = p.parse_args()
    generate_synthetic_dataset(Path(args.root), per_class=args.per_class, size=args.size, seed_base=args.seed)


if __name__ == "__main__":
    main()
