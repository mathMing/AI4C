# -*- coding: utf-8 -*-
"""
prepare_ddr.py
==============
把 DDR 数据集转换为 train.py 可直接使用的 ImageFolder 目录结构。

DDR 原始结构：
    DDR-dataset/
    ├── DR_grading/
    │   ├── train.txt    (6835 行: "007-0004-000.jpg 0")
    │   ├── valid.txt    (2733 行)
    │   └── test.txt     (4105 行)
    └── lesion_segmentation/
        ├── train/image/  (~6750 张)
        ├── valid/image/  (~2733 张)
        └── test/image/   (~4105 张)

输出结构（output_root）：
    DDR_processed/
    ├── 0_No_DR/
    │   ├── 007-0004-000.jpg
    │   └── ...
    ├── 1_Mild_NPDR/
    ├── 2_Moderate_NPDR/
    ├── 3_Severe_NPDR/
    ├── 4_Proliferative_DR/
    └── splits/
        ├── train.txt   (再生成一次，方便后续检查)
        ├── valid.txt
        └── test.txt

用法：
    python prepare_ddr.py \
      --ddr_root "/root/autodl-tmp/AI4C/algo/data/DDR dataset/DDR-dataset" \
      --output_root "/root/autodl-tmp/AI4C/algo/data/DDR_processed"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

CLASS_NAMES = {
    0: "0_No_DR",
    1: "1_Mild_NPDR",
    2: "2_Moderate_NPDR",
    3: "3_Severe_NPDR",
    4: "4_Proliferative_DR",
}

# DDR 标签文件里只有 0/1/2/3/4 数字，简单粗暴 map 即可
def map_label(line_label: str) -> int | None:
    try:
        lab = int(line_label.strip())
        if lab in CLASS_NAMES:
            return lab
    except ValueError:
        pass
    return None


def resolve_image(img_name: str, lesion_root: Path) -> Path | None:
    """
    在 lesion_segmentation/{split}/image/ 下找图片
    训练脚本调用前会预先 cd 到 ddr_root/lesion_segmentation，所以传入名字
    """
    # 直接在三处都找一下（保险）
    for split in ("train", "valid", "test"):
        p = lesion_root / split / "image" / img_name
        if p.exists():
            return p
    return None


def process_split(txt_path: Path, lesion_root: Path, output_root: Path, split_name: str) -> dict:
    """
    处理单个 split (train/valid/test) 的标签文件
    返回: {label_int: count}
    """
    counts = {i: 0 for i in CLASS_NAMES}
    missing = 0
    total = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_name, lbl_str = parts[0], parts[1]
            label = map_label(lbl_str)
            if label is None:
                print(f"[{split_name}] 跳过无效标签: {line.strip()!r}")
                continue

            src = resolve_image(img_name, lesion_root)
            if src is None:
                missing += 1
                if missing <= 3:
                    print(f"[{split_name}] 找不到图: {img_name}")
                continue

            dst_dir = output_root / CLASS_NAMES[label]
            dst_dir.mkdir(parents=True, exist_ok=True)
            # 在文件名里加 split 前缀，避免 train/valid/test 同一张图互相覆盖（test 图也可能出现在 train）
            dst = dst_dir / f"{split_name}_{img_name}"
            if not dst.exists():
                shutil.copy2(src, dst)

            counts[label] += 1
            total += 1

    print(f"[{split_name}] 处理完成: 共 {total} 张, 缺失 {missing} 张")
    print(f"[{split_name}] 类别分布: " + ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in CLASS_NAMES))
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ddr_root",
        type=str,
        default="/root/autodl-tmp/AI4C/algo/data/DDR dataset/DDR-dataset",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="/root/autodl-tmp/AI4C/algo/data/DDR_processed",
    )
    p.add_argument("--no_copy", action="store_true", help="用 symlink 代替 copy，省空间")
    args = p.parse_args()

    ddr_root = Path(args.ddr_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    grading_dir = ddr_root / "DR_grading"
    lesion_root = ddr_root / "lesion_segmentation"

    if not grading_dir.exists():
        raise FileNotFoundError(f"找不到 {grading_dir}")
    if not lesion_root.exists():
        raise FileNotFoundError(f"找不到 {lesion_root}")

    print(f"DDR root:      {ddr_root}")
    print(f"Lesion root:   {lesion_root}")
    print(f"Output root:   {output_root}")
    print(f"Mode:          {'symlink' if args.no_copy else 'copy'}")
    print("=" * 60)

    total_counts = {i: 0 for i in CLASS_NAMES}
    for split in ("train", "valid", "test"):
        txt = grading_dir / f"{split}.txt"
        if not txt.exists():
            print(f"??  跳过 {split}: {txt} 不存在")
            continue
        counts = process_split(txt, lesion_root, output_root, split)
        for i in CLASS_NAMES:
            total_counts[i] += counts[i]

    print("=" * 60)
    print("? DDR 数据集预处理完成")
    print("类别合计: " + ", ".join(f"{CLASS_NAMES[i]}={total_counts[i]}" for i in CLASS_NAMES))
    print(f"总样本:   {sum(total_counts.values())}")
    print(f"输出目录: {output_root}")


if __name__ == "__main__":
    main()
