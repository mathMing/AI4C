# -*- coding: utf-8 -*-
"""
prepare_ddr.py
==============
°Ñ DDR Êý¾Ý¼¯×ª»»Îª train.py ¿ÉÖ±½ÓÊ¹ÓÃµÄ ImageFolder Ä¿Â¼½á¹¹¡£

DDR Ô­Ê¼½á¹¹£º
    DDR-dataset/
    ©À©¤©¤ DR_grading/
    ©¦   ©À©¤©¤ train.txt    (6835 ÐÐ: "007-0004-000.jpg 0")
    ©¦   ©À©¤©¤ valid.txt    (2733 ÐÐ)
    ©¦   ©¸©¤©¤ test.txt     (4105 ÐÐ)
    ©¸©¤©¤ lesion_segmentation/
        ©À©¤©¤ train/image/  (~6750 ÕÅ)
        ©À©¤©¤ valid/image/  (~2733 ÕÅ)
        ©¸©¤©¤ test/image/   (~4105 ÕÅ)

Êä³ö½á¹¹£¨output_root£©£º
    DDR_processed/
    ©À©¤©¤ 0_No_DR/
    ©¦   ©À©¤©¤ 007-0004-000.jpg
    ©¦   ©¸©¤©¤ ...
    ©À©¤©¤ 1_Mild_NPDR/
    ©À©¤©¤ 2_Moderate_NPDR/
    ©À©¤©¤ 3_Severe_NPDR/
    ©À©¤©¤ 4_Proliferative_DR/
    ©¸©¤©¤ splits/
        ©À©¤©¤ train.txt   (ÔÙÉú³ÉÒ»´Î£¬·½±ãºóÐø¼ì²é)
        ©À©¤©¤ valid.txt
        ©¸©¤©¤ test.txt

ÓÃ·¨£º
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

# DDR ±êÇ©ÎÄ¼þÀïÖ»ÓÐ 0/1/2/3/4 Êý×Ö£¬¼òµ¥´Ö±© map ¼´¿É
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
    ÔÚ lesion_segmentation/{split}/image/ ÏÂÕÒÍ¼Æ¬
    ÑµÁ·½Å±¾µ÷ÓÃÇ°»áÔ¤ÏÈ cd µ½ ddr_root/lesion_segmentation£¬ËùÒÔ´«ÈëÃû×Ö
    """
    # Ö±½ÓÔÚÈý´¦¶¼ÕÒÒ»ÏÂ£¨±£ÏÕ£©
    for split in ("train", "valid", "test"):
        p = lesion_root / split / "image" / img_name
        if p.exists():
            return p
    return None


def process_split(txt_path: Path, lesion_root: Path, output_root: Path, split_name: str) -> dict:
    """
    ´¦Àíµ¥¸ö split (train/valid/test) µÄ±êÇ©ÎÄ¼þ
    ·µ»Ø: {label_int: count}
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
                print(f"[{split_name}] Ìø¹ýÎÞÐ§±êÇ©: {line.strip()!r}")
                continue

            src = resolve_image(img_name, lesion_root)
            if src is None:
                missing += 1
                if missing <= 3:
                    print(f"[{split_name}] ÕÒ²»µ½Í¼: {img_name}")
                continue

            dst_dir = output_root / CLASS_NAMES[label]
            dst_dir.mkdir(parents=True, exist_ok=True)
            # ÔÚÎÄ¼þÃûÀï¼Ó split Ç°×º£¬±ÜÃâ train/valid/test Í¬Ò»ÕÅÍ¼»¥Ïà¸²¸Ç£¨test Í¼Ò²¿ÉÄÜ³öÏÖÔÚ train£©
            dst = dst_dir / f"{split_name}_{img_name}"
            if not dst.exists():
                shutil.copy2(src, dst)

            counts[label] += 1
            total += 1

    print(f"[{split_name}] ´¦ÀíÍê³É: ¹² {total} ÕÅ, È±Ê§ {missing} ÕÅ")
    print(f"[{split_name}] Àà±ð·Ö²¼: " + ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in CLASS_NAMES))
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
    p.add_argument("--no_copy", action="store_true", help="ÓÃ symlink ´úÌæ copy£¬Ê¡¿Õ¼ä")
    args = p.parse_args()

    ddr_root = Path(args.ddr_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    grading_dir = ddr_root / "DR_grading"
    lesion_root = ddr_root / "lesion_segmentation"

    if not grading_dir.exists():
        raise FileNotFoundError(f"ÕÒ²»µ½ {grading_dir}")
    if not lesion_root.exists():
        raise FileNotFoundError(f"ÕÒ²»µ½ {lesion_root}")

    print(f"DDR root:      {ddr_root}")
    print(f"Lesion root:   {lesion_root}")
    print(f"Output root:   {output_root}")
    print(f"Mode:          {'symlink' if args.no_copy else 'copy'}")
    print("=" * 60)

    total_counts = {i: 0 for i in CLASS_NAMES}
    for split in ("train", "valid", "test"):
        txt = grading_dir / f"{split}.txt"
        if not txt.exists():
            print(f"??  Ìø¹ý {split}: {txt} ²»´æÔÚ")
            continue
        counts = process_split(txt, lesion_root, output_root, split)
        for i in CLASS_NAMES:
            total_counts[i] += counts[i]

    print("=" * 60)
    print("? DDR Êý¾Ý¼¯Ô¤´¦ÀíÍê³É")
    print("Àà±ðºÏ¼Æ: " + ", ".join(f"{CLASS_NAMES[i]}={total_counts[i]}" for i in CLASS_NAMES))
    print(f"×ÜÑù±¾:   {sum(total_counts.values())}")
    print(f"Êä³öÄ¿Â¼: {output_root}")


if __name__ == "__main__":
    main()
