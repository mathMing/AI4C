#!/usr/bin/env python3
import shutil, re, random, os
from pathlib import Path

SRC = Path("/root/autodl-tmp/AI4C/algo/data/DDR-dataset/DR_grading")
DST = Path("/root/autodl-tmp/AI4C/algo/data/DDR_real")
SEED = 42

LABEL_MAP = {
    "000": "0_No_DR", "100": "1_Mild_NPDR",
    "200": "2_Moderate_NPDR", "300": "3_Severe_NPDR",
    "400": "4_Proliferative_DR",
}

def get_label(stem: str):
    m = re.search(r'(\d{3})$', stem)
    if m and m.group(1) in LABEL_MAP:
        return LABEL_MAP[m.group(1)]
    return None

# 清空旧目录
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)

# 收集 + 去重
seen = set()
all_images = []
for split in ["train", "valid", "test"]:
    n_split = 0
    for f in (SRC / split).glob("*.jpg"):
        lbl = get_label(f.stem)
        if lbl and f.stem not in seen:
            seen.add(f.stem)
            all_images.append((f, lbl))
            n_split += 1
    print(f"  {split}: {n_split} 张")
print(f"去重后: {len(all_images)} 张\n")

# 统计
from collections import Counter
counter = Counter([cls for _, cls in all_images])
for cls_name, cnt in counter.most_common():
    print(f"  {cls_name}: {cnt}")
print()

# 8:2 划分
random.seed(SEED)
random.shuffle(all_images)
n_val = int(len(all_images) * 0.2)
val_imgs = all_images[:n_val]
train_imgs = all_images[n_val:]
print(f"train={len(train_imgs)}  val={len(val_imgs)}\n")

# 软链接（省 8.5GB 空间，秒级完成）
for phase, imgs in [("train", train_imgs), ("val", val_imgs)]:
    for i, (src, cls) in enumerate(imgs):
        dst_dir = DST / phase / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            os.symlink(src, dst)
        if (i + 1) % 2000 == 0:
            print(f"  {phase}: {i+1}/{len(imgs)}")
    print(f"  ✅ {phase} done")

# 验证
print("\n=== 验证 ===")
for phase in ["train", "val"]:
    for cls_name in LABEL_MAP.values():
        count = len(list((DST / phase / cls_name).iterdir()))
        print(f"  {phase}/{cls_name}: {count}")
print(f"\n✅ {DST}")
