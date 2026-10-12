#!/usr/bin/env python3
import shutil, re, os
from pathlib import Path

SRC = Path("/root/autodl-tmp/AI4C/algo/data/DDR-dataset/DR_grading")
DST = Path("/root/autodl-tmp/AI4C/algo/data/DDR_real")

LABEL_MAP = {
    "000": "0_No_DR", "100": "1_Mild_NPDR",
    "200": "2_Moderate_NPDR", "300": "3_Severe_NPDR",
    "400": "4_Proliferative_DR",
}

def get_label(stem: str):
    m = re.search(r'(\d{3})$', stem)
    return LABEL_MAP.get(m.group(1)) if m else None

# 清空
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)

# 建子目录
for cls_name in LABEL_MAP.values():
    (DST / cls_name).mkdir(parents=True, exist_ok=True)

# 收集所有图片（train+valid+test 合并）
seen = set()
n_total, n_skip = 0, 0
for split in ["train", "valid", "test"]:
    for f in (SRC / split).glob("*.jpg"):
        cls = get_label(f.stem)
        if not cls:
            n_skip += 1
            continue
        if f.stem in seen:
            continue
        seen.add(f.stem)
        dst = DST / cls / f.name
        if not dst.exists():
            os.symlink(f, dst)
        n_total += 1

print(f"✅ {n_total} 张图片（跳过 {n_skip} 张异常）")
print(f"   目录: {DST}\n")

for cls_name in LABEL_MAP.values():
    cnt = len(list((DST / cls_name).iterdir()))
    print(f"   {cls_name}: {cnt}")

# 测试 build_dataloaders 能不能扫到
print("\n=== 测试 scan_dir_structure ===")
import sys
sys.path.insert(0, '/root/autodl-tmp/AI4C/algo')
from datasets.fundus_dataset import scan_dir_structure
samples = scan_dir_structure(DST)
print(f"扫描到 {len(samples)} 个样本")
if samples:
    print(f"  示例: {samples[0]}")
    print(f"  类别分布: {set(s[1] for s in samples)}")
