import os, shutil
from pathlib import Path

src = Path("/root/autodl-tmp/AI4C/algo/data/DDR-dataset/DR_grading")
dst = Path("/root/autodl-tmp/AI4C/algo/data/DDR_processed")

# 5 个 DR 分级（只取 0-4，丢弃 5）
classes = {
    "0": "0_No_DR",
    "1": "1_Mild_NPDR",
    "2": "2_Moderate_NPDR",
    "3": "3_Severe_NPDR",
    "4": "4_Proliferative_DR",
}
for cname in classes.values():
    (dst / cname).mkdir(parents=True, exist_ok=True)

splits = {"train": "train", "valid": "valid", "test": "test"}
stats = {c: 0 for c in classes.values()}
total_kept = 0
total_skip = 0

for split_file, img_subdir in splits.items():
    txt = src / f"{split_file}.txt"
    img_dir = src / img_subdir   # train/ valid/ test/
    if not img_dir.exists():
        # 验证/测试图直接在 DR_grading/ 根目录
        img_dir = src
    print(f"📄 {split_file}.txt  →  图目录: {img_dir}")
    
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2: continue
            img_name, label = parts[0], parts[1]
            
            # 关键：train 文件夹里图，验证/测试图在 DR_grading/ 根目录
            img_path = src / img_subdir / img_name if (src / img_subdir).exists() else src / img_name
            if not img_path.exists():
                img_path = src / img_name  # 兜底
                if not img_path.exists():
                    total_skip += 1
                    continue
            
            if label not in classes:  # 跳过 5
                total_skip += 1
                continue
            
            cname = classes[label]
            # 加 split 前缀避免不同 split 重名图覆盖
            new_name = f"{split_file}_{img_name}"
            shutil.copy(img_path, dst / cname / new_name)
            stats[cname] += 1
            total_kept += 1

print("\n" + "="*60)
print("✅ 修复完成！")
for c, n in stats.items():
    print(f"  {c:30s} : {n:5d} 张")
print(f"\n  总保留: {total_kept} 张")
print(f"  跳过(标签 5 / 找不到图): {total_skip} 张")
print("="*60)
