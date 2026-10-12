"""准备 DDR 测试集给 AutoMorph"""
import os, csv
from pathlib import Path

DDR_TEST = Path("/root/autodl-tmp/AI4C/algo/data/DDR-dataset/DR_grading/test")
AUTOMORPH = Path("/root/autodl-tmp/AI4C/AutoMorph-main")
INPUT_DIR = AUTOMORPH / "input" / "eyepacs"
RESOLUTION_CSV = AUTOMORPH / "resolution_information.csv"
MAPPING_CSV = AUTOMORPH / "name_mapping.csv"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
jpgs = sorted([p for p in DDR_TEST.iterdir() if p.suffix.lower() == '.jpg'])
print(f"找到 {len(jpgs)} 张测试图")

with open(RESOLUTION_CSV, 'w', newline='') as fr, open(MAPPING_CSV, 'w', newline='') as fm:
    wr_r = csv.writer(fr)
    wr_r.writerow(['fundus', 'res'])
    wr_m = csv.writer(fm)
    wr_m.writerow(['orig_name', 'id'])
    for i, jp in enumerate(jpgs):
        new_id = f"{i+1:05d}.jpg"
        target = INPUT_DIR / new_id
        if not target.exists():
            os.symlink(jp.absolute(), target)
        wr_r.writerow([new_id, '1'])
        wr_m.writerow([jp.name, new_id.replace('.jpg', '')])

print(f"✅ 软链完成 -> {INPUT_DIR}")
print(f"✅ resolution csv -> {RESOLUTION_CSV}")
print(f"✅ 映射表 csv     -> {MAPPING_CSV}")
