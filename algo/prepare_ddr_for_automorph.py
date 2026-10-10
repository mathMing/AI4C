"""
准备 DDR 测试集给 AutoMorph
- 把 DDR 测试图软链到 AutoMorph/input/eyepacs/00001.jpg 格式
- 写 resolution_information.csv（fundus, res）
- 写 name_mapping.csv（原文件名 → 数字ID）

用法: python prepare_ddr_for_automorph.py
"""
import os, shutil, csv
from pathlib import Path

DDR_TEST = Path("/root/autodl-tmp/AI4C/algo/data/DDR-dataset/DR_grading/test")
AUTOMORPH = Path("/root/autodl-tmp/AI4C/AutoMorph-main")
INPUT_DIR = AUTOMORPH / "input" / "eyepacs"
RESOLUTION_CSV = AUTOMORPH / "resolution_information.csv"
MAPPING_CSV = AUTOMORPH / "name_mapping.csv"

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 找所有 jpg
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
            wr_r.writerow([new_id, '1'])  # res=1 (粗略估计，需要的话后续调)
            wr_m.writerow([jp.name, new_id.replace('.jpg', '')])

    print(f"? 软链完成 -> {INPUT_DIR}")
    print(f"? resolution csv -> {RESOLUTION_CSV}")
    print(f"? 映射表 csv     -> {MAPPING_CSV}")
    print(f"\n前5个示例:")
    for j in jpgs[:5]:
        idx = jpgs.index(j)
        print(f"  {j.name}  ->  {idx+1:05d}.jpg")

if __name__ == "__main__":
    main()
