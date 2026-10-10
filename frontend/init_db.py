# -*- coding: utf-8 -*-
"""
init_db.py
==========
运行一次即可：建表 + 插入演示数据。

用法：
    cd frontend_B
    python init_db.py            # 已有数据则跳过
    python init_db.py --force    # 强制重置演示数据（清空后再插入）
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.db import DB_DIR, DB_PATH, ensure_db, seed_demo_data


def reset_db() -> None:
    """删除旧数据库文件并重建。"""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"🗑 已删除旧数据库：{DB_PATH}")
    ensure_db()
    print("✅ 已重建表结构")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重置演示数据")
    args = parser.parse_args()

    os.makedirs(DB_DIR, exist_ok=True)

    if args.force:
        reset_db()
        n = seed_demo_data(force=True)
        print(f"✅ 强制插入 {n} 条演示记录")
    else:
        ensure_db()
        n = seed_demo_data(force=False)
        if n > 0:
            print(f"✅ 首次运行，已插入 {n} 条演示记录")
        else:
            print(f"ℹ 数据库已就绪：{DB_PATH}（无需再插入演示数据）")

    print(f"📁 数据库位置：{DB_PATH}")


if __name__ == "__main__":
    main()