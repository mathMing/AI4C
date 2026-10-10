# -*- coding: utf-8 -*-
"""
db.py
=====
SQLite 患者随访数据库工具层。

数据库文件默认在 data/patient_records.db，第一次调用 ensure_db() 时建表。

表结构：
    screenings (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT,
        patient_name    TEXT,
        age             INTEGER,
        gender          TEXT,
        diabetes        INTEGER (0/1),
        screening_date  TEXT (YYYY-MM-DD),
        diagnosis       TEXT,
        dr_grade        INTEGER (0-4),
        cv_risk         TEXT,
        htn_risk        TEXT,
        ckd_risk        TEXT,
        confidence      REAL,
        gradcam_path    TEXT
    )
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "patient_records.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db() -> None:
    """幂等地创建表结构。v2.0 新增多病种联合诊断字段。"""
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_name TEXT,
                age INTEGER,
                gender TEXT,
                diabetes INTEGER DEFAULT 0,
                screening_date TEXT NOT NULL,
                diagnosis TEXT,
                dr_grade INTEGER DEFAULT 0,
                cv_risk TEXT,
                htn_risk TEXT,
                ckd_risk TEXT,
                confidence REAL DEFAULT 0.0,
                gradcam_path TEXT,
                glaucoma_risk TEXT DEFAULT '低风险',
                glaucoma_prob REAL DEFAULT 0.0,
                amd_risk TEXT DEFAULT '无',
                amd_prob REAL DEFAULT 0.0
            )
            """
        )
        # 向后兼容：老库（v1.0）若没有多病种字段，补上
        for col, ddl in [
            ("glaucoma_risk", "TEXT DEFAULT '低风险'"),
            ("glaucoma_prob", "REAL DEFAULT 0.0"),
            ("amd_risk", "TEXT DEFAULT '无'"),
            ("amd_prob", "REAL DEFAULT 0.0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE screenings ADD COLUMN {col} {ddl}")
            except Exception:
                # 列已存在（SQLite 报 duplicate column）—— 跳过
                pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_screenings_pid ON screenings(patient_id)"
        )
        conn.commit()


def add_screening(
    patient_id: str,
    patient_name: str,
    dr_grade: int,
    cv_risk: str,
    htn_risk: str,
    ckd_risk: str,
    diagnosis: str,
    confidence: float,
    gradcam_path: str = "",
    age: int = 0,
    gender: str = "—",
    diabetes: bool = False,
    screening_date: Optional[str] = None,
    glaucoma_risk: str = "低风险",
    glaucoma_prob: float = 0.0,
    amd_risk: str = "无",
    amd_prob: float = 0.0,
) -> int:
    """插入一条筛查记录，返回新记录 id。v2.0 含多病种联合诊断字段。"""
    if screening_date is None:
        screening_date = datetime.now().strftime("%Y-%m-%d")
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO screenings
            (patient_id, patient_name, age, gender, diabetes,
             screening_date, diagnosis, dr_grade,
             cv_risk, htn_risk, ckd_risk, confidence, gradcam_path,
             glaucoma_risk, glaucoma_prob, amd_risk, amd_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                patient_name,
                age,
                gender,
                int(bool(diabetes)),
                screening_date,
                diagnosis,
                int(dr_grade),
                cv_risk,
                htn_risk,
                ckd_risk,
                float(confidence),
                gradcam_path,
                glaucoma_risk,
                float(glaucoma_prob),
                amd_risk,
                float(amd_prob),
            ),
        )
        conn.commit()
        return cur.lastrowid or 0


def fetch_patient_history(patient_id: str) -> "Any":
    """按时间顺序返回该患者所有记录（pandas DataFrame）。"""
    import pandas as pd

    ensure_db()
    with _conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM screenings WHERE patient_id = ? ORDER BY screening_date",
            conn,
            params=(patient_id,),
        )
    return df


def fetch_all_screenings() -> "Any":
    """返回全表（pandas DataFrame）。"""
    import pandas as pd

    ensure_db()
    with _conn() as conn:
        df = pd.read_sql_query("SELECT * FROM screenings ORDER BY screening_date DESC", conn)
    return df


def list_patient_ids() -> List[str]:
    """返回去重后的患者ID列表，按最近一次筛查倒序。"""
    ensure_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT patient_id FROM screenings "
            "ORDER BY patient_id"
        ).fetchall()
    return [r["patient_id"] for r in rows]


def seed_demo_data(force: bool = False) -> int:
    """
    初始化几条演示数据，避免首次启动页面空荡荡。
    返回插入条数。
    """
    import random
    from datetime import datetime, timedelta

    ensure_db()
    if not force and list_patient_ids():
        return 0

    base = datetime(2025, 1, 1)
    seed = [
        # (patient_id, name, age, gender, diabetes, grade, cv, htn, ckd, day_offset)
        ("P001", "张三", 62, "男", True, 0, "低", "低", "低", 0),
        ("P001", "张三", 62, "男", True, 1, "低", "低", "低", 90),
        ("P001", "张三", 62, "男", True, 2, "中", "中", "中", 180),
        ("P001", "张三", 62, "男", True, 2, "中", "中", "中", 270),
        ("P002", "李四", 55, "女", False, 1, "低", "中", "不适用（非糖尿病患者）", 30),
        ("P002", "李四", 55, "女", False, 0, "低", "低", "不适用（非糖尿病患者）", 150),
        ("P003", "王五", 71, "男", True, 3, "高", "高", "高", 60),
        ("P003", "王五", 71, "男", True, 4, "高", "高", "高", 200),
        ("P004", "赵六", 48, "女", False, 0, "低", "低", "不适用（非糖尿病患者）", 45),
        ("P005", "钱七", 67, "男", True, 2, "中", "中", "中", 75),
    ]
    n = 0
    for pid, name, age, gender, dia, g, cv, htn, ckd, off in seed:
        date = (base + timedelta(days=off)).strftime("%Y-%m-%d")
        add_screening(
            patient_id=pid,
            patient_name=name,
            age=age,
            gender=gender,
            diabetes=dia,
            dr_grade=g,
            cv_risk=cv,
            htn_risk=htn,
            ckd_risk=ckd,
            diagnosis=f"DR {g} 级",
            confidence=round(random.uniform(0.78, 0.95), 4),
            gradcam_path="",
            screening_date=date,
        )
        n += 1
    return n