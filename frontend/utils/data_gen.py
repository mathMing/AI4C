# -*- coding: utf-8 -*-
"""
data_gen.py
===========
区域驾驶舱用的模拟数据生成器。

特性：
- 10 个北京风格社区
- 随机但稳定的种子
- 包含年龄、DR 等级、心血管/高血压/慢性肾病风险
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

COMMUNITIES = [
    "和平社区",
    "朝阳社区",
    "海淀社区",
    "西城社区",
    "东城社区",
    "丰台社区",
    "石景山社区",
    "通州社区",
    "昌平社区",
    "大兴社区",
]

CV_RISK_LEVELS = ["低", "中", "高"]
HTN_RISK_LEVELS = ["低", "中", "高"]
CKD_RISK_LEVELS = ["低", "中", "高", "不适用（非糖尿病患者）"]


def generate_region_data(
    communities: Optional[list] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """生成区域模拟数据。"""
    rng = np.random.default_rng(seed)
    if communities is None:
        communities = COMMUNITIES

    rows = []
    for comm in communities:
        n = int(rng.integers(50, 200))
        for _ in range(n):
            dr = int(rng.choice([0, 1, 2, 3, 4], p=[0.5, 0.2, 0.15, 0.1, 0.05]))
            # DR 等级越高，全身风险概率越大
            cv_p = [0.7, 0.25, 0.05] if dr <= 1 else ([0.4, 0.4, 0.2] if dr >= 3 else [0.55, 0.32, 0.13])
            htn_p = [0.7, 0.25, 0.05] if dr <= 1 else ([0.45, 0.35, 0.2] if dr >= 3 else [0.55, 0.32, 0.13])
            cv = rng.choice(CV_RISK_LEVELS, p=cv_p)
            htn = rng.choice(HTN_RISK_LEVELS, p=htn_p)
            ckd = (
                rng.choice(CKD_RISK_LEVELS, p=[0.5, 0.3, 0.1, 0.1])
                if dr >= 2
                else rng.choice(["低", "中", "不适用（非糖尿病患者）"], p=[0.5, 0.2, 0.3])
            )
            rows.append(
                {
                    "community": comm,
                    "dr_grade": dr,
                    "cv_risk": str(cv),
                    "htn_risk": str(htn),
                    "ckd_risk": str(ckd),
                    "age": int(rng.integers(30, 80)),
                    "patient_id": f"{comm[0]}{rng.integers(1000, 9999)}",
                }
            )
    return pd.DataFrame(rows)