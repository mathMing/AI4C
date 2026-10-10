# -*- coding: utf-8 -*-
"""
performance.py
==============
性能监控前端工具：

    fetch_metrics() : 调后端 /api/metrics，返回 dict（带兜底）
    metrics_to_dataframe() : history 段转 pandas DataFrame
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import requests

from utils.api_client import get_api_url


# 离线/无后端时的兜底数据
_FALLBACK = {
    "uptime_sec": 0,
    "total_calls": 0,
    "window_calls": 0,
    "tps": 0.0,
    "success_rate": 0.0,
    "p50_ms": 0,
    "p95_ms": 0,
    "p99_ms": 0,
    "avg_ms": 0,
    "max_ms": 0,
    "dr_grade_dist": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    "history": [
        {"sec": 0, "calls": 0, "avg_ms": 0} for _ in range(60)
    ],
    "last_calls": [],
    "service": "—",
    "ts": "—",
    "_offline": True,
}


def fetch_metrics(timeout: float = 3.0) -> Dict[str, Any]:
    """
    抓后端 /api/metrics。失败返回 _FALLBACK（带 _offline=True）。
    """
    try:
        r = requests.get(f"{get_api_url()}/api/metrics", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        data.setdefault("history", [])
        data.setdefault("last_calls", [])
        data.setdefault("dr_grade_dist", {})
        data["_offline"] = False
        return data
    except Exception as e:
        out = dict(_FALLBACK)
        out["_error"] = str(e)
        return out


def metrics_to_dataframe(history: list) -> pd.DataFrame:
    """把 history 段（60 个分桶）转成 DataFrame。"""
    if not history:
        return pd.DataFrame(columns=["sec", "calls", "avg_ms"])
    df = pd.DataFrame(history)
    if "sec" in df.columns:
        # 转成 mm:ss 字符串
        df["time"] = pd.to_datetime(df["sec"], unit="s").dt.strftime("%H:%M:%S")
    return df


def last_calls_to_dataframe(last_calls: list) -> pd.DataFrame:
    """把 last_calls 转成 DataFrame。"""
    if not last_calls:
        return pd.DataFrame(columns=["ts", "latency_ms", "dr_grade", "status", "model_version"])
    return pd.DataFrame(last_calls)


def grade_label(g: int) -> str:
    return {0: "无 DR", 1: "轻度 NPDR", 2: "中度 NPDR", 3: "重度 NPDR", 4: "增殖期"}.get(int(g), f"DR{g}")


def grade_color(g: int) -> str:
    return {0: "#16a34a", 1: "#84cc16", 2: "#d97706", 3: "#ea580c", 4: "#dc2626"}.get(int(g), "#94a3b8")
