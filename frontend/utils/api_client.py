# -*- coding: utf-8 -*-
"""
api_client.py
=============
与 A 同学模型 API 的统一通信层。

约定字段（A同学 /api/predict 返回）：
    diagnosis, confidence, dr_grade, gradcam_url, all_probs,
    cv_risk, htn_risk, ckd_risk, model_version, timestamp

支持的 API_URL 配置（按优先级）：
    1. 环境变量 TONGGUAN_API_URL
    2. .streamlit/secrets.toml 中的 [api] url
    3. 默认 http://localhost:8000
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests

DEFAULT_TIMEOUT = 30


def get_api_url() -> str:
    """获取模型API基础URL。"""
    # 1) 环境变量
    env = os.environ.get("TONGGUAN_API_URL")
    if env:
        return env.rstrip("/")
    # 2) streamlit secrets
    try:
        import streamlit as st

        if "api" in st.secrets and "url" in st.secrets["api"]:
            return st.secrets["api"]["url"].rstrip("/")
    except Exception:
        # 没有 secrets 文件或读取失败
        pass
    # 3) 默认
    return "http://localhost:8000"


def health_check(timeout: float = 5.0) -> Dict[str, Any]:
    """健康检查，失败返回 {ok: False, error: ...}。"""
    try:
        r = requests.get(f"{get_api_url()}/api/health", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            data["ok"] = True
            return data
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def call_predict(
    image_bytes: bytes,
    filename: str = "fundus.jpg",
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    调模型API，返回归一化后的结果字典。

    关键点：
        - 即使后端字段命名风格不同（snake / camel），都尽量兼容
        - 把 all_probs 拆出来成 probs_dict，方便前端 DataFrame 显示
    """
    url = f"{get_api_url()}/api/predict"
    files = {"file": (filename, image_bytes, "image/jpeg")}
    r = requests.post(url, files=files, timeout=timeout)
    r.raise_for_status()
    raw = r.json()

    return _normalize(raw)


def _normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    """兼容多种字段命名。"""
    out = dict(raw)
    # all_probs → probs_dict
    if "probs_dict" not in out:
        out["probs_dict"] = out.get("all_probs") or out.get("probs") or {}
    # 兜底：DR 主诊断
    out.setdefault("diagnosis", "—")
    out.setdefault("confidence", 0.0)
    out.setdefault("dr_grade", 0)
    out.setdefault("cv_risk", "—")
    out.setdefault("htn_risk", "—")
    out.setdefault("ckd_risk", "—")
    out.setdefault("gradcam_url", "")
    out.setdefault("model_version", "—")
    # 多病种联合诊断（v2.0）：青光眼 + AMD
    out.setdefault("glaucoma_risk", "低风险")
    out.setdefault("glaucoma_prob", 0.0)
    out.setdefault("amd_risk", "无")
    out.setdefault("amd_prob", 0.0)
    out.setdefault("gradcam_url_glaucoma", "")
    out.setdefault("gradcam_url_amd", "")
    return out


def get_gradcam_url(result: Dict[str, Any]) -> Optional[str]:
    """根据返回字典补全完整的 gradcam URL。"""
    url = result.get("gradcam_url") or ""
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"{get_api_url()}{url}"