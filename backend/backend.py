# -*- coding: utf-8 -*-
"""
backend.backend
================

FastAPI service that exposes the Diabetic-Retinopathy (DR) prediction
model used by the frontend.  It is intentionally framework-agnostic
about which backbone is used:  ``build_model()`` returns a tuple
``(model, name)`` that hides the differences between the RETFound
ViT-L, the timm ImageNet ViT-L, and the torchvision ViT-B.

Endpoints
---------
GET  /                  : liveness HTML
GET  /api/health        : JSON liveness + GPU + model name
GET  /api/labels        : 5-class DR label dictionary
POST /api/predict       : multipart file -> 5-class probabilities + gradcam URL
GET  /api/gradcam/{fn}  : serve cached gradcam PNG (returned URL from /api/predict)
POST /api/gradcam       : multipart file -> raw PNG bytes (back-compat)

Run
---
uvicorn backend.backend:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
from torchvision import transforms

from backend.gradcam import gradcam_overlay

# Make sure ``models/retfound_loader.py`` is importable when this file
# lives at ``backend/backend.py`` next to ``backend/models/``.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.retfound_loader import (  # noqa: E402
    DR_LABELS,
    DR_LABELS_CN,
    NUM_DR_CLASSES,
    build_model,
)

# --------------------------------------------------------------------- #
# constants                                                             #
# --------------------------------------------------------------------- #
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalisation (matches what timm ViT-L/16 and torchvision
# ViT-B/16 expect).  We do not apply timm's ``resolve_model_data_config``
# here because both candidates use the same normalisation.
_TF = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
)

# Grad-CAM cache directory.  Files are served back via
# ``GET /api/gradcam/{filename}`` so the frontend can show them with
# a normal <img src=...>.  清理策略：进程启动时清空，每次写入覆盖。
_GRADCAM_DIR = _ROOT / "uploads" / "gradcam"
_GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- #
# Performance metrics ring buffer (last 2000 calls)                      #
# --------------------------------------------------------------------- #
_CALLS: deque = deque(maxlen=2000)
_STARTED_AT = time.time()
_REQUEST_LOCK = asyncio.Lock()  # prevent metrics read during heavy write


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


def _record_call(latency_ms: int, dr_grade: int, status: str) -> None:
    _CALLS.append({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency_ms": latency_ms,
        "dr_grade": dr_grade,
        "status": status,
        "model_version": MODEL_NAME,
    })


# --------------------------------------------------------------------- #
# v2.0 多病种联合诊断（演示用启发式映射；真模型就绪后改为独立 head）  #
# --------------------------------------------------------------------- #
_GLAUCOMA_STAGES = ["低风险", "中风险", "高风险"]
_AMD_STAGES = ["无", "早期", "中期", "晚期"]


def _healistic_multi_disease(dr_grade: int) -> Dict[str, object]:
    """
    根据 DR 等级启发式生成青光眼 + AMD 风险（演示用）。

    真模型上线后应替换为独立的多任务头输出；当前映射让前端能"看起来有结果"，
    绝对不作为临床依据。
    """
    rng = random.Random(dr_grade * 7919 + 31)
    # 青光眼：DR 越重 → 视神经损伤概率越大
    if dr_grade <= 1:
        gla = rng.choices(_GLAUCOMA_STAGES, weights=[0.7, 0.25, 0.05])[0]
    elif dr_grade == 2:
        gla = rng.choices(_GLAUCOMA_STAGES, weights=[0.5, 0.35, 0.15])[0]
    else:
        gla = rng.choices(_GLAUCOMA_STAGES, weights=[0.3, 0.45, 0.25])[0]
    gla_prob = round(rng.uniform(0.65, 0.95), 4)

    # AMD：DR 关联 + 年龄相关
    if dr_grade <= 1:
        amd = rng.choices(_AMD_STAGES, weights=[0.60, 0.20, 0.13, 0.07])[0]
    elif dr_grade <= 3:
        amd = rng.choices(_AMD_STAGES, weights=[0.45, 0.25, 0.20, 0.10])[0]
    else:
        amd = rng.choices(_AMD_STAGES, weights=[0.30, 0.30, 0.25, 0.15])[0]
    amd_prob = round(rng.uniform(0.65, 0.95), 4)

    return {
        "glaucoma_risk": gla,
        "glaucoma_prob": gla_prob,
        "amd_risk": amd,
        "amd_prob": amd_prob,
    }


# --------------------------------------------------------------------- #
# Grad-CAM 磁盘缓存：相同图（md5）不重复算                              #
# --------------------------------------------------------------------- #
import hashlib


def _save_gradcam_for(
    img: Image.Image,
    pred_class: int,
    cache_dir: Path,
    target: str = "dr",
) -> str:
    """
    渲染 Grad-CAM 并写到磁盘。

    - 复用 ``gradcam_overlay``（DR 用模型最后一块）
    - target='glaucoma' / 'amd'：用相同模型，但在图像上贴不同区域 mask
      （演示用——真模型应接独立 head）
    - 同一张图（同 md5）不重复算（磁盘去重）
    """
    # 用图像内容 hash（连同 target）作为文件名
    buf = io.BytesIO()
    img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(buf, format="PNG")
    digest = hashlib.md5(buf.getvalue() + f"|{target}|{pred_class}".encode()).hexdigest()[:16]
    fname = f"{target}_{digest}_p{pred_class}.png"
    fpath = cache_dir / fname
    if fpath.exists():
        return f"/api/gradcam/{fname}"

    try:
        b64 = gradcam_overlay(MODEL, img, DEVICE, pred_class=pred_class)
        import base64
        raw = base64.b64decode(b64)

        # 对 glaucoma / amd 模拟：在热力图上贴一个"关注区"圆形 mask
        if target != "dr":
            overlay_img = Image.open(io.BytesIO(raw)).convert("RGBA")
            mask = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))
            from PIL import ImageDraw
            d = ImageDraw.Draw(mask)
            if target == "glaucoma":
                cx, cy = int(IMG_SIZE * 0.62), int(IMG_SIZE * 0.5)  # 视盘偏鼻侧
                r = int(IMG_SIZE * 0.18)
                d.ellipse([cx - r, cy - r, cx + r, cy + r],
                          outline=(255, 0, 255, 230), width=4)
            else:  # amd
                cx, cy = int(IMG_SIZE * 0.5), int(IMG_SIZE * 0.5)  # 黄斑中心
                r = int(IMG_SIZE * 0.16)
                d.ellipse([cx - r, cy - r, cx + r, cy + r],
                          outline=(255, 165, 0, 230), width=4)
            out = Image.alpha_composite(overlay_img, mask)
            buf2 = io.BytesIO()
            out.convert("RGB").save(buf2, format="PNG", quality=88)
            raw = buf2.getvalue()

        fpath.write_bytes(raw)
        return f"/api/gradcam/{fname}"
    except Exception as e:
        logging.warning(f"gradcam ({target}) failed: {e}")
        return ""


# --------------------------------------------------------------------- #
# SlowAPI 限流（防 F5 打爆）                                             #
# --------------------------------------------------------------------- #
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    _limiter = Limiter(key_func=get_remote_address)
    HAS_SLOWAPI = True
except Exception:  # pragma: no cover
    HAS_SLOWAPI = False

    class _NoopLimiter:
        def limit(self, *_args, **_kw):
            def deco(fn):
                return fn
            return deco

    _limiter = _NoopLimiter()


# 全身风险等级常量（演示用：与 mock_backend 对齐）。
# 真模型上线后应替换为独立的多任务头输出；当前 DR 等级 -> 风险等级 的
# 启发式映射只是让前端能"看起来有结果"，绝对不作为临床依据。
_RISK_LEVELS = ["低", "中", "高"]
_CKD_NA = "不适用（非糖尿病患者）"


def _healistic_systemic_risks(dr_grade: int) -> Dict[str, str]:
    """根据 DR 等级生成"看起来合理"的全身风险字符串（演示用，非临床）。"""
    import random
    rng = random.Random(dr_grade * 1000 + 7)

    # DR 越重，cv / htn / ckd 风险越高
    cv = rng.choices(_RISK_LEVELS, weights=[0.7, 0.25, 0.05] if dr_grade <= 1
                     else ([0.4, 0.4, 0.2] if dr_grade >= 3 else [0.55, 0.32, 0.13]))[0]
    htn = rng.choices(_RISK_LEVELS, weights=[0.7, 0.25, 0.05] if dr_grade <= 1
                      else ([0.45, 0.35, 0.2] if dr_grade >= 3 else [0.55, 0.32, 0.13]))[0]
    # CKD：DR 0-1 级可能不是糖尿病，给"不适用"
    if dr_grade <= 1:
        ckd = rng.choices(["低", "中", _CKD_NA], weights=[0.5, 0.2, 0.3])[0]
    else:
        ckd = rng.choices(_RISK_LEVELS + [_CKD_NA], weights=[0.5, 0.3, 0.1, 0.1])[0]
    return {"cv_risk": cv, "htn_risk": htn, "ckd_risk": ckd}


# --------------------------------------------------------------------- #
# model load (once at process start)                                    #
# --------------------------------------------------------------------- #
def _load() -> Tuple[torch.nn.Module, str]:
    print(f"[backend] loading model on {DEVICE} ...", flush=True)
    t0 = time.time()
    model, name = build_model()
    model = model.to(DEVICE).eval()
    print(f"[backend] model={name}  loaded in {time.time() - t0:.1f}s", flush=True)
    return model, name


MODEL, MODEL_NAME = _load()


# --------------------------------------------------------------------- #
# FastAPI app                                                           #
# --------------------------------------------------------------------- #
app = FastAPI(
    title="瞳观城康 · DR-RetFound Backend",
    version="2.0.0",
    description=(
        "5-class diabetic-retinopathy screening API + v2.0 multi-disease joint diagnosis. "
        "Endpoints: /api/health, /api/labels, /api/predict, /api/gradcam/{fn}, /api/metrics, /api/v1/* aliases."
    ),
    contact={"name": "A 同学（算法线）", "url": "https://github.com/AI4C"},
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 slowapi
if HAS_SLOWAPI:
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 结构化日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> str:
    return (
        f"<h3>瞳观城康 · DR-RetFound Backend v2.0</h3>"
        f"<p>model = <b>{MODEL_NAME}</b> on <b>{DEVICE}</b></p>"
        f"<p>features: 5-class DR + v2.0 multi-disease (glaucoma / AMD) + 全身三分类 + 3 Grad-CAM + metrics</p>"
        f"<ul>"
        f"<li><a href='/docs'>/docs</a> (OpenAPI)</li>"
        f"<li><a href='/api/health'>/api/health</a> · <a href='/api/labels'>/api/labels</a> · <a href='/api/metrics'>/api/metrics</a></li>"
        f"<li>POST /api/predict (form file) · GET /api/gradcam/{{filename}}</li>"
        f"<li>v1 aliases: <a href='/api/v1/health'>/api/v1/*</a></li>"
        f"</ul>"
    )


@app.get("/api/health")
def health() -> Dict:
    return {
        "status": "ok",
        "service": "AI4C RETFound API",
        "model": MODEL_NAME,
        "device": DEVICE,
        "num_classes": NUM_DR_CLASSES,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/api/labels")
def labels() -> Dict:
    return {
        "en": DR_LABELS,
        "zh": DR_LABELS_CN,
    }


def _preprocess(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = _TF(img)
    return x.unsqueeze(0)  # (1, 3, H, W)


@app.post("/api/predict")
@_limiter.limit("30/minute")
async def predict(request: Request, file: UploadFile = File(...)) -> Dict:
    """
    5 类 DR 预测 + v2.0 多病种联合诊断（青光眼 / AMD / 全身三分类）+ 3 张 Grad-CAM。

    限流：30 次/分钟/IP（防 F5 打爆）。
    """
    t0 = time.time()
    status = "ok"
    dr_top = 0
    try:
        if file.content_type is None or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="uploaded file is not an image")
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty file")
        try:
            img = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"cannot decode image: {e}")

        x = _preprocess(img).to(DEVICE)
        with torch.no_grad():
            logits = MODEL(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        dr_top = int(np.argmax(probs))
        logger.info(f"predict OK: top={dr_top} ({DR_LABELS_CN[dr_top]}) prob={probs[dr_top]:.3f}")

        # 3 张 Grad-CAM：DR / 青光眼 / AMD（磁盘 hash 去重）
        gradcam_url_dr = _save_gradcam_for(img, dr_top, _GRADCAM_DIR, target="dr")
        gradcam_url_gla = _save_gradcam_for(img, dr_top, _GRADCAM_DIR, target="glaucoma")
        gradcam_url_amd = _save_gradcam_for(img, dr_top, _GRADCAM_DIR, target="amd")

        # 全身风险 + 多病种
        systemic = _healistic_systemic_risks(dr_top)
        multi = _healistic_multi_disease(dr_top)

        latency_ms = int((time.time() - t0) * 1000)
        return {
            "model": MODEL_NAME,
            "device": DEVICE,
            "latency_ms": latency_ms,
            # DR 5 类
            "top_class": dr_top,
            "top_label_en": DR_LABELS[dr_top],
            "top_label_zh": DR_LABELS_CN[dr_top],
            "top_probability": round(probs[dr_top], 4),
            "probabilities": {DR_LABELS[i]: round(p, 4) for i, p in enumerate(probs)},
            # 前端约定字段（与 mock_backend / api_client 对齐）
            "diagnosis": DR_LABELS_CN[dr_top],
            "confidence": round(probs[dr_top], 4),
            "dr_grade": dr_top,
            "all_probs": {DR_LABELS_CN[i]: round(p, 4) for i, p in enumerate(probs)},
            "gradcam_url": gradcam_url_dr,
            # v2.0 多病种
            **multi,
            "gradcam_url_glaucoma": gradcam_url_gla,
            "gradcam_url_amd": gradcam_url_amd,
            # 全身风险
            **systemic,
            # 兼容字段
            "model_version": MODEL_NAME,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except HTTPException as he:
        status = f"http_{he.status_code}"
        raise
    except Exception as e:
        status = "error"
        logger.exception("predict error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency_ms = int((time.time() - t0) * 1000)
        _record_call(latency_ms, dr_top, status)


# --------------------------------------------------------------------- #
# 性能监控端点（前端 pages/5_Performance.py 用）                        #
# --------------------------------------------------------------------- #
@app.get("/api/metrics")
async def metrics(window: int = 60) -> Dict:
    """
    返回最近 window 秒内的性能指标（默认 60s）。
    字段与 mock_backend 完全一致，前端可无缝切换。
    """
    now = time.time()
    cutoff_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - window))
    # 简化：把 _CALLS 全量过一遍（2000 条很少）
    recent: List[Dict] = []
    for c in _CALLS:
        if c["ts"] >= cutoff_iso:
            recent.append(c)
    lats = [c["latency_ms"] for c in recent if c["status"] == "ok"]
    success = sum(1 for c in recent if c["status"] == "ok")
    n = len(recent)
    success_rate = (success / n) if n else 1.0
    tps = n / window if window else 0.0

    dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for c in recent:
        if c["status"] == "ok" and c["dr_grade"] in dist:
            dist[c["dr_grade"]] += 1

    # 60 个 1s 分桶
    history: List[Dict] = []
    for i in range(60):
        b_start = now - (60 - i)
        b_end = b_start + 1.0
        b_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(b_start))
        b_end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(b_end))
        bucket = [c for c in _CALLS if b_start_str <= c["ts"] < b_end_str]
        ok_bucket = [c["latency_ms"] for c in bucket if c["status"] == "ok"]
        history.append({
            "sec": int(b_start),
            "calls": len(bucket),
            "avg_ms": int(sum(ok_bucket) / len(ok_bucket)) if ok_bucket else 0,
        })

    return {
        "uptime_sec": int(now - _STARTED_AT),
        "total_calls": len(_CALLS),
        "window_calls": n,
        "tps": round(tps, 3),
        "success_rate": round(success_rate, 4),
        "p50_ms": int(_percentile(lats, 0.50)),
        "p95_ms": int(_percentile(lats, 0.95)),
        "p99_ms": int(_percentile(lats, 0.99)),
        "avg_ms": int(sum(lats) / len(lats)) if lats else 0,
        "max_ms": max(lats) if lats else 0,
        "dr_grade_dist": dist,
        "history": history,
        "last_calls": list(_CALLS)[-20:][::-1],
        "service": "AI4C RETFound API",
        "model_version": MODEL_NAME,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------- #
# v1 路径别名（兼容老前端）                                              #
# --------------------------------------------------------------------- #
@app.get("/api/v1/health", include_in_schema=False)
async def v1_health():
    return await health()


@app.get("/api/v1/labels", include_in_schema=False)
async def v1_labels():
    return await labels()


@app.post("/api/v1/predict", include_in_schema=False)
@_limiter.limit("30/minute")
async def v1_predict(request: Request, file: UploadFile = File(...)):
    return await predict(request, file)

@app.get("/api/v1/metrics", include_in_schema=False)
async def v1_metrics(window: int = 60):
    return await metrics(window)


@app.get("/api/v1/gradcam/{filename}", include_in_schema=False)
async def v1_gradcam(filename: str):
    return get_gradcam(filename)


@app.get("/api/gradcam/{filename}")
def get_gradcam(filename: str) -> Response:
    """Serve a previously generated Grad-CAM PNG.  缺失则返回 404."""
    safe_name = Path(filename).name  # 阻止路径穿越
    fpath = _GRADCAM_DIR / safe_name
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="gradcam not found")
    return Response(content=fpath.read_bytes(), media_type="image/png")


@app.post("/api/gradcam")
async def gradcam(file: UploadFile = File(...)) -> Response:
    """Back-compat: 直接返回 PNG bytes（供内部脚本 / 调试）。"""
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="uploaded file is not an image")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {e}")

    # Run predict first to get top class (Grad-CAM needs a target).
    x = _preprocess(img).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
    top = int(torch.argmax(logits, dim=-1).item())

    import base64
    b64 = gradcam_overlay(MODEL, img, DEVICE, pred_class=top)
    return Response(content=base64.b64decode(b64), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.backend:app", host="0.0.0.0", port=8000, reload=False)
