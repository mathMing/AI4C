# -*- coding: utf-8 -*-
"""
瞳观城康 · B同学（工程线）模拟后端
====================================

功能：
    在A同学（算法线）的真实模型API尚未就绪时，提供一个完全兼容的假API，
    让B同学可以先把前端、驾驶舱、随访全部跑通。约定接口与A同学最终交付的
    FastAPI服务完全一致（见：backend_A/main.py 中的 /api/predict），等A同学
    交付后，只需修改 .streamlit/secrets.toml 或前端侧 API_URL 即可对接。

接口约定（与A同学一致）：
    请求：POST /api/predict，form-data 字段名 file（图片文件）
    响应：
        {
            "diagnosis": "中度NPDR",          # 中文诊断文本
            "confidence": 0.8732,             # 置信度 0-1
            "dr_grade": 2,                    # DR分级 0-4
            "gradcam_url": "/api/gradcam/xxx.jpg",  # 热力图相对路径
            "all_probs": {                    # 各类别概率
                "无DR": 0.05, "轻度NPDR": 0.10, ...
            },
            "cv_risk": "中",   # 心血管风险（演示用）
            "htn_risk": "低", # 高血压风险
            "ckd_risk": "低", # 慢性肾病风险
            "model_version": "mock-1.0"  # 标识这是模拟结果
        }

启动：
    cd frontend_B
    uvicorn mock_backend:app --reload --port 8000
"""
from __future__ import annotations

import io
import os
import random
import uuid
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw

# ----------------------------- 全局配置 ----------------------------- #
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "mock_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============== 性能监控（轻量 in-memory 环形缓冲） ============== #
# 每次 /api/predict 完成后写入一条记录（ts / latency_ms / dr_grade / model_version / status）
# /api/metrics 汇总并返回
from collections import deque
import time as _time

_CALLS = deque(maxlen=2000)            # 最近 2000 次调用
_STARTED_AT = _time.time()              # 进程启动时间
_PROCESS_LATENCY_MS = 320              # 模拟模型本身推理耗时（不随请求变化）

# DR五分类
DR_LABELS = {
    0: "无DR",
    1: "轻度NPDR",
    2: "中度NPDR",
    3: "重度NPDR",
    4: "增殖期DR",
}

# 全身风险等级
CV_RISK_LEVELS = ["低", "中", "高"]
HTN_RISK_LEVELS = ["低", "中", "高"]
CKD_RISK_LEVELS = ["低", "中", "高", "不适用（非糖尿病患者）"]

# 多病种联合诊断（v2.0）
GLAUCOMA_LEVELS = ["低风险", "中风险", "高风险"]   # 青光眼三分类
AMD_STAGES = ["无", "早期", "中期", "晚期"]         # AMD 四期

app = FastAPI(title="瞳观城康 Mock API", version="mock-1.0")

# 允许跨域：A同学的代码也是 allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模拟不同切面分析，给热力图加点"看起来合理"的位置
np.random.seed(2025)


# ----------------------------- 工具函数 ----------------------------- #
def synthesize_gradcam(
    upload_filename: str,
    pred_class: int,
    target: str = "dr",
) -> str:
    """
    生成一张模拟的热力图。
    target:
        - "dr"      : DR 病灶位置（按 DR 分级在 4 象限之一）
        - "glaucoma": 青光眼视盘位置（中心偏一侧）
        - "amd"     : AMD 黄斑位置（中心）
    """
    size = 224
    img = Image.new("RGB", (size, size), color=(40, 40, 60))
    draw = ImageDraw.Draw(img)

    if target == "glaucoma":
        # 青光眼关注视盘（中心偏鼻侧）
        cx, cy = (140, 112)
    elif target == "amd":
        # AMD 关注黄斑中心凹
        cx, cy = (112, 112)
    else:
        centers = {
            0: (112, 112), 1: (80, 80), 2: (150, 90),
            3: (90, 150), 4: (140, 150),
        }
        cx, cy = centers.get(pred_class, (112, 112))
    for r, alpha in [(60, 200), (40, 220), (25, 240)]:
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=(255, 80, 30),
            outline=(255, 220, 80),
            width=2,
        )
    # 模拟Grad-CAM热度叠加：中心红、外圈黄
    arr = np.array(img).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    heat = np.clip(1.0 - d / 80.0, 0, 1)
    heat = (heat * 255).astype(np.uint8)
    heat_img = np.stack([heat, heat * 0.4, heat * 0.1], axis=-1)
    out = (arr * 0.3 + heat_img * 0.7).clip(0, 255).astype(np.uint8)

    save_name = f"mock_{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    Image.fromarray(out).save(save_path, quality=85)
    return save_name


def make_response(grade: int) -> dict:
    """根据 DR 分级产出完整的模拟返回（含多病种联合诊断）。"""
    confidence = round(random.uniform(0.72, 0.97), 4)
    # 让 all_probs 看起来合理：预测类概率最大，其他类小一些
    rest = round((1 - confidence) / 4, 4)
    probs = {DR_LABELS[i]: (confidence if i == grade else rest) for i in range(5)}

    # 全身风险：DR等级越高，全身风险概率越大
    cv_pool = (["低"] * 4 + ["中"] * 3 + ["高"] * max(1, grade + 1))
    htn_pool = (["低"] * 5 + ["中"] * 3 + ["高"] * max(1, grade))
    ckd_pool = (
        ["低"] * 5 + ["中"] * 3 + ["高"] * max(1, grade) if grade >= 2
        else ["低", "中", "不适用（非糖尿病患者）"]
    )

    # 多病种联合诊断（v2.0）
    # 青光眼：DR 等级越高 → 视神经损伤概率越大 → 中高风险概率上调
    glaucoma_pool = (
        ["低风险"] * (6 - grade) + ["中风险"] * 3 + ["高风险"] * max(1, grade // 2 + 1)
    )
    glaucoma = random.choice(glaucoma_pool)
    glaucoma_prob = round(random.uniform(0.65, 0.95), 4)

    # AMD：年龄相关，DR 等级关联（黄斑区微血管损伤）
    amd_stage = random.choices(
        [0, 1, 2, 3],
        weights=[0.60, 0.20, 0.13, 0.07] if grade <= 1
        else ([0.45, 0.25, 0.20, 0.10] if grade <= 3 else [0.30, 0.30, 0.25, 0.15]),
        k=1,
    )[0]
    amd_label = AMD_STAGES[amd_stage]
    amd_prob = round(random.uniform(0.65, 0.95), 4)

    return {
        "diagnosis": DR_LABELS[grade],
        "confidence": confidence,
        "dr_grade": grade,
        "all_probs": probs,
        "cv_risk": random.choice(cv_pool),
        "htn_risk": random.choice(htn_pool),
        "ckd_risk": random.choice(ckd_pool),
        # ---- 多病种联合诊断 v2.0 ----
        "glaucoma_risk": glaucoma,            # "低风险" / "中风险" / "高风险"
        "glaucoma_prob": glaucoma_prob,       # 0-1
        "amd_risk": amd_label,                # "无" / "早期" / "中期" / "晚期"
        "amd_prob": amd_prob,                 # 0-1
        "model_version": "mock-2.0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ----------------------------- 接口 ----------------------------- #
@app.get("/")
async def root():
    return {
        "service": "瞳观城康 Mock API",
        "version": "mock-1.0",
        "endpoints": [
            "POST /api/predict",
            "GET /api/gradcam/{filename}",
            "GET /api/health",
        ],
        "note": "此服务用于B同学前端并行开发，A同学真模型API就绪后请切换API_URL。",
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    接收眼底图像，返回与A同学完全一致的诊断JSON（v2.0 多病种联合诊断）。
    """
    t0 = _time.time()
    status = "ok"
    grade = 0
    model_version = "mock-2.0"

    contents = await file.read()
    if not contents:
        status = "bad_request"
        _record_call(t0, 0, status, 0)
        return JSONResponse({"error": "empty file"}, status_code=400)

    try:
        Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        status = "bad_image"
        _record_call(t0, 0, status, 0)
        return JSONResponse({"error": "invalid image"}, status_code=400)

    grade = random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.2, 0.15, 0.1, 0.05])[0]
    payload = make_response(grade)
    # 三个独立热力图：DR / 青光眼 / AMD
    payload["gradcam_url"] = f"/api/gradcam/{synthesize_gradcam(file.filename or 'upload.jpg', grade, 'dr')}"
    payload["gradcam_url_glaucoma"] = f"/api/gradcam/{synthesize_gradcam(file.filename or 'upload.jpg', grade, 'glaucoma')}"
    payload["gradcam_url_amd"] = f"/api/gradcam/{synthesize_gradcam(file.filename or 'upload.jpg', grade, 'amd')}"

    # 模拟推理耗时
    await _simulate_inference()
    _record_call(t0, grade, status, model_version)

    return JSONResponse(payload)


@app.get("/api/gradcam/{filename}")
async def get_gradcam(filename: str):
    """返回模拟的Grad-CAM热力图。"""
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        # 如果找不到，临时生成一张红色占位图
        img = Image.new("RGB", (224, 224), color=(200, 50, 50))
        img.save(path)
    return FileResponse(path)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "瞳观城康 Mock API", "ts": datetime.now().isoformat()}


# ====================================================================== #
# 性能监控端点
# ====================================================================== #
async def _simulate_inference() -> None:
    """模拟模型推理耗时（实际不会阻塞，用 sleep 让指标真实）。"""
    jitter = random.gauss(0, 60)
    ms = max(80, _PROCESS_LATENCY_MS + jitter)
    import asyncio
    await asyncio.sleep(ms / 1000.0)


def _record_call(t0: float, dr_grade: int, status: str, model_version) -> None:
    """写一条调用记录到环形缓冲。"""
    latency_ms = int((_time.time() - t0) * 1000)
    _CALLS.append({
        "ts": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "latency_ms": latency_ms,
        "dr_grade": dr_grade,
        "model_version": str(model_version),
        "status": status,
    })


def _percentile(xs, p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


@app.get("/api/metrics")
async def metrics(window: int = 60):
    """
    返回最近 window 秒内的性能指标。

    字段：
        uptime_sec                 : 进程启动时长
        total_calls                : 环形缓冲中总调用数
        window_calls               : 时间窗内调用数
        tps                        : 每秒事务数（最近 60s）
        success_rate               : 成功率 0-1
        p50_ms / p95_ms / p99_ms   : 延迟分位
        avg_ms / max_ms            : 平均 / 最大
        dr_grade_dist              : DR 0-4 等级分布
        history                    : 最近 60 个分桶（秒级）TPS + 平均延迟
        last_calls                 : 最近 20 次调用明细
    """
    now = _time.time()
    recent = [c for c in _CALLS if (now - datetime.fromisoformat(c["ts"]).timestamp()) <= window]
    lats = [c["latency_ms"] for c in recent if c["status"] == "ok"]
    success = sum(1 for c in recent if c["status"] == "ok")
    n = len(recent)
    success_rate = (success / n) if n else 1.0
    tps = n / window if window else 0.0

    # DR 等级分布
    dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for c in recent:
        if c["status"] == "ok" and c["dr_grade"] in dist:
            dist[c["dr_grade"]] += 1

    # 60 个 1s 分桶（最近 60 秒）
    history = []
    for i in range(60):
        b_start = now - (60 - i)
        b_end = b_start + 1.0
        bucket = [c for c in _CALLS
                  if b_start <= datetime.fromisoformat(c["ts"]).timestamp() < b_end]
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
        "service": "瞳观城康 Mock API",
        "ts": datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mock_backend:app", host="0.0.0.0", port=8000, reload=True)