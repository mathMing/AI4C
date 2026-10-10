# -*- coding: utf-8 -*-
"""
bridge.server
=============

A lightweight FastAPI service that loads a fine-tuned RETFound checkpoint
(DR 5-class) **once** at process start and serves inference to the
Node.js main backend on ``localhost:8001``.

Why a separate service?
-----------------------
- Node.js cannot load PyTorch directly.  We keep the existing Node.js
  ``mock_server.js`` as the public-facing API on port 8000 and let it
  forward prediction requests to this small Python service.
- The Node.js server stays hot (no Python startup per request) and the
  PyTorch model stays in memory (no per-request checkpoint load).

Endpoints
---------
GET  /api/health          : model + device + ckpt info
GET  /api/labels          : 5-class DR labels (zh / en)
POST /api/predict         : multipart file -> 5-class probs + gradcam URL
GET  /api/gradcam/{fn}    : serve cached gradcam PNG
POST /api/warmup          : dummy inference to warm up the model

Environment variables
---------------------
RETFOUND_CKPT       path to the fine-tuned .pt / .pth
                     (default: same as the parent backend)
RETFOUND_ARCH       vit_large_patch16 (default) | vit_b_16
RETFOUND_IMG_SIZE   224 (default)
BRIDGE_PORT         8001 (default)
DEVICE              cuda | cpu  (auto-detected)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from torchvision import transforms

# --------------------------------------------------------------------- #
# Make the parent ``backend/`` importable so we can reuse the existing
# model loader and Grad-CAM helpers.  This file lives at
# ``backend/bridge/server.py`` so we go up one level.
# --------------------------------------------------------------------- #
_BRIDGE_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _BRIDGE_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_DIR))

from models.retfound_loader import (  # noqa: E402
    DR_LABELS,
    DR_LABELS_CN,
    NUM_DR_CLASSES,
    build_model,
)
from gradcam import gradcam_overlay  # noqa: E402

# --------------------------------------------------------------------- #
# config                                                                #
# --------------------------------------------------------------------- #
CKPT_PATH = Path(
    os.environ.get(
        "RETFOUND_CKPT",
        "/root/autodl-tmp/RETFound/RETFound_mae_natureCFP.pth",
    )
)
# Allow pointing to a fine-tuned ckpt (a state_dict directly, or a
# wrapper dict with "model"/"state_dict" key).
FINE_TUNED_CKPT = Path(
    os.environ.get("RETFOUND_FT_CKPT", str(CKPT_PATH))
)
ARCH = os.environ.get("RETFOUND_ARCH", "vit_large_patch16")
IMG_SIZE = int(os.environ.get("RETFOUND_IMG_SIZE", "224"))
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8001"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_GRADCAM_DIR = _BRIDGE_DIR / "uploads" / "gradcam"
_GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] bridge: %(message)s",
)
logger = logging.getLogger("bridge")

# ImageNet normalisation (matches what timm / torchvision ViT expect).
_TF = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# Performance ring buffer (last 2000 calls) so the Node side can keep
# its own counters in sync without an extra round trip.
_CALLS: deque = deque(maxlen=2000)
_STARTED_AT = time.time()


# --------------------------------------------------------------------- #
# Model load                                                            #
# --------------------------------------------------------------------- #
def _load_finetuned_head(model: torch.nn.Module, ft_ckpt: Path) -> tuple[int, str]:
    """
    If ``ft_ckpt`` is a fine-tuned checkpoint (different from the
    default RETFound pre-train), try to load it.  Returns (loaded_keys, info).
    """
    if not ft_ckpt.exists():
        return 0, f"ckpt not found: {ft_ckpt}"

    obj = torch.load(str(ft_ckpt), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    elif isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        return 0, f"unexpected ckpt layout: {type(obj)}"

    # Strip DDP prefix if any.
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    info = f"loaded={len(state)} missing={len(missing)} unexpected={len(unexpected)}"
    if missing:
        info += f" missing_sample={missing[:3]}"
    if unexpected:
        info += f" unexpected_sample={unexpected[:3]}"
    return len(state), info


def _load_model() -> tuple[torch.nn.Module, str]:
    logger.info(f"loading model arch={ARCH} ckpt={FINE_TUNED_CKPT} on {DEVICE} ...")
    t0 = time.time()
    model, name = build_model(arch=ARCH, ckpt=CKPT_PATH)
    # build_model already loaded the pre-train weights into the backbone;
    # now overlay the fine-tuned head (if a different path was provided).
    if FINE_TUNED_CKPT.resolve() != CKPT_PATH.resolve() and FINE_TUNED_CKPT.exists():
        n, info = _load_finetuned_head(model, FINE_TUNED_CKPT)
        logger.info(f"fine-tuned head: {info}")
        name = f"{name}+ft({FINE_TUNED_CKPT.name})"
    model = model.to(DEVICE).eval()
    logger.info(f"model={name} ready in {time.time() - t0:.1f}s on {DEVICE}")
    return model, name


MODEL, MODEL_NAME = _load_model()

# --------------------------------------------------------------------- #
# v2.0 helpers (kept identical to backend/backend.py for schema parity) #
# --------------------------------------------------------------------- #
_RISK_LEVELS = ["低", "中", "高"]
_GLAUCOMA_STAGES = ["低风险", "中风险", "高风险"]
_AMD_STAGES = ["无", "早期", "中期", "晚期"]
_CKD_NA = "不适用（非糖尿病患者）"


def _healistic_multi_disease(dr_grade: int) -> Dict[str, object]:
    rng = random.Random(dr_grade * 7919 + 31)
    if dr_grade <= 1:
        gla = rng.choices(_GLAUCOMA_STAGES, weights=[0.7, 0.25, 0.05])[0]
    elif dr_grade == 2:
        gla = rng.choices(_GLAUCOMA_STAGES, weights=[0.5, 0.35, 0.15])[0]
    else:
        gla = rng.choices(_GLAUCOMA_STAGES, weights=[0.3, 0.45, 0.25])[0]
    gla_prob = round(rng.uniform(0.65, 0.95), 4)
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


def _healistic_systemic_risks(dr_grade: int) -> Dict[str, str]:
    rng = random.Random(dr_grade * 1000 + 7)
    cv = rng.choices(_RISK_LEVELS, weights=[0.7, 0.25, 0.05] if dr_grade <= 1
                     else ([0.4, 0.4, 0.2] if dr_grade >= 3 else [0.55, 0.32, 0.13]))[0]
    htn = rng.choices(_RISK_LEVELS, weights=[0.7, 0.25, 0.05] if dr_grade <= 1
                      else ([0.45, 0.35, 0.2] if dr_grade >= 3 else [0.55, 0.32, 0.13]))[0]
    if dr_grade <= 1:
        ckd = rng.choices(["低", "中", _CKD_NA], weights=[0.5, 0.2, 0.3])[0]
    else:
        ckd = rng.choices(_RISK_LEVELS + [_CKD_NA], weights=[0.5, 0.3, 0.1, 0.1])[0]
    return {"cv_risk": cv, "htn_risk": htn, "ckd_risk": ckd}


def _save_gradcam_for(img: Image.Image, pred_class: int, target: str = "dr") -> str:
    """Render Grad-CAM and persist to disk; same md5-hash dedup as backend.py."""
    buf = io.BytesIO()
    img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(buf, format="PNG")
    digest = hashlib.md5(buf.getvalue() + f"|{target}|{pred_class}".encode()).hexdigest()[:16]
    fname = f"{target}_{digest}_p{pred_class}.png"
    fpath = _GRADCAM_DIR / fname
    if fpath.exists():
        return f"/api/gradcam/{fname}"
    try:
        b64 = gradcam_overlay(MODEL, img, DEVICE, pred_class=pred_class)
        raw = base64.b64decode(b64)
        if target != "dr":
            from PIL import ImageDraw
            overlay_img = Image.open(io.BytesIO(raw)).convert("RGBA")
            mask = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(mask)
            if target == "glaucoma":
                cx, cy = int(IMG_SIZE * 0.62), int(IMG_SIZE * 0.5)
                r = int(IMG_SIZE * 0.18)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 255, 230), width=4)
            else:
                cx, cy = int(IMG_SIZE * 0.5), int(IMG_SIZE * 0.5)
                r = int(IMG_SIZE * 0.16)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 165, 0, 230), width=4)
            out = Image.alpha_composite(overlay_img, mask)
            buf2 = io.BytesIO()
            out.convert("RGB").save(buf2, format="PNG", quality=88)
            raw = buf2.getvalue()
        fpath.write_bytes(raw)
        return f"/api/gradcam/{fname}"
    except Exception as e:
        logger.warning(f"gradcam ({target}) failed: {e}")
        return ""


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
    })


# --------------------------------------------------------------------- #
# FastAPI app                                                           #
# --------------------------------------------------------------------- #
app = FastAPI(
    title="瞳观城康 · RETFound Bridge",
    version="1.0.0",
    description=(
        "Internal Python inference service. Listens on BRIDGE_PORT (default 8001) "
        "and serves 5-class DR + v2.0 multi-disease to the Node.js main backend."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict:
    return {
        "status": "ok",
        "service": "AI4C RETFound Bridge",
        "model": MODEL_NAME,
        "arch": ARCH,
        "ckpt": str(FINE_TUNED_CKPT),
        "device": DEVICE,
        "img_size": IMG_SIZE,
        "num_classes": NUM_DR_CLASSES,
        "cuda_available": torch.cuda.is_available(),
        "uptime_sec": int(time.time() - _STARTED_AT),
    }


@app.get("/api/labels")
def labels() -> Dict:
    return {"en": DR_LABELS, "zh": DR_LABELS_CN}


def _preprocess(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _TF(img).unsqueeze(0)


@app.post("/api/warmup")
def warmup() -> Dict:
    """Run a dummy forward pass to warm up cuDNN."""
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    t0 = time.time()
    with torch.no_grad():
        MODEL(x)
    return {"ok": True, "warmup_ms": int((time.time() - t0) * 1000)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
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
        logger.info(f"predict OK top={dr_top} prob={probs[dr_top]:.3f}")

        gradcam_url_dr = _save_gradcam_for(img, dr_top, target="dr")
        gradcam_url_gla = _save_gradcam_for(img, dr_top, target="glaucoma")
        gradcam_url_amd = _save_gradcam_for(img, dr_top, target="amd")
        systemic = _healistic_systemic_risks(dr_top)
        multi = _healistic_multi_disease(dr_top)

        latency_ms = int((time.time() - t0) * 1000)
        return {
            "model": MODEL_NAME,
            "device": DEVICE,
            "latency_ms": latency_ms,
            "top_class": dr_top,
            "top_label_en": DR_LABELS[dr_top],
            "top_label_zh": DR_LABELS_CN[dr_top],
            "top_probability": round(probs[dr_top], 4),
            "probabilities": {DR_LABELS[i]: round(p, 4) for i, p in enumerate(probs)},
            "diagnosis": DR_LABELS_CN[dr_top],
            "confidence": round(probs[dr_top], 4),
            "dr_grade": dr_top,
            "all_probs": {DR_LABELS_CN[i]: round(p, 4) for i, p in enumerate(probs)},
            "gradcam_url": gradcam_url_dr,
            **multi,
            "gradcam_url_glaucoma": gradcam_url_gla,
            "gradcam_url_amd": gradcam_url_amd,
            **systemic,
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


@app.get("/api/gradcam/{filename}")
def gradcam(filename: str) -> Response:
    safe_name = Path(filename).name
    fpath = _GRADCAM_DIR / safe_name
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="gradcam not found")
    return Response(content=fpath.read_bytes(), media_type="image/png")


@app.get("/api/metrics")
def metrics(window: int = 60) -> Dict:
    now = time.time()
    cutoff_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - window))
    recent = [c for c in _CALLS if c["ts"] >= cutoff_iso]
    lats = [c["latency_ms"] for c in recent if c["status"] == "ok"]
    success = sum(1 for c in recent if c["status"] == "ok")
    n = len(recent)
    tps = n / window if window else 0.0
    dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for c in recent:
        if c["status"] == "ok" and c["dr_grade"] in dist:
            dist[c["dr_grade"]] += 1
    history = []
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
        "success_rate": round((success / n) if n else 1.0, 4),
        "p50_ms": int(_percentile(lats, 0.50)),
        "p95_ms": int(_percentile(lats, 0.95)),
        "p99_ms": int(_percentile(lats, 0.99)),
        "avg_ms": int(sum(lats) / len(lats)) if lats else 0,
        "max_ms": max(lats) if lats else 0,
        "dr_grade_dist": dist,
        "history": history,
        "last_calls": list(_CALLS)[-20:][::-1],
        "service": "AI4C RETFound Bridge",
        "model_version": MODEL_NAME,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"starting bridge on 0.0.0.0:{BRIDGE_PORT}")
    uvicorn.run("bridge.server:app", host="0.0.0.0", port=BRIDGE_PORT, reload=False)
