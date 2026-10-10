# -*- coding: utf-8 -*-
"""
image_qc.py
===========
基层眼底图像的离线质控模块。

功能：
    1. 分辨率检查       —— 是否在可接受尺寸范围（>= 256 任一边）
    2. 亮度过暗/过曝     —— 灰度均值
    3. 模糊度            —— Laplacian 方差，越大越清晰
    4. 视盘区域亮度比    —— 中心 30% 平均亮度 / 周边平均亮度，> 1.0 视作"含视盘"
    5. 左右眼自动识别   —— 用图像宽中线分成左右两半，比较平均亮度与方差，
                          视盘较亮的那一侧 = 视盘侧，即 OS/OD 推断
    6. 综合评分         —— 0-100 整数，> 60 视作通过

设计原则：
    - 零重型依赖（只依赖 PIL + numpy）
    - 全部纯函数，可单元测试
    - 即使图片完全不是眼底图（如风景/自拍），也只会返回低分 + 明确提示
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter


# ----------------------------- 阈值常量 ----------------------------- #
MIN_SIDE = 256           # 任一边最少像素
DARK_MEAN = 35.0         # 平均灰度低于此视为过暗
BRIGHT_MEAN = 220.0      # 平均灰度高于此视为过曝
BLUR_VAR = 60.0          # Laplacian 方差低于此视为模糊
QC_PASS_SCORE = 60       # 综合分阈值
LATERAL_SPLIT = 0.5      # 左右眼分界线（图像宽中线）


# ----------------------------- 结果数据类 ----------------------------- #
@dataclass
class QCResult:
    ok: bool                       # 是否通过质控
    score: int                     # 0-100
    reason: str                    # 若不通过，给出人类可读原因
    width: int
    height: int
    mean_gray: float               # 灰度均值
    blur_var: float                # Laplacian 方差
    disc_ratio: float              # 视盘区域亮度比
    laterality: str                # "OD" / "OS" / "UNKNOWN"
    details: Dict[str, Any]        # 详细信息，方便前端展示

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------- 核心函数 ----------------------------- #
def _to_gray_array(img: Image.Image) -> np.ndarray:
    """PIL.Image -> 灰度 uint8 ndarray."""
    return np.asarray(img.convert("L"), dtype=np.uint8)


def _laplacian_var(gray: np.ndarray) -> float:
    """Laplacian 方差：越大越清晰，越小越模糊。"""
    pil = Image.fromarray(gray)
    edge = pil.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edge, dtype=np.float32)
    return float(arr.var())


def _disc_ratio(gray: np.ndarray) -> float:
    """
    视盘区域亮度比 = 中心 30% 平均 / 周边 30% 平均。
    视盘是眼底图最亮的圆形结构，所以中心>周边通常>1.0。
    """
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.15), int(w * 0.15)   # 中心区半径
    rin_h, rin_w = int(h * 0.35), int(w * 0.35)  # 环带内半径
    rout_h, rout_w = int(h * 0.50), int(w * 0.50)  # 环带外半径

    center = gray[cy - rh:cy + rh, cx - rw:cx + rw].mean()
    ring = gray[
        max(0, cy - rout_h):max(0, cy - rin_h),
        max(0, cx - rout_w):max(0, cx - rin_w),
    ]
    if ring.size == 0 or float(ring.mean()) < 1e-3:
        return 1.0
    return float(center / (ring.mean() + 1e-6))


def _infer_laterality(gray: np.ndarray) -> str:
    """
    简单左右眼识别：把图像按宽中线分成左右两半，
    视盘（更亮、纹理更密）所在的一侧 = 视盘侧。

    临床约定：
        - 视盘位于鼻侧 (nasal)
        - 右眼 (OD) 视盘在图像左侧
        - 左眼 (OS) 视盘在图像右侧
    因此更亮的那一侧如果是图像的左半 → OD；右半 → OS。
    """
    h, w = gray.shape
    mid = int(w * LATERAL_SPLIT)
    left = gray[:, :mid]
    right = gray[:, mid:]

    # 用"亮度 + 边缘能量"组合判定
    def energy(region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        # 边缘能量
        pil = Image.fromarray(region)
        edge = np.asarray(pil.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        return float(region.mean() + 0.5 * edge.mean())

    el = energy(left)
    er = energy(right)

    # 差距过小 → 难以判断
    if abs(el - er) < 3.0:
        return "UNKNOWN"
    return "OD" if el > er else "OS"


# ----------------------------- 主入口 ----------------------------- #
def run_image_qc(
    img: Image.Image,
    min_side: int = MIN_SIDE,
    dark_mean: float = DARK_MEAN,
    bright_mean: float = BRIGHT_MEAN,
    blur_var: float = BLUR_VAR,
) -> QCResult:
    """
    对单张 RGB 眼底图做完整质控。

    Parameters
    ----------
    img : PIL.Image
        RGB 图像。
    min_side, dark_mean, bright_mean, blur_var : float
        阈值参数，可由调用方覆盖（用于产品级自定义）。

    Returns
    -------
    QCResult
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    gray = _to_gray_array(img)
    mean_gray = float(gray.mean())
    blur = _laplacian_var(gray)
    disc = _disc_ratio(gray)
    lat = _infer_laterality(gray)

    # ---------- 评分 ----------
    score = 100
    reasons = []

    if min(w, h) < min_side:
        score -= 35
        reasons.append(f"分辨率过小（{w}×{h}，需 ≥ {min_side}）")
    if mean_gray < dark_mean:
        score -= 25
        reasons.append(f"图像过暗（灰度均值 {mean_gray:.1f}）")
    elif mean_gray > bright_mean:
        score -= 20
        reasons.append(f"图像过曝（灰度均值 {mean_gray:.1f}）")
    if blur < blur_var:
        score -= 20
        reasons.append(f"图像模糊（边缘能量 {blur:.1f}）")
    if disc < 0.95:
        score -= 10
        reasons.append("未检测到典型视盘结构（可能非眼底图）")
    if lat == "UNKNOWN":
        score -= 5
        reasons.append("左右眼无法自动判定")

    score = max(0, min(100, int(score)))
    ok = score >= QC_PASS_SCORE
    reason = "；".join(reasons) if reasons else "通过"

    return QCResult(
        ok=ok,
        score=score,
        reason=reason,
        width=w,
        height=h,
        mean_gray=mean_gray,
        blur_var=blur,
        disc_ratio=disc,
        laterality=lat,
        details={
            "min_side": min_side,
            "dark_mean": dark_mean,
            "bright_mean": bright_mean,
            "blur_var": blur_var,
            "pass_score": QC_PASS_SCORE,
        },
    )


# ----------------------------- CLI 自检 ----------------------------- #
if __name__ == "__main__":
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python image_qc.py <image_path>")
        sys.exit(1)

    img = Image.open(sys.argv[1])
    r = run_image_qc(img)
    import json
    print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
