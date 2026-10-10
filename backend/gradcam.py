# -*- coding: utf-8 -*-
"""
backend.gradcam
================

Generate a Grad-CAM heatmap for the last block of a ViT model and
overlay it on the original image.  Returns a base64-encoded PNG so it
can be returned directly from a JSON API.

We use the official ``grad-cam`` package (aka ``pytorch-grad-cam``)
which is already in ``requirements`` as ``grad-cam``.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# pytorch-grad-cam ships as ``gradcam`` / ``grad-cam`` / ``pytorch_grad_cam``
# depending on install path.  Try several import styles.
try:
    from gradcam import GradCAM as _GradCAM
    from gradcam.utils.image import show_cam_on_image as _show_cam_on_image
except ImportError:  # pragma: no cover
    try:
        from grad_cam import GradCAM as _GradCAM  # type: ignore
        from grad_cam.utils.image import show_cam_on_image as _show_cam_on_image  # type: ignore
    except ImportError:  # pragma: no cover
        from pytorch_grad_cam import GradCAM as _GradCAM  # type: ignore
        from pytorch_grad_cam.utils.image import (  # type: ignore
            show_cam_on_image as _show_cam_on_image,
        )


IMG_SIZE = 224


def _make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def _pick_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Pick a sensible last conv-ish layer for Grad-CAM.

    For timm ViT-L we target the last ``Block`` (which has a 1x1
    attention projection with strong gradients).  For torchvision
    ``vit_b_16`` the same approach works because it uses the same
    structure under a different attribute path.
    """
    # timm plain ViT: blocks are in ``model.blocks`` (a Sequential).
    blocks = getattr(model, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        return blocks[-1]
    # torchvision vit_b_16: encoder layers in ``model.encoder.layers``
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "layers") and len(enc.layers) > 0:
        return enc.layers[-1]
    raise RuntimeError("cannot locate a suitable target layer for Grad-CAM")


def gradcam_overlay(
    model: torch.nn.Module,
    img: Image.Image,
    device: str,
    pred_class: Optional[int] = None,
) -> str:
    """Compute the Grad-CAM heatmap and return a base64 PNG string."""
    tf = _make_transform()
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Original (un-normalised) image in [0, 1] for ``show_cam_on_image``.
    rgb = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    x = tf(img).unsqueeze(0).to(device)

    target_layer = _pick_target_layer(model)

    # Build the CAM object per call; it is cheap and avoids stale state
    # between requests.
    cam = _GradCAM(model=model, target_layers=[target_layer])
    if pred_class is None:
        # ask for the highest-scoring class
        with torch.no_grad():
            logits = model(x)
        pred_class = int(torch.argmax(logits, dim=-1).item())

    targets = None  # default = highest scoring class
    grayscale = cam(input_tensor=x, targets=targets)  # (1, H, W)
    cam_map = grayscale[0]

    overlay = _show_cam_on_image(rgb, cam_map, use_rgb=True)

    out_img = Image.fromarray((overlay * 255).astype(np.uint8))
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def heatmap_png_bytes(
    model: torch.nn.Module,
    img: Image.Image,
    device: str,
    pred_class: Optional[int] = None,
) -> bytes:
    """Return the raw PNG bytes (used by ``/api/gradcam``)."""
    return base64.b64decode(gradcam_overlay(model, img, device, pred_class))
