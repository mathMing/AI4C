# -*- coding: utf-8 -*-
"""
backend.models.retfound_loader
================================

Load a RETFound-style ViT-Large backbone, replace its classification head
with a 5-class DR head, and expose a small wrapper for inference.

Strategy
--------
1) Preferred:  timm `vit_large_patch16` initialized from the official
   RETFound MAE pre-trained checkpoint (RETFound_mae_natureCFP.pth).
   This is a 1.3 GB checkpoint that contains the full ViT-Large weights.
2) Fallback-1: timm `vit_large_patch16` initialized with ImageNet-1k
   pre-trained weights (no fine-tuning, but the architecture is right).
3) Fallback-2: torchvision `vit_b_16` with ImageNet-1k weights (used
   before this loader was written — same API, smaller model).

The interface is uniform: ``build_model()`` returns a tuple
``(model, name)`` where ``name`` is a human-readable string used by the
API for ``/api/health`` and ``model_version``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Project-relative default path.  Resolved lazily so this module can be
# imported even if the weights are not yet present.
DEFAULT_CFP_CKPT = Path(
    os.environ.get(
        "RETFOUND_CKPT",
        "/root/autodl-tmp/RETFound/RETFound_mae_natureCFP.pth",
    )
)

NUM_DR_CLASSES = 5

DR_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
DR_LABELS_CN = {0: "无DR", 1: "轻度NPDR", 2: "中度NPDR", 3: "重度NPDR", 4: "增殖期DR"}


# --------------------------------------------------------------------- #
# helpers                                                               #
# --------------------------------------------------------------------- #
def _replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the classification head of a timm / torchvision ViT."""
    # timm ViT
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes)
        return model
    # torchvision ViT (vit_b_16 / vit_l_16)
    if hasattr(model, "heads") and isinstance(model.heads, nn.Module):
        in_features = model.heads.head.in_features
        model.heads = nn.Linear(in_features, num_classes)
        return model
    raise RuntimeError("Unknown backbone — cannot replace classification head")


def _load_pretrained_timm(arch: str, ckpt: Optional[Path]) -> Tuple[nn.Module, str]:
    """Build a timm ViT and try to load the RETFound checkpoint."""
    import timm  # local import: timm is large, only pay the cost if we use it

    # ``timm`` >= 1.0 removed the bare ``vit_large_patch16`` alias.  We
    # have to use the explicit ``_224`` variant for plain ViT-L.
    if arch == "vit_large_patch16":
        arch = "vit_large_patch16_224"

    # Build the bare ViT-L/16 with no head first.
    model = timm.create_model(arch, pretrained=False, num_classes=0)
    if ckpt is not None and ckpt.exists():
        state = torch.load(str(ckpt), map_location="cpu")
        # RETFound checkpoints are either a raw state_dict, or wrapped in
        # {"model": ..., "state_dict": ...}.  Normalise.
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        # Some checkpoints prefix everything with "module." (DDP); strip.
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # RETFound saves the MAE encoder; the classification head is
        # missing, so allow missing keys (the head) and unexpected keys
        # (the original 1000-class head).
        missing, unexpected = model.load_state_dict(state, strict=False)
        if any(("head" not in k) for k in missing):
            raise RuntimeError(f"load_state_dict missing non-head keys: {[k for k in missing if 'head' not in k][:5]}")
        name = "RETFound_mae_natureCFP"
    else:
        # Fallback-1: ImageNet pre-trained.
        model = timm.create_model(arch, pretrained=True, num_classes=0)
        name = f"timm_{arch}_in1k"
    model = _replace_head(model, NUM_DR_CLASSES)
    return model, name


def _load_torchvision_fallback() -> Tuple[nn.Module, str]:
    """Fallback-2: torchvision vit_b_16 (the model that was already used)."""
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    m = _replace_head(m, NUM_DR_CLASSES)
    return m, "vit-b-16-imagenet1k"


# --------------------------------------------------------------------- #
# public API                                                            #
# --------------------------------------------------------------------- #
def build_model(arch: str = "vit_large_patch16", ckpt: Optional[Path] = None) -> Tuple[nn.Module, str]:
    """Build a DR classifier.

    Returns
    -------
    (model, name)
        ``name`` is a short string suitable for the API.
    """
    if ckpt is None:
        ckpt = DEFAULT_CFP_CKPT

    last_err: Optional[Exception] = None
    # Strategy 1/2: timm
    try:
        model, name = _load_pretrained_timm(arch, ckpt)
        return model, name
    except Exception as e:  # pragma: no cover — best-effort fallback
        last_err = e
        print(f"[retfound] timm path failed ({e}); falling back to torchvision ViT-B/16")

    # Strategy 3: torchvision
    try:
        return _load_torchvision_fallback()
    except Exception as e:
        raise RuntimeError(
            f"All backends failed.  timm error: {last_err}; torchvision error: {e}"
        )


if __name__ == "__main__":
    # CLI sanity check.
    m, name = build_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        y = m(x)
    print(f"OK: {name} on {device} -> logits {tuple(y.shape)}")
