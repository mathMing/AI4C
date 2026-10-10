# -*- coding: utf-8 -*-
"""
retfound_model.py
-----------------
Load the official RETFound MAE ViT-Large/16 backbone from
/root/autodl-tmp/RETFound/.../RETFound_mae_natureCFP.pth
and attach a 5-class diabetic-retinopathy classification head.

Architecture
------------
* backbone: timm vit_large_patch16_224 (num_classes=0)  → 1024-d feature
* head:     nn.Linear(1024, 5)                         → DR grade 0..4

Loading
-------
The official checkpoint is a dict with keys {model, optimizer, epoch, scaler, args}.
state_dict lives under obj['model'] and contains MAE encoder + decoder.
timm's vit_large_patch16_224 has no decoder, so we load with strict=False and drop
mask_token + decoder.* keys automatically.

Usage
-----
    from retfound_model import build_model, load_pretrained
    model = build_model(num_classes=5)
    model = load_pretrained(model,
        "/root/autodl-tmp/RETFound/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth",
        device="cuda")
    model.eval()
    with torch.no_grad():
        logits = model(images)   # [B, 5]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_CKPT = "/root/autodl-tmp/RETFound/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth"
BACKBONE_NAME = "vit_large_patch16_224"


def build_model(num_classes: int = 5, drop_path_rate: float = 0.1) -> nn.Module:
    """Construct a timm ViT-L/16 backbone + linear classification head."""
    import timm
    backbone = timm.create_model(
        BACKBONE_NAME,
        pretrained=False,
        num_classes=0,                # remove classifier
        global_pool="token",          # use CLS token
        drop_path_rate=drop_path_rate,
    )
    head = nn.Linear(backbone.num_features, num_classes)
    return _Classifier(backbone, head)


class _Classifier(nn.Module):
    """backbone -> global pool -> linear head. Simple wrapper for clarity."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)

    @property
    def num_features(self) -> int:
        return self.backbone.num_features


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #
def _strip_module_prefix(sd: dict) -> dict:
    """strip DDP 'module.' prefix if present."""
    if sd and any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _select_backbone_keys(sd: dict) -> Tuple[dict, list, list]:
    """
    RETFound MAE ckpt has extra decoder keys (mask_token, decoder_*).
    timm's vit_large_patch16_224 has none of those.
    Return (filtered_sd, kept, dropped).
    """
    kept, dropped = {}, []
    for k, v in sd.items():
        if k.startswith("decoder_") or k == "mask_token":
            dropped.append(k)
            continue
        kept[k] = v
    return kept, [k for k in kept], dropped


def load_pretrained(
    model: nn.Module,
    ckpt_path: str = DEFAULT_CKPT,
    device: str = "cuda",
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[nn.Module, dict]:
    """
    Load RETFound MAE weights into a built model.

    Returns
    -------
    (model, info)  where info contains:
        - epoch: int (from ckpt)
        - args:  dict (subset of Namespace)
        - loaded_keys: int
        - dropped_decoder_keys: int
        - missing_in_ckpt: int    (timm keys not found in ckpt; usually 2 = head.weight/bias)
        - unexpected_in_ckpt: int (should be 0 after filtering decoder)
    """
    ckpt_path = str(ckpt_path)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError(f"unexpected ckpt layout; top-level keys = {list(obj.keys()) if isinstance(obj, dict) else type(obj)}")

    sd = _strip_module_prefix(obj["model"])
    filtered, kept_keys, dropped = _select_backbone_keys(sd)

    # strict=False because:
    #   1. timm has no decoder/mask_token  (already filtered)
    #   2. our model has a fresh head (head.weight/bias are missing in ckpt)
    #   3. timm key names may differ slightly from MAE (e.g. norm_pre)
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # load_state_dict returns (missing_keys, unexpected_keys)
    model = model.to(device)

    args = obj.get("args", None)
    args_dict = {}
    if args is not None and hasattr(args, "__dict__"):
        # only keep a small whitelist
        keep = ("model", "batch_size", "epochs", "accum_iter", "lr",
                "weight_decay", "warmup_epochs", "mask_ratio",
                "data_path", "output_dir", "norm_pix_loss")
        args_dict = {k: getattr(args, k) for k in keep if hasattr(args, k)}

    info = {
        "ckpt": ckpt_path,
        "ckpt_size_gb": round(Path(ckpt_path).stat().st_size / 1024**3, 2),
        "epoch": obj.get("epoch", None),
        "args": args_dict,
        "loaded_keys": len(kept_keys),
        "dropped_decoder_keys": len(dropped),
        "missing_in_ckpt": len(missing),
        "unexpected_in_ckpt": len(unexpected),
        "missing_sample": missing[:10],
        "unexpected_sample": unexpected[:10],
    }
    if verbose:
        print(f"[retfound] ckpt={info['ckpt']} ({info['ckpt_size_gb']} GB) "
              f"epoch={info['epoch']} model={info['args'].get('model')}")
        print(f"[retfound] loaded={info['loaded_keys']} "
              f"dropped_decoder={info['dropped_decoder_keys']} "
              f"missing={info['missing_in_ckpt']} "
              f"unexpected={info['unexpected_in_ckpt']}")
        if missing:
            print(f"[retfound]   missing sample: {missing[:5]}")
        if unexpected:
            print(f"[retfound]   unexpected sample: {unexpected[:5]}")
    return model, info


# --------------------------------------------------------------------- #
# Image preprocessing (ImageNet stats, 224x224, like the RETFound paper)
# --------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def make_transform(img_size: int = 224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
