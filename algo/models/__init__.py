"""
模型定义：DR 5 分类器。
- vit_b_16：torchvision ImageNet ViT（兜底）
- retfound_mae：RETFound nature-CFP（MAE 自监督预训练）
"""
from .vit_classifier import build_model, ViTClassifier
from .retfound_classifier import RETFoundClassifier, build_retfound_model

__all__ = ["build_model", "ViTClassifier", "RETFoundClassifier", "build_retfound_model"]


def build_model_unified(cfg):
    """
    根据配置选择模型：
        cfg["model"]["name"] == "retfound_mae" → RETFound
        其他 → torchvision ViT
    """
    name = cfg["model"].get("name", "vit_b_16")
    if name == "retfound_mae":
        return build_retfound_model(cfg)
    else:
        return build_model(cfg)
