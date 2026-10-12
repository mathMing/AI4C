# -*- coding: utf-8 -*-
"""
vit_classifier.py
================
眼底图像分类器：基于 torchvision ViT 的迁移学习。

支持：
    - vit_b_16（默认，ImageNet1K 预训练）
    - vit_l_16 / vit_h_14（大模型）
    - timm ViT（如未来接入 RETFound）
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models


class ViTClassifier(nn.Module):
    """
    ViT 分类器封装：
    - 加载 torchvision 预训练 backbone
    - 替换分类头为 num_classes 输出
    - 提供特征提取接口（用于 Grad-CAM）
    """

    def __init__(self, name: str = "vit_b_16", num_classes: int = 5, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        self.name = name
        self.num_classes = num_classes

        # 加载 backbone
        model_fn = getattr(tv_models, name, None)
        if model_fn is None:
            raise ValueError(f"unsupported model: {name}")

        weights_enum = None
        if pretrained:
            try:
                weights_enum = self._get_weights(name)
            except Exception as e:
                print(f"[ViTClassifier] no weights for {name} ({e})")

        if weights_enum is not None:
            try:
                self.backbone = model_fn(weights=weights_enum)
            except Exception as e:
                print(f"[ViTClassifier] pretrained load failed ({e}), using random init")
                self.backbone = model_fn(weights=None)
        else:
            self.backbone = model_fn(weights=None)

        # 替换分类头
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # 暴露 patch grid 大小（用于 Grad-CAM）
        # torchvision ViT patch16 @ 224x224 -> 14x14 = 196 patches
        self.patch_size = 16

    @staticmethod
    def _get_weights(name: str):
        """Try several weight enum names; return the first that exists."""
        from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights
        candidates = {
            "vit_b_16": [getattr(ViT_B_16_Weights, "IMAGENET1K_V1", None), getattr(ViT_B_16_Weights, "DEFAULT", None)],
            "vit_l_16": [getattr(ViT_L_16_Weights, "IMAGENET1K_V1", None), getattr(ViT_L_16_Weights, "DEFAULT", None)],
            "vit_h_14": [],  # SWAG weights may not exist in current torchvision
        }
        for w in candidates.get(name, []):
            if w is not None:
                return w
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_last_block(self):
        """获取 ViT 最后一个 encoder block（用于 Grad-CAM hook）。"""
        return self.backbone.encoder.layers[-1]

    def get_class_token_idx(self) -> int:
        """CLS token 在序列中的索引（torchvision ViT = 0）。"""
        return 0


def build_model(cfg) -> ViTClassifier:
    """从配置构建模型。"""
    return ViTClassifier(
        name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"].get("dropout", 0.1),
    )
