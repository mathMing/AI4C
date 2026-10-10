# -*- coding: utf-8 -*-
"""
local_demo.py
=============
离线 / 演示模式：合成数据 → 2 epoch 训练 → 评估 → 导出与 backend 兼容的 final ckpt。

使用：
    python local_demo.py

不依赖任何外网、任何真实数据集、任何预训练权重下载：
- 数据：本地生成 5 类合成眼底图（每类 80 张）
- 模型：torchvision ViT-B/16（如果环境无预训练权重，则随机初始化）
- 目标：跑通整个 pipeline 证明流程可工作
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 兼容 Windows / Linux 路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prepare_data import generate_synthetic_dataset
from datasets import build_dataloaders
from models import build_model
from transforms import build_train_transform, build_val_transform
from utils.logger import get_logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def main():
    data_root = ROOT / "data"
    ckpt_dir = ROOT / "runs" / "local_demo"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("local_demo", log_dir=ckpt_dir / "logs")
    logger.info(f"data_root = {data_root}")
    logger.info(f"ckpt_dir  = {ckpt_dir}")

    # 1) 合成数据
    if not data_root.exists() or not any(data_root.iterdir()):
        logger.info("synthetic data not found, generating...")
        generate_synthetic_dataset(data_root, per_class=80, size=384)
    else:
        logger.info("synthetic data already present, skip generating")

    # 2) 加载 config
    cfg_path = ROOT / "configs" / "dr_vit_b16.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["root"] = str(data_root)
    cfg["data"]["num_workers"] = 0  # Windows 友好
    cfg["train"]["epochs"] = 2
    cfg["train"]["batch_size"] = 8

    # 3) transforms + dataloaders
    train_t = build_train_transform(img_size=cfg["data"]["img_size"], mode="weak")
    val_t = build_val_transform(img_size=cfg["data"]["img_size"])
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        data_root=data_root,
        train_transform=train_t,
        val_transform=val_t,
        batch_size=cfg["data"]["batch_size"],
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    )
    logger.info(
        f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}"
    )

    # 4) 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = build_model(cfg).to(device)
        logger.info(f"model built: {cfg['model']['name']}")
    except Exception as e:
        logger.warning(f"build_model failed ({e}), falling back to random ViT")
        from torchvision.models import vit_b_16
        from models.vit_classifier import ViTClassifier
        backbone = vit_b_16(weights=None)
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, 5)
        model = ViTClassifier(backbone, num_classes=5, name="vit_b_16_random").to(device)

    # 5) 训练 2 epoch（纯 CPU 也跑得动）
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        losses = []
        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            if step % 5 == 0:
                logger.info(f"epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")
        logger.info(f"epoch {epoch} avg train loss = {sum(losses)/len(losses):.4f}")

        # 简单 val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        acc = correct / max(1, total)
        logger.info(f"epoch {epoch} val acc = {acc:.4f}")

    # 6) 保存 final ckpt（与 backend 兼容）
    final_path = ckpt_dir / "final_classifier.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": getattr(model, "name", "vit_b_16"),
            "num_classes": getattr(model, "num_classes", 5),
            "epoch": 1,
            "metrics": {"val_acc": acc},
        },
        final_path,
    )
    logger.info(f"DONE. final ckpt -> {final_path}")


if __name__ == "__main__":
    main()
