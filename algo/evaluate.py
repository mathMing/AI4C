# -*- coding: utf-8 -*-
"""
evaluate.py
===========
独立评估脚本：加载 best.pt 在 val/test 集上评估。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from datasets import build_dataloaders
from transforms import build_val_transform
from utils.logger import get_logger
from utils.metrics import pretty_print_metrics, compute_metrics
from train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="/root/autodl-tmp/AI4C/algo/runs/final_classifier.pt")
    ap.add_argument("--data_root", type=str, default="/root/autodl-tmp/AI4C/algo/data")
    ap.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    logger = get_logger("eval")
    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    logger.info(f"loaded {args.ckpt} (epoch={ckpt.get('epoch')}, model={ckpt.get('model_name')})")

    from models import ViTClassifier
    model = ViTClassifier(name=ckpt["model_name"], num_classes=ckpt["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_t = build_val_transform(img_size=224)
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        data_root=Path(args.data_root),
        train_transform=val_t,
        val_transform=val_t,
        batch_size=32,
        num_workers=2,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device, ckpt["num_classes"])
    logger.info("\n" + pretty_print_metrics(metrics, ckpt["num_classes"]))


if __name__ == "__main__":
    main()
