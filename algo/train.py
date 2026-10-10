# -*- coding: utf-8 -*-
"""
train.py
========
A 同学训练主入口。

用法：
    python train.py --config configs/dr_vit_b16.yaml
    python train.py --config configs/dr_vit_b16.yaml --epochs 5   # 快速 sanity check

训练流程：
1. 加载数据
2. 构建模型（torchvision ViT-B/16 ImageNet pretrained）
3. 优化器（backbone vs head 不同 lr）+ cosine schedule + warmup
4. AMP 混合精度训练
5. 每个 epoch 在 val 集评估：acc / QWK / AUC
6. 保存 best ckpt（按 QWK 选）
7. 早停（patience 5）
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from datasets import build_dataloaders
from models import build_model_unified
from transforms import build_train_transform, build_val_transform
from utils.logger import AverageMeter, Timer, get_logger
from utils.metrics import compute_metrics, pretty_print_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cast_lr(cfg: dict):
    """YAML 会把 1e-4 读成字符串，强制转 float。"""
    t = cfg.get("train", {})
    for k in ("lr", "head_lr", "min_lr", "weight_decay", "label_smoothing", "grad_clip"):
        if k in t and isinstance(t[k], str):
            try:
                t[k] = float(t[k])
            except ValueError:
                pass
    for k in ("batch_size", "epochs", "warmup_epochs", "num_workers", "early_stop_patience"):
        if k in t and isinstance(t[k], str):
            try:
                t[k] = int(t[k])
            except ValueError:
                pass
    return cfg


def build_optimizer(model: nn.Module, cfg: dict) -> AdamW:
    """
    分三组学习率（LoRA 微调模式）：
    - backbone (requires_grad=False, LoRA 旁路另外处理)
    - lora: LoRA 旁路 A/B 参数 → 用 head_lr × 0.1（推荐 1e-4 量级）
    - head:  分类头 → 用 head_lr（最大）
    """
    lora_params, head_params, other_trainable = [], [], []
    head_lr = cfg.get("head_lr", cfg["lr"] * 10)
    lora_lr = cfg.get("lora_lr", head_lr * 0.1)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # LoRA 旁路
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
        # 分类头
        elif "head" in name:
            head_params.append(param)
        else:
            other_trainable.append(param)

    groups = []
    if other_trainable:
        groups.append({"params": other_trainable, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]})
    if lora_params:
        groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": 0.0})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": cfg["weight_decay"]})

    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in head_params)
    n_other = sum(p.numel() for p in other_trainable)
    print(f"[Optimizer] lora={n_lora/1e6:.2f}M @ lr={lora_lr:.1e} | head={n_head/1e6:.3f}M @ lr={head_lr:.1e} | other={n_other/1e6:.3f}M @ lr={cfg['lr']:.1e}")

    return AdamW(groups)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """cosine + linear warmup。"""
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(cfg["min_lr"] / cfg["lr"], 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, epoch, logger, amp_enabled=True, grad_clip=1.0, log_every=20):
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    pbar = tqdm(loader, desc=f"epoch {epoch:02d} train", ncols=100)
    for step, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(imgs)
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        loss_meter.update(loss.item(), n=imgs.size(0))
        acc_meter.update(acc, n=imgs.size(0))

        if step % log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.3f}", lr=f"{lr_now:.2e}")

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, amp_enabled=True, return_probs=True):
    model.eval()
    loss_meter = AverageMeter("loss")
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="eval", ncols=100):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(imgs)
            loss = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if return_probs:
            all_probs.append(probs.cpu().numpy())

        loss_meter.update(loss.item(), n=imgs.size(0))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs) if return_probs else None

    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    metrics["loss"] = loss_meter.avg
    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path: Path, save_lora_separately: bool = True):
    """
    保存 checkpoint。
    默认：模型权重（完整） + LoRA 单独文件（<20MB） + 优化器等状态合并到主 ckpt
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "metrics": metrics,
        "model_name": model.name,
        "num_classes": model.num_classes,
    }
    torch.save(state, path)
    print(f"[ckpt] saved {path}")

    # 额外保存 LoRA adapter（轻量版，方便分发/继续训练）
    if save_lora_separately and hasattr(model, "get_lora_state_dict"):
        lora_path = path.with_suffix(".lora.pt")
        lora_sd = model.get_lora_state_dict()
        torch.save({
            "epoch": epoch,
            "lora_state": lora_sd,
            "model_name": model.name,
            "num_classes": model.num_classes,
            "metrics": metrics,
        }, lora_path)
        print(f"[ckpt] saved LoRA adapter: {lora_path} ({sum(v.numel() for v in lora_sd.values())/1e6:.2f}M params)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dr_vit_b16.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = cast_lr(cfg)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root

    set_seed(cfg["seed"])

    output_dir = Path(args.output_dir or cfg["output"]["ckpt_dir"]) / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train", log_dir=log_dir)
    logger.info(f"output dir: {output_dir}")
    logger.info(f"config: {cfg}")

    # --- data --- #
    data_root = Path(cfg["data"]["root"])
    if not data_root.exists() or not any(data_root.iterdir()):
        logger.warning(f"data root empty: {data_root}, generating synthetic data")
        from prepare_data import generate_synthetic_dataset
        generate_synthetic_dataset(data_root, per_class=80, size=384)
    else:
        logger.info(f"data root: {data_root}")

    train_t = build_train_transform(img_size=cfg["data"]["img_size"], mode=cfg["data"].get("augment", "strong"))
    val_t = build_val_transform(img_size=cfg["data"]["img_size"])
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        data_root=data_root,
        train_transform=train_t,
        val_transform=val_t,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["seed"],
    )
    logger.info(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")
    logger.info(f"class weights: {class_weights}")

    # --- model --- #
    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_model_unified(cfg).to(device)
    logger.info(f"model: {cfg['model']['name']}  num_classes={cfg['model']['num_classes']}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # --- loss --- #
    if cfg["train"].get("use_class_weight") and class_weights:
        weights = torch.tensor([class_weights[i] for i in range(cfg["model"]["num_classes"])], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["train"].get("label_smoothing", 0.1))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.1))

    # --- optimizer & scheduler --- #
    optimizer = build_optimizer(model, cfg["train"])
    scheduler = build_scheduler(optimizer, cfg["train"], steps_per_epoch=len(train_loader))
    scaler = GradScaler("cuda", enabled=cfg["train"].get("amp", True) and not args.no_amp)

    # --- train loop --- #
    best_qwk = -1.0
    no_improve = 0
    patience = cfg["train"].get("early_stop_patience", 5)
    timer = Timer()

    for epoch in range(cfg["train"]["epochs"]):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, device,
            epoch, logger, amp_enabled=not args.no_amp,
            grad_clip=cfg["train"].get("grad_clip", 1.0),
        )
        v_metrics = evaluate(model, val_loader, criterion, device, cfg["model"]["num_classes"], amp_enabled=not args.no_amp)
        logger.info(f"\n[epoch {epoch:02d}] train_loss={t_loss:.4f}  train_acc={t_acc:.4f}\n{pretty_print_metrics(v_metrics, cfg['model']['num_classes'])}")

        # save best (by QWK)
        if v_metrics["qwk"] > best_qwk:
            best_qwk = v_metrics["qwk"]
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, v_metrics, output_dir / "best.pt")
            no_improve = 0
        else:
            no_improve += 1

        # save latest
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, v_metrics, output_dir / "latest.pt")

        if no_improve >= patience:
            logger.warning(f"early stop at epoch {epoch} (best QWK={best_qwk:.4f})")
            break

    # --- test --- #
    logger.info("=" * 60)
    logger.info("final test set evaluation (using best ckpt)")
    ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device, cfg["model"]["num_classes"], amp_enabled=not args.no_amp)
    logger.info("\n" + pretty_print_metrics(test_metrics, cfg["model"]["num_classes"]))

    # save final ckpt compatible with backend.py
    final_path = Path("/root/autodl-tmp/AI4C/algo/runs/final_classifier.pt")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, scheduler, scaler, epoch, test_metrics, final_path)
    logger.info(f"DONE in {timer}, best QWK={best_qwk:.4f}")


if __name__ == "__main__":
    main()
