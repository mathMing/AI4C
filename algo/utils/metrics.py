# -*- coding: utf-8 -*-
"""
metrics.py
=========
分类指标：
- accuracy
- quadratic weighted kappa（DR 任务必备）
- per-class precision / recall
- AUC (one-vs-rest)
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """Quadratic weighted kappa（DR 任务的官方评估指标之一）。"""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    num_classes: int = 5,
) -> dict:
    """计算全套指标。"""
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["qwk"] = float(quadratic_weighted_kappa(y_true, y_pred))

    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
    metrics["per_class_precision"] = {int(i): float(p[i]) for i in range(num_classes)}
    metrics["per_class_recall"] = {int(i): float(r[i]) for i in range(num_classes)}
    metrics["per_class_f1"] = {int(i): float(f1[i]) for i in range(num_classes)}
    metrics["per_class_support"] = {int(i): int(support[i]) for i in range(num_classes)}

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(num_classes))))
        except Exception:
            metrics["auc_ovr"] = None

    return metrics


def pretty_print_metrics(metrics: dict, num_classes: int = 5) -> str:
    lines = [
        f"Accuracy : {metrics['accuracy']:.4f}",
        f"QWK      : {metrics['qwk']:.4f}",
    ]
    if "auc_ovr" in metrics and metrics["auc_ovr"] is not None:
        lines.append(f"AUC OvR  : {metrics['auc_ovr']:.4f}")
    lines.append("Per-class (P / R / F1 / support):")
    for i in range(num_classes):
        lines.append(
            f"  Class {i}  P={metrics['per_class_precision'][i]:.3f}  "
            f"R={metrics['per_class_recall'][i]:.3f}  "
            f"F1={metrics['per_class_f1'][i]:.3f}  "
            f"N={metrics['per_class_support'][i]}"
        )
    return "\n".join(lines)
