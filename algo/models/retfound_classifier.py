# -*- coding: utf-8 -*-
"""
retfound_classifier.py
======================
RETFound (MAE / nature-CFP 预训练) 5 类 DR 分类器 + LoRA 微调。

加载策略：
1. 读 RETFound_mae_natureCFP.pth（嵌套 dict，key 是 "model"/"state_dict"/...）
2. 用 timm 构造同等结构 ViT-L/16（不加载预训练）作为 `backbone`
3. 把 RETFound 权重复制进去（自动跳过 mask_token/decoder_*/head.*）
4. 替换 head 为 5 类
5. 【新增】冻结 backbone，在 qkv/proj/fc1/fc2 上注入 LoRA

参考：
    https://github.com/rmaphoh/RETFound
    timm vit_large_patch16_224 的结构
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("需要 timm: pip install timm>=0.9") from e


# =====================================================================
# LoRA 工具
# =====================================================================

class LoRALinear(nn.Module):
    """
    原始线性层 + 低秩旁路：y = W x + (B @ A) x
    - 训练时只更新 A、B
    - 推理时可合并到 W（不增加参数）
    """

    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 冻结原始权重
        self.weight = original_linear.weight  # 不复制，引用原 linear
        self.bias = original_linear.bias
        # 关闭原权重的梯度
        for p in self.weight, self.bias:
            if p is not None:
                p.requires_grad = False

        # LoRA 旁路（dropout + A + B）
        self.lora_dropout = nn.Dropout(dropout)
        # A: [r, in], B: [out, r]，初始化 A 走 Kaiming，B 走 0（保证初始旁路为 0）
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主路径（冻结）
        out = nn.functional.linear(x, self.weight, self.bias)
        # LoRA 旁路：A → dropout → x @ A^T @ B^T
        lora_x = self.lora_dropout(x)
        lora_out = nn.functional.linear(
            nn.functional.linear(lora_x, self.lora_A),
            self.lora_B,
        )
        return out + lora_out * self.scaling

    def merge_into(self):
        """把 LoRA 合并到主权重（推理用）。返回一个新的 nn.Linear。"""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            merged.weight.copy_(self.weight + delta_w)
            if self.bias is not None:
                merged.bias.copy_(self.bias)
        return merged

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}"


def inject_lora_into_vit(
    vit_model: nn.Module,
    target_substrings=("qkv", "proj", "fc1", "fc2"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> dict:
    """
    在 timm ViT 的 Attention/MLP 块中，把目标 nn.Linear 替换成 LoRALinear。
    冻结其他所有参数。
    返回 {name: LoRALinear}。
    """
    # 1) 冻结全部
    for p in vit_model.parameters():
        p.requires_grad = False

    lora_modules: dict[str, LoRALinear] = {}
    replaced = 0

    for name, module in vit_model.named_modules():
        # 跳过非叶子（容器）
        for sub in target_substrings:
            if sub in name and isinstance(module, nn.Linear):
                # 找到父模块
                parent_name, _, attr = name.rpartition(".")
                parent = vit_model
                if parent_name:
                    parent = dict(vit_model.named_modules())[parent_name]
                # 替换
                new_mod = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, attr, new_mod)
                lora_modules[name] = new_mod
                replaced += 1
                break

    return lora_modules


# =====================================================================
# 主模型
# =====================================================================

class RETFoundClassifier(nn.Module):
    """
    RETFound 5 类分类器（LoRA 微调版）
    - backbone: timm ViT-L/16
    - head: nn.Linear(1024, num_classes)
    - lora_modules: 注入到 attention + MLP 的低秩旁路
    """

    def __init__(
        self,
        num_classes: int = 5,
        weights_path: Optional[str] = None,
        img_size: int = 224,
        dropout: float = 0.1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.name = "retfound_mae_lora"
        self.num_classes = num_classes
        self.patch_size = 16
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # 1) 构造 timm ViT-L/16（与 RETFound MAE 结构一致）
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            img_size=img_size,
        )
        feat_dim = self.backbone.num_features  # 1024

        # 2) 加载 RETFound 权重
        if weights_path and os.path.exists(weights_path):
            print(f"[RETFound] 加载权重: {weights_path}")
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                if "model" in ckpt and isinstance(ckpt["model"], dict):
                    state = ckpt["model"]
                elif "state_dict" in ckpt:
                    state = ckpt["state_dict"]
                else:
                    state = ckpt
            else:
                state = ckpt
            state = {k.replace("module.", ""): v for k, v in state.items()}

            own_state = self.backbone.state_dict()
            new_state = {}
            for k, v in state.items():
                if k in own_state and own_state[k].shape == v.shape:
                    new_state[k] = v
            own_state.update(new_state)
            self.backbone.load_state_dict(own_state, strict=False)
            print(f"[RETFound] 加载 {len(new_state)}/{len(own_state)} 个 encoder 权重")
        else:
            print(f"[RETFound] 警告：未找到权重 {weights_path}，使用随机初始化")

        # 3) 替换 head 为 5 类（这部分不冻结，全量训练）
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

        # 4) 注入 LoRA
        self.lora_modules = inject_lora_into_vit(
            self.backbone,
            target_substrings=("qkv", "proj", "fc1", "fc2"),
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        # 统计
        lora_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"[LoRA] 已注入 {len(self.lora_modules)} 个 LoRA 层 | "
            f"trainable={lora_params/1e6:.2f}M / total={total_params/1e6:.1f}M "
            f"({100*lora_params/total_params:.2f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def get_last_block(self):
        return self.backbone.blocks[-1]

    def get_class_token_idx(self) -> int:
        return 0

    def get_lora_state_dict(self) -> dict:
        """只导出 LoRA + head 权重（20MB 左右）。"""
        sd = {}
        for name, mod in self.lora_modules.items():
            sd[f"lora.{name}.lora_A"] = mod.lora_A
            sd[f"lora.{name}.lora_B"] = mod.lora_B
        sd["head"] = self.head.state_dict()
        return sd

    def load_lora_state_dict(self, sd: dict):
        """加载 LoRA + head 权重。"""
        for name, mod in self.lora_modules.items():
            mod.lora_A.data = sd[f"lora.{name}.lora_A"].to(mod.lora_A.device)
            mod.lora_B.data = sd[f"lora.{name}.lora_B"].to(mod.lora_B.device)
        self.head.load_state_dict(sd["head"])

    def merge_lora(self) -> "RETFoundClassifier":
        """返回一个新模型，LoRA 已合并到主权重（推理用，无需再传 adapter）。"""
        # 深拷贝
        import copy
        merged_model = copy.deepcopy(self)
        for name, mod in merged_model.lora_modules.items():
            # 找到对应位置（基于 name）
            parent_name, _, attr = name.rpartition(".")
            parent = merged_model
            if parent_name:
                parent = dict(merged_model.named_modules())[parent_name]
            setattr(parent, attr, mod.merge_into())
        merged_model.lora_modules = {}  # 清空
        return merged_model


def build_retfound_model(cfg) -> RETFoundClassifier:
    """按配置构建 RETFound 分类器。"""
    lora_cfg = cfg.get("lora", {})
    return RETFoundClassifier(
        num_classes=cfg["model"]["num_classes"],
        weights_path=cfg["model"].get("retfound_weights", None),
        img_size=cfg["data"]["img_size"],
        dropout=cfg["model"].get("dropout", 0.1),
        lora_r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
    )
