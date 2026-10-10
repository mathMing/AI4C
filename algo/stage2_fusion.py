"""
Stage 2: 把 AutoMorph 血管特征融合到 5 类 DR 模型
====================================================
输入:
  - runs/20261009_120703/best.pt  (ViT-L LoRA, 0.7827 QWK)
  - AutoMorph 输出 CSV (4105 张测试图)
  - DDR 5 类真实标签
输出:
  - 多模态融合模型 QWK 评估
  - 存 best_fusion.pt
"""
import os, json, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from timm.layers import adapt_input_conv
from sklearn.metrics import cohen_kappa_score, accuracy_score
from peft import LoraConfig, get_peft_model

DDR_TEST = "/root/autodl-tmp/AI4C/algo/data/DDR-dataset/DR_grading/test"
CSV_MERGE = "/root/autodl-tmp/AI4C/AutoMorph-main/Results/M3/measurement_results.csv"
AUTOMORPH_CSV = "/root/autodl-tmp/AI4C/AutoMorph-main/Results/M3/measurement_results.csv"
MAPPING_CSV = "/root/autodl-tmp/AI4C/AutoMorph-main/name_mapping.csv"
DDR_LABELS = "/root/autodl-tmp/AI4C/algo/data/DDR-dataset/DR_grading/test.txt"
CKPT = "/root/autodl-tmp/AI4C/algo/runs/20261009_120703/best.pt"
OUT = "/root/autodl-tmp/AI4C/algo/runs/stage2_fusion"

os.makedirs(OUT, exist_ok=True)

def load_labels():
    """读 DDR 测试图标签"""
    labels = {}
    with open(DDR_LABELS) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return labels

def load_automorph_features():
    """读 AutoMorph CSV, 通过映射表匹配 DDR 标签"""
    df = pd.read_csv(AUTOMORPH_CSV)
    print(f"AutoMorph features: {df.shape}")
    print(f"列名: {list(df.columns)[:20]}...")

    mapping = pd.read_csv(MAPPING_CSV)
    df = df.merge(mapping, left_on='Name', right_on='id', how='inner')
    print(f"匹配后: {df.shape}")

    labels = load_labels()
    df['label'] = df['orig_name'].map(labels)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f"有标签的: {df.shape}, 分布: \n{df['label'].value_counts().sort_index()}")
    return df

def main():
    df = load_automorph_features()
    # 数值列
    feat_cols = [c for c in df.columns if c not in ['Name','id','orig_name','label'] and df[c].dtype != 'O']
    print(f"\n特征列数: {len(feat_cols)}")
    print(f"示例: {feat_cols[:10]}")

    # 保存
    df.to_csv(f"{OUT}/merged_features.csv", index=False)
    print(f"\n? 保存: {OUT}/merged_features.csv")

if __name__ == "__main__":
    main()
