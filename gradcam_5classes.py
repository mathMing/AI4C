"""
为 5 分类 DR 生成 Grad-CAM 热力图
"""
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os, glob

# ========== 参数配置 ==========
CKPT_PATH = "runs/20261009_112946/best.pt"
IMG_DIR   = "data/DDR_real/images"
NUM_CLASSES = 5
CLASS_NAMES = ["无病变(0)", "轻度(1)", "中度(2)", "重度(3)", "增殖性(4)"]

# ========== 加载模型 ==========
# 这里需要你把 train.py 里的模型构建逻辑复制过来
# 我先给你一个简化版，你需要对应修改
import sys
sys.path.insert(0, '/root/autodl-tmp/AI4C/algo')
from fundus_dataset import FundusDataset  # 如果有 dataset 类

# 加载权重看看结构
ckpt = torch.load(CKPT_PATH, map_location="cpu")
print("=== Checkpoint keys ===")
for k in ckpt.keys():
    v = ckpt[k]
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__}")
