# 🧠 AI4C · `algo/` — 模型训练与推理

> 这是 `algo/` 子目录的**详细文档**。根目录文档见 [`README.md`](../README.md)。
> 本文档面向 **A 同学（算法线）** 与需要微调 / 重新训练 / 评估模型的所有贡献者。

本目录包含 **瞳观城康 / AI4C** 项目中与**模型训练、推理、评估、可解释性（Grad-CAM）** 相关的全部代码。

---

## 📂 目录结构

```
algo/
├── configs/                # YAML 训练配置
│   ├── retfound_b16.yaml   # RETFound ViT-B/16（默认）
│   └── dr_vit_b16.yaml     # 通用 ViT-B/16
│
├── datasets/               # PyTorch Dataset 封装
│   └── fundus_dataset.py
│
├── models/                 # 模型定义
│   ├── retfound_classifier.py
│   └── vit_classifier.py
│
├── transforms/             # 数据增强
│   └── fundus_augment.py
│
├── utils/                  # 训练工具
│   ├── logger.py           # 统一 Logger
│   └── metrics.py          # QWK / Acc / F1 / Confusion Matrix
│
├── data/                   # 训练数据（⚠ gitignore，**不进仓库**）
│
├── runs/                   # 训练产物（⚠ gitignore，**不进仓库**）
│
├── train.py                # 训练主入口（AMP + cosine + QWK 选 best）
├── evaluate.py             # 评估脚本
├── gradcam.py              # Grad-CAM 可视化
├── gradcam_5classes.py     # 5 类 Grad-CAM
├── local_demo.py           # 本地推理 demo
├── prepare_data.py         # 通用数据准备
├── prepare_ddr.py          # DDR 切分 / 索引
├── prepare_ddr_real.py     # DDR 真实分布切分
├── prepare_ddr_real_flat.py
├── prepare_ddr_for_automorph.py  # 给 AutoMorph 喂数据
├── fix_ddr.py              # 修复 DDR 标签文件
├── stage2_fusion.py        # 多模态融合
├── run_automorph_full.sh   # AutoMorph 完整流程
├── requirements.txt
├── ALGO_README.md          # 本文件
└── README.md               # 根 README 的别名（保持旧链接有效）
```

---

## 🧠 模型权重（**不在仓库**）

| 文件 | 大小 | 描述 |
|------|------|------|
| `RetiZero.pth` | ~1.6 GB | RetiZero 官方权重，需自行下载到 `backend/models/` |
| `best.lora.pt` | ~12 MB | 本项目 LoRA 微调产出，训练后由 `train.py` 写出 |
| `AutoMorph-main.zip` | ~1.6 GB | AutoMorph 官方发布包，仅作数据准备参考 |
| `runs/final_classifier.pt` | ~1.2 GB | 全参数微调版，**不在仓库**（`.gitignore` 已排除） |

> ⚠ **重要**：仓库**当前没有**直接 commit 任何 `.pt / .pth` 权重。
> 根 README 与本文件早期版本中 "tracked in git" 的描述为旧版本残留，**已与 `.gitignore` 修正一致**。

---

## 🚀 快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 准备数据（DDR）
python prepare_ddr.py --src /path/to/DDR --dst data/DDR
python prepare_ddr_real.py --src /path/to/DDR --dst data/DDR_real

# 3) 训练（默认使用 RETFound ViT-B/16）
python train.py --config configs/retfound_b16.yaml

# 4) 评估
python evaluate.py --weights runs/final_classifier.pt

# 5) Grad-CAM 可视化（5 类）
python gradcam_5classes.py --weights runs/final_classifier.pt --image path/to/fundus.jpg
```

---

## 🔬 训练特性

- **AMP 混合精度** + `torch.cuda.amp.GradScaler`
- **优化器**：AdamW（`weight_decay=0.05`，`betas=(0.9, 0.999)`）
- **调度器**：cosine + warmup（`warmup_epochs=2`）
- **损失函数**：Soft Label Cross-Entropy + **QWK（Quadratic Weighted Kappa）** 作 best 指标
- **数据增强**：参见 `transforms/fundus_augment.py`（随机翻转、旋转、色彩抖动、随机擦除）
- **类别映射**：0=无 DR / 1=轻度 NPDR / 2=中度 NPDR / 3=重度 NPDR / 4=增殖期 DR
- **可复现**：所有实验支持 `SEED` 环境变量 / `--seed` 参数

---

## 📊 评估指标

- Accuracy / Balanced Accuracy
- Macro-F1 / Per-class F1
- QWK（**主指标**，与临床一致）
- AUC（One-vs-Rest）
- Confusion Matrix（heatmap 落盘到 `runs/`）

---

## 🖼 Grad-CAM

```bash
# 单图
python gradcam.py --weights runs/final_classifier.pt --image demo.jpg

# 5 类一次性
python gradcam_5classes.py --weights runs/final_classifier.pt --image demo.jpg
```

热力图保存到 `runs/gradcam/`，可直接附在 README 或论文中。

---

## 🛠 依赖（与 requirements.txt 一致）

- tqdm / PyYAML / opencv-python-headless
- scikit-learn / matplotlib / seaborn / scipy / pandas
- timm（可选，用于加载 RETFound 风格 ViT；离线环境可关）

---

## 📚 数据集

| 数据集 | 来源 | 类别 | 用途 |
|--------|------|------|------|
| **DDR** | https://github.com/nkicsl/DDR-dataset | 5 类 DR | 训练 / 评估 |
| EyePACS | Kaggle 比赛 | 5 类 DR | 训练（大规模） |
| AutoMorph MESSR | 官方 | 形态学 | 特征管线 |

> 数据集原始文件**不在仓库**，需自行准备并放到 `data/<dataset>/`。

---

## 🔗 相关资源

- **RETFound 官方仓库**：<https://github.com/rmaphoh/RETFound_MAE>
- **RetiZero 官方仓库**：<https://github.com/...>
- **AutoMorph 官方仓库**：<https://github.com/...>
- **DDR 数据集**：<https://github.com/nkicsl/DDR-dataset>

---

## 📜 许可

本目录代码与本仓库其它部分遵循相同的 **Research-Only License**（见根目录 [`LICENSE`](../LICENSE)）。
