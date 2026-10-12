# 👁 瞳观城康 · AI4C

> **AI for Cities** — 研究生智慧城市比赛 · 智慧医疗赛道
> 基于视网膜基础模型（**RETFound**）的 **眼底—全身慢病早筛** 与 **城市健康管理** 一体化平台

> 一张眼底照，洞察眼与全身；一个平台，连接社区与城市。

[![License: Research Only](https://img.shields.io/badge/License-Research%20Only-orange.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Code style: PEP 8](https://img.shields.io/badge/Code%20Style-PEP%208-yellow.svg)](https://peps.python.org/pep-0008/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)
[![GitHub stars](https://img.shields.io/github/stars/mathMing/AI4C.svg)](https://github.com/mathMing/AI4C/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/mathMing/AI4C.svg)](https://github.com/mathMing/AI4C/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/mathMing/AI4C.svg)](https://github.com/mathMing/AI4C)

---

## 📌 项目简介

**瞳观城康**（TongGuan ChengKang）面向 **基层社区** 与 **城市公卫** 两个使用场景，构建"眼底一张图 → 多病种早筛 → 区域驾驶舱"的一体化产品。

- 🩺 **基层筛查工作台**：社区医生上传眼底影像 → AI 秒级返回 DR（糖尿病视网膜病变）5 级分级 + Grad-CAM 可解释热力图 + 心血管 / 高血压 / 慢性肾病三联风险 → 一键入档、生成 PDF 报告。
- 🏙 **区域眼健康驾驶舱**：10 社区筛查数据汇总，提供 KPI、DR 分级饼图、疾病谱堆叠、转诊建议、社区排名等可视化。
- 🧑‍⚕️ **患者随访档案**：同一患者多次筛查趋势、跨次对比、转诊建议、增量录入。
- 🛰 **城市数字孪生**：基于地理坐标的 3D 城市—社区—患者三层下钻（Plotly 3D）。
- ⚙ **性能监控**：在线展示后端推理延迟、吞吐、错误率（实时仪表盘）。
- 🟣 **多病种联合诊断 v2.0**：DR + 青光眼 + AMD 一次出（5 类）。
- 🧪 **图像质控**：离线亮度 / 模糊 / 视野圆检测，不合格图直接提醒。
- 📄 **PDF/CSV 报告导出**：单次筛查 PDF、批量 CSV 包、区域数据 CSV。

**核心算法** 基于 **RETFound** 视网膜基础模型（ViT-L/16，ImageNet-21k + 1.6M 眼底图自监督预训练），并融合 **RetiZero** 零样本分类能力以及 **AutoMorph** 形态学特征管线。

---

## 🖼 项目截图

> 提交 PR 时欢迎附图；本地项目封面图待上传（参见下方"贡献指南"）。

| 基层筛查工作台 | 区域眼健康驾驶舱 | 城市数字孪生 |
|:-:|:-:|:-:|
| *TODO* | *TODO* | *TODO* |

---

## ✨ 功能亮点

| 模块 | 关键能力 |
|------|---------|
| 📤 **基层筛查工作台** | 单张精看 / 批量流水线（图像质控 → AI 诊断 → Grad-CAM → 入库 → PDF） |
| 🏙 **区域眼健康驾驶舱** | 10 社区 KPI · DR 分级饼图 · 疾病谱堆叠 · 转诊建议 · 风险排名 |
| 📋 **患者随访档案** | DR 趋势折线 · 三联风险曲线 · 详情表 · 转诊建议 · 增量录入 |
| 🛰 **城市数字孪生** | Plotly 3D 城市—社区—患者三层下钻 + 热力圈层 |
| ⚙ **性能监控** | 后端延迟 / 吞吐 / 错误率实时仪表盘 |
| 🟣 **多病种联合诊断 v2.0** | DR + 青光眼 + AMD 一次出（5 类） |
| 🧪 **图像质控** | 离线亮度 / 模糊 / 视野圆检测，不合格图直接提醒 |
| 📄 **PDF/CSV 报告导出** | 单次筛查 PDF、批量 CSV 包、区域数据 CSV |

---

## 🗂 仓库结构

```
AI4C/
├── README.md                    # 本文档
├── LICENSE                      # Research-Only 许可声明
├── CONTRIBUTING.md              # 贡献指南
├── CHANGELOG.md                 # 变更日志
├── .gitignore                   # 忽略大文件、日志、临时脚本
├── .gitattributes                # LFS / 文本归一化
├── init_db.sql                  # SQLite 表结构（CLI 初始化版）
│
├── frontend/                    # B 同学工程线：Streamlit 多页面 Web
│   ├── app.py                   # 主入口 / 路由 / 首页 Hero
│   ├── init_db.py               # 演示数据初始化脚本
│   ├── mock_backend.py          # 模拟 FastAPI 后端（A 同学未就绪时使用）
│   ├── requirements.txt         # 前端依赖
│   ├── README.md                # 前端详细文档
│   ├── pages/                   # Streamlit 多页面
│   │   ├── 1_筛查工作台.py
│   │   ├── 2_区域驾驶舱.py
│   │   ├── 3_患者随访.py
│   │   ├── 4_数字孪生.py
│   │   └── 5_性能监控.py
│   ├── utils/                   # 工具模块
│   │   ├── api_client.py        # FastAPI 调用封装
│   │   ├── db.py                # SQLite 随访数据库
│   │   ├── data_gen.py          # 区域模拟数据生成
│   │   ├── geo.py               # 城市—社区—患者地理坐标
│   │   ├── image_qc.py          # 离线图像质控
│   │   ├── page_header.py       # 统一页头 / 页脚 / 导航
│   │   ├── performance.py       # 性能指标采集
│   │   ├── region_3d.py         # 3D 数字孪生
│   │   ├── report.py            # PDF / CSV 报告
│   │   └── theme.py             # 主题色 / 样式注入
│   ├── assets/                  # 字体与图片资源
│   ├── data/                    # 运行时 SQLite 库（已 .gitignore）
│   └── mock_uploads/            # 热力图缓存（已 .gitignore）
│
├── backend/                     # A 同学算法线：真模型 FastAPI 服务
│   ├── backend.py               # FastAPI 入口（/api/predict, /api/gradcam…）
│   ├── retfound_model.py        # RETFound 加载 + 推理
│   ├── gradcam.py               # Grad-CAM 可视化
│   ├── inspect_ckpt.py          # ckpt 健康度检查
│   ├── test_backend.py          # 后端 smoke test
│   ├── requirements.txt         # 后端依赖
│   ├── models/                  # 模型构建 / 权重（已 .gitignore）
│   └── bridge/                  # 桥接脚本（mock ↔ real）
│
├── algo/                        # A 同学训练线（详见 algo/ALGO_README.md）
│   ├── train.py                 # 训练主入口（AMP + cosine + QWK 选 best）
│   ├── evaluate.py              # 评估脚本
│   ├── prepare_data.py          # DDR 数据集准备
│   ├── prepare_ddr.py           # DDR 切分 / 索引
│   ├── prepare_ddr_for_automorph.py  # 给 AutoMorph 喂数据
│   ├── local_demo.py            # 本地 demo
│   ├── stage2_fusion.py         # 多模态融合
│   ├── run_automorph_full.sh    # AutoMorph 完整流程
│   ├── configs/                 # 训练配置（yaml）
│   ├── datasets/                # 数据集定义
│   ├── models/                  # 训练用模型定义
│   ├── transforms/              # 数据增强
│   └── utils/                   # 训练工具（Logger / Metrics…）
│
└── scripts/                     # 工具脚本
    └── check_env.py             # 环境自检
```

> **未进仓库**（见 `.gitignore`）：`best.lora.pt`、`RetiZero.pth`、`AutoMorph-main.zip` 等大文件 / 压缩包 / 模型权重；`*.log`、`logs/`、临时测试脚本（`test_*.js/bat/sh`）以及一键启停脚本。这些保留在本地，clone 仓库即可。

---

## 🚀 快速开始

### 0. 克隆仓库

```bash
git clone https://github.com/mathMing/AI4C.git
cd AI4C
```

### 1. 准备环境

| 组件 | Python | 建议环境 |
|------|--------|---------|
| 前端 (frontend) | 3.10 | `conda create -n tongguan_web python=3.10 -y` |
| 后端 (backend) | 3.12 + CUDA | 建议 PyTorch 2.x + CUDA 12.x |
| 训练 (algo) | 3.10 | 显存 ≥ 16GB |

### 2. 启动模型后端（任选其一）

#### 选项 A：模拟后端（推荐，A 同学模型未就绪 / 离线演示时使用）

```bash
cd frontend
pip install -r requirements.txt
uvicorn mock_backend:app --reload --port 8000
```

- 健康检查：<http://localhost:8000/api/health>
- 接口文档：<http://localhost:8000/docs>

#### 选项 B：真模型后端（RETFound）

```bash
# 先把权重放到 backend/models/（仓库未跟踪大文件，详见下方「模型权重」）
cd backend
pip install -r requirements.txt
uvicorn backend:app --host 0.0.0.0 --port 8000
```

### 3. 启动前端

```bash
cd frontend
streamlit run app.py
```

浏览器访问 <http://localhost:8501>，左侧导航：

1. **基层筛查工作台** — 上传眼底图 → AI 诊断 + Grad-CAM
2. **区域眼健康驾驶舱** — 城市 KPI 大屏
3. **患者随访档案** — 趋势 + 转诊
4. **城市数字孪生** — 3D 下钻
5. **性能监控** — 实时延迟 / 吞吐

### 4. 切换到真模型 API

修改 `frontend/.streamlit/secrets.toml`（参考 `secrets.toml.example`）：

```toml
[api]
url = "http://localhost:8000"   # 改成真后端地址
```

或环境变量：

```bash
# Windows PowerShell
$env:TONGGUAN_API_URL = "http://localhost:8000"

# Linux / macOS
export TONGGUAN_API_URL=http://localhost:8000
```

> `utils/api_client.py` 已封装好三种来源优先级：环境变量 > `secrets.toml` > 默认值，**前端代码无需改动**。

---

## 🔌 API 约定（前后端契约）

### `POST /api/predict`

**请求**：`multipart/form-data`，字段名 `file`（图像）  
**响应**：

```json
{
  "diagnosis": "中度NPDR",
  "confidence": 0.8732,
  "dr_grade": 2,
  "gradcam_url": "/api/gradcam/abc123.jpg",
  "all_probs": {
    "无DR": 0.05, "轻度NPDR": 0.10,
    "中度NPDR": 0.87, "重度NPDR": 0.05, "增殖期DR": 0.03
  },
  "cv_risk": "中",
  "htn_risk": "中",
  "ckd_risk": "中",
  "model_version": "RETFound-finetuned-v1",
  "timestamp": "2025-10-08 12:30:00"
}
```

### `GET /api/gradcam/{filename}`

返回 Grad-CAM 热力图（JPG）。

### `GET /api/health`

健康检查 + GPU 信息 + 当前模型名。

### `GET /api/labels`

返回 5 类 DR 标签字典。

完整字段说明见 [`backend/backend.py`](./backend/backend.py) 与 [`frontend/utils/api_client.py`](./frontend/utils/api_client.py)。

---

## 🧠 模型权重（大文件说明）

| 文件 | 来源 | 大小 | 备注 |
|------|------|------|------|
| `RetiZero.pth` | RetiZero 官方发布 | ~1.6 GB | **不在仓库**，需手动放入 `backend/models/` |
| `best.lora.pt` | 本项目 LoRA 微调产出 | ~12 MB | **不在仓库**，训练完成后自行放置 |
| `AutoMorph-main.zip` | AutoMorph 官方发布 | ~1.6 GB | **不在仓库**，仅作数据准备参考 |

> GitHub 单文件 ≤ 100 MB，仓库建议 ≤ 1 GB。`.gitignore` 已将 `*.pt / *.pth / *.zip` 全部排除。
> 注意：仓库**当前没有**直接 commit 任何模型权重（`runs/`、`best.lora.pt` 等均被排除）。`ALGO_README.md` 中 "tracked in git" 的描述为旧版本残留，已在本仓库与 `.gitignore` 一致后修正。

**下载 / 放置位置**：

```bash
# 1) RetiZero（官方）
# https://github.com/...  →  下载到 backend/models/RetiZero.pth

# 2) RETFound 官方权重
# https://github.com/rmaphoh/RETFound_MAE  →  下载到 backend/models/retfound_v1.pth

# 3) 本项目 LoRA（用 algo/train.py 训练得到）
python algo/train.py --config algo/configs/retfound_b16.yaml
# 产出 best.lora.pt → backend/models/best.lora.pt
```

---

## 🩺 业务规则（演示版）

| 风险类型 | 等级 | 规则 |
|---------|------|------|
| 心血管 (cv) | 低 / 中 / 高 | DR ≤ 1 倾向低风险；DR ≥ 3 倾向高风险 |
| 高血压 (htn) | 低 / 中 / 高 | 同上 |
| 慢性肾病 (ckd) | 低 / 中 / 高 / 不适用 | DR < 2 视为"不适用"（非糖尿病） |

> 真实部署需结合年龄、性别、病史等结构化字段做联合建模。

---

## 🛠 技术栈

**前端** · Streamlit 1.30+ · Plotly 5.18+ · Pandas · ReportLab · Pillow  
**后端** · FastAPI 0.110+ · Uvicorn · SlowAPI（限流）· Pillow · OpenCV-headless  
**模型** · PyTorch 2.0+ · Torchvision · timm · Transformers · grad-cam · ttach  
**训练** · AMP 混合精度 · AdamW + cosine + warmup · QWK 选 best · 早停  
**数据** · SQLite（随访档案）· DDR / EyePACS（训练）· AutoMorph 形态学特征  
**可解释** · Grad-CAM 热力图 · 离线图像质控（亮度 / 模糊 / 视野圆）

---

## 🧪 本地自检

```bash
# 环境 / 依赖
python scripts/check_env.py

# 后端 smoke
python backend/test_backend.py

# 端到端
# 终端 1：cd frontend && uvicorn mock_backend:app --port 8000
# 终端 2：cd frontend && streamlit run app.py
# 浏览器：访问 8501，上传任意图查看 AI 诊断 + Grad-CAM
```

---

## 👥 团队分工

| 角色 | 负责 | 目录 |
|------|------|------|
| **A 同学（算法线）** | RETFound 推理 / 微调 · Grad-CAM · RetiZero 零样本 · AutoMorph 特征 · FastAPI 模型服务 | `backend/` · `algo/` |
| **B 同学（工程线）** | 模拟后端 · Streamlit 五页面 Web · SQLite 随访库 · 城市驾驶舱 · 3D 数字孪生 · 文档与部署脚本 | `frontend/` |

> 欢迎更多同学加入（详见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)）。

---

## 📊 项目数据（演示）

| 指标 | 数值 |
|------|------|
| 训练数据 | DDR 5 类 DR 分级 |
| 模型规模 | RETFound ViT-L/16 (304M) / LoRA (~12MB) |
| 推理延迟 | mock < 50ms / 真模型 ≤ 300ms (单卡 4090) |
| 演示规模 | 10 社区 / 1000+ 模拟患者 |

---

## 🗺 路线图

- [x] 0.x：DR 单病种 5 级分级 + Streamlit 工作台
- [x] 1.0：基层 → 区域 → 城市三层驾驶舱 + 数字孪生
- [x] 1.1：多病种联合诊断（DR + 青光眼 + AMD）
- [ ] 1.2：增量学习 / 联邦学习支持
- [ ] 1.3：临床结构化数据联合建模
- [ ] 2.0：基座模型升级 + 多模态（眼底+ OCT）

---

## 🤝 贡献指南

我们欢迎任何形式的贡献（bug 报告、功能建议、PR、Demo 案例）！请先阅读 [`CONTRIBUTING.md`](./CONTRIBUTING.md)：
- 提交 Issue 前请先搜索是否已有同类问题
- PR 请保持单一职责，并附上简要说明
- 代码风格遵循 PEP 8，关键函数请加 docstring

---

## 📜 许可与引用

- **本仓库代码**：仅供科研 / 比赛演示使用，**不作为临床诊断依据**。详见 [`LICENSE`](./LICENSE)。
- **RETFound**：Zhou et al., *A foundation model for generalizable disease detection from retinal images*, Nature 2023.
- **RetiZero**：Wang et al., 2024.
- **AutoMorph**：Zhou et al., 2022.

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@misc{ai4c2026,
  author = {mathMing and contributors},
  title  = {TongGuan ChengKang (AI4C): A Retinal-Foundation-Model Driven Platform for City-Wide Chronic Disease Screening},
  year   = {2026},
  howpublished = {GitHub repository},
  url    = {https://github.com/mathMing/AI4C}
}
```

---

## ⚠ 免责声明

本项目所有 AI 输出仅供科研与教学演示，**不可**作为任何医疗诊断、治疗或预防的依据。任何临床决策须由具有相应资质的医师作出。详见 [`LICENSE`](./LICENSE)。

---

<div align="center">

**用 ❤️ 与 🤖 建造  ·  Built with care for primary care**

[⬆ 回到顶部](#-瞳观城康--ai4c)

</div>
