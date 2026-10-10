# 👁 瞳观城康 · B 同学工程线

> 基于视网膜基础模型的眼底—全身慢病早筛与城市健康管理平台  
> **研究生智慧城市比赛 · 智慧医疗赛道** · B 同学（工程线）前端代码

## ✨ 项目简介

瞳观城康以 **RETFound 视网膜基础模型** 为核心，提供：

- 📤 **基层筛查工作台**：上传眼底图 → AI 诊断 + Grad-CAM 热力图 → 一键入库
- 🏙 **区域眼健康驾驶舱**：城市级眼健康数字孪生大屏（Plotly 交互式图表）
- 📋 **患者随访档案**：多次筛查趋势 + 转诊建议 + 全身风险变化曲线

> 一张眼底照，洞察眼与全身；一个平台，连接社区与城市。

---

## 🗂 目录结构

```
frontend_B/
├── app.py                    # 主入口（首页/路由）
├── init_db.py                # 初始化脚本（建表 + 演示数据）
├── mock_backend.py           # 模拟后端（A 同学模型未就绪时使用）
├── requirements.txt          # Python 依赖
├── README.md                 # 本文档
├── pages/                    # Streamlit 多页面
│   ├── 1_筛查工作台.py
│   ├── 2_区域驾驶舱.py
│   └── 3_患者随访.py
├── utils/                    # 工具模块
│   ├── api_client.py         # 与模型 API 通信
│   ├── db.py                 # SQLite 随访数据库
│   ├── data_gen.py           # 区域模拟数据生成
│   └── report.py             # PDF / CSV 报告导出（reportlab + matplotlib）
├── data/                     # SQLite 数据库目录
└── mock_uploads/             # 模拟热力图缓存目录
```

---

## 🚀 快速启动

### 1. 环境准备

需要 **Python 3.10+**（推荐）。

```bash
# 创建并激活虚拟环境
conda create -n tongguan_web python=3.10 -y
conda activate tongguan_web

# 安装依赖
cd frontend_B
pip install -r requirements.txt
```

### 2. 启动模型服务（任选其一）

#### 选项 A：先跑模拟后端（推荐，A 同学未就绪时用）

```bash
# 终端 1：模拟 A 同学的 FastAPI 模型服务
cd frontend_B
uvicorn mock_backend:app --reload --port 8000
```

访问 `http://localhost:8000/docs` 看接口文档，`http://localhost:8000/api/health` 检查健康。

#### 选项 B：A 同学交付了真模型 API 后

修改 `frontend_B/.streamlit/secrets.toml`：

```toml
[api]
url = "http://A同学的IP:8000"
```

或设置环境变量：

```bash
export TONGGUAN_API_URL=http://192.168.1.100:8000
```

### 3. 初始化数据库 + 启动前端

```bash
# 终端 2：首次运行，插入演示数据
cd frontend_B
python init_db.py             # 普通模式：已有数据就跳过
python init_db.py --force     # 强制重置演示数据

# 启动 Streamlit
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

---

## 🔌 API 约定

### POST `/api/predict`

**请求**：`multipart/form-data`，字段名 `file`（图像文件）

**响应**（与 A 同学 FastAPI 完全对齐）：

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

### GET `/api/gradcam/{filename}`

返回 Grad-CAM 热力图 JPG 文件。

### GET `/api/health`

健康检查。

---

## 🧪 模拟后端说明（mock_backend.py）

为支持 A、B 同学并行开发，提供一个**完全兼容**的 FastAPI 模拟后端：

- 接口路径与字段命名 100% 对齐真模型 API
- DR 等级按真实流行病学分布随机抽样（无 DR 50% / 轻度 20% / 中度 15% / 重度 10% / 增殖 5%）
- 热力图自动合成（红黄渐变 + 224×224 占位），保证前端可视化效果
- 全身风险等级与 DR 等级联动

> 切换到真 API 时，只需修改 `secrets.toml` 里的 URL，**无需改动任何前端代码**。

---

## 🩺 业务亮点

| 模块 | 功能 |
|------|------|
| **筛查工作台** | 上传图像 → AI 诊断 + 置信度 + 概率分布 + 全身风险 + Grad-CAM 热力图 → 一键入库 |
| **区域驾驶舱** | 10 社区 KPI、社区筛查量柱状图、DR 分级饼图、社区疾病谱堆叠对比、转诊建议、心血管风险社区排名 |
| **患者随访** | DR 分级趋势折线、全身风险三联趋势、详情表、随访/转诊建议、新增筛查快捷入口 |
| **CSV 导出** | 首页与驾驶舱一键导出患者筛查记录 CSV / 区域驾驶舱数据打包 CSV |

### 全身风险映射规则（演示版）

| 风险类型 | 等级 | 规则 |
|---------|------|------|
| 心血管 | 低 / 中 / 高 | DR ≤ 1 低概率高风险；DR ≥ 3 高概率高风险 |
| 高血压 | 低 / 中 / 高 | 同上 |
| 慢性肾病 | 低 / 中 / 高 / 不适用 | DR < 2 视为"不适用"（非糖尿病） |

---

## 🛠 切换到生产模型 API

`utils/api_client.py` 已经做了完整封装：

```python
from utils.api_client import get_api_url, call_predict, health_check

url = get_api_url()             # 自动从 secrets.toml / 环境变量 / 默认值读取
health = health_check()         # 健康检查
result = call_predict(bytes, "fundus.jpg")  # 调 /api/predict
```

A 同学交付后只需：

1. 把真模型服务跑起来
2. 改 `.streamlit/secrets.toml` 的 `api.url`
3. 重启 Streamlit

---

## 📦 常见问题

**Q：Streamlit 中文显示乱码？**  
A：Streamlit 默认支持中文。如出现乱码，在文件开头加 `# -*- coding: utf-8 -*-`。

**Q：A 同学的 API 连不上？**  
A：先确认 A 的后端在跑（访问 `http://A的IP:8000/api/health`），再看防火墙 / 网络。最后回退到 `mock_backend.py` 调试前端。

**Q：数据库在哪？**  
A：`frontend_B/data/patient_records.db`。删掉这个文件再跑 `python init_db.py --force` 即可重建。

---

## 👥 团队分工

- **A 同学（算法线）**：RETFound 推理 / 微调、Grad-CAM、RetiZero 零样本、AutoMorph 特征提取、FastAPI 模型服务
- **B 同学（工程线，本仓库）**：模拟后端、Streamlit 三页面 Web 工作台、SQLite 随访数据库、城市驾驶舱、文档与部署脚本

---

⚠ **本系统仅供科研演示，不作为临床诊断依据。**