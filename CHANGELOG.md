# 📝 变更日志 · Changelog

本项目所有 **用户可见** 的重要变更都记录在这里。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。

---

## [Unreleased]

### Added
- 📄 新增 `LICENSE`（Research-Only）
- 🤝 新增 `CONTRIBUTING.md`
- 📝 新增 `CHANGELOG.md`（本文件）
- 🖼 根 README 加入 6 类项目徽章（License / Python / Streamlit / FastAPI / PyTorch / stars / size）
- 🗺 根 README 加入「路线图」「项目数据」章节
- 🧩 补全 API 约定的返回字段说明、模型权重说明校正

### Changed
- ✏️ 校正 `ALGO_README.md` 中关于"模型权重 tracked in git"的旧描述
  （仓库实际未跟踪任何 .pt / .pth 权重，已与 `.gitignore` 一致）
- 📚 仓库结构图中明确各子目录的详细文档入口

### Fixed
- 🧹 根目录已无 `algo/` 副本残留（前序 commit `6e7a393` 已清理）

---

## [1.1.0] - 2026-10-12

### Added
- 🟣 多病种联合诊断 v2.0：DR + 青光眼 + AMD 一次出（5 类）
- 📋 患者随访档案：DR 趋势折线 · 三联风险曲线 · 详情表 · 转诊建议
- 🛰 城市数字孪生：Plotly 3D 城市—社区—患者三层下钻
- ⚙ 性能监控仪表盘：后端延迟 / 吞吐 / 错误率
- 🧪 离线图像质控：亮度 / 模糊 / 视野圆
- 📄 PDF / CSV 报告导出

### Changed
- 🏗 仓库结构重构为 `frontend / backend / algo` 三线 + `scripts / docs`
- 🪪 顶层 README 全面重写：徽章 / 简介 / 快速开始 / API 约定 / 团队分工

---

## [1.0.0] - 2026-10-10

### Added
- 🎉 瞳观城康（AI4C）首版开源
- 🩺 基层筛查工作台：单张精看 + 批量流水线
- 🏙 区域眼健康驾驶舱：10 社区 KPI · DR 分级饼图 · 疾病谱堆叠
- 🔌 FastAPI 模型后端（mock + 真模型两套）
- 🌐 Streamlit 多页面 Web（5 个页面）

---

## [0.1.0] - 2026-10-08

### Added
- 🏗 仓库初始化（Initial commit）

---

<!--
模板（后续版本请复制这一段并改日期）
## [X.Y.Z] - YYYY-MM-DD

### Added  - 新功能
### Changed - 已有功能 / 行为变化
### Deprecated - 即将移除
### Removed - 已移除
### Fixed   - 修 bug
### Security- 安全相关
-->
