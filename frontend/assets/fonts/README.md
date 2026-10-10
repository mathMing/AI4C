# 字体说明

本目录预留用于存放 ReportLab / Matplotlib 渲染中文所需的 TTF/TTC 字体。

## 当前实现策略：运行时探测 Windows 系统字体

为了不向仓库 commit 大体积二进制字体文件，`utils/report.py` 启动时会按以下顺序探测：

1. `C:/Windows/Fonts/msyh.ttc` — 微软雅黑（Windows 自带，最优先）
2. `C:/Windows/Fonts/simhei.ttf` — 黑体（Windows 自带，兜底）
3. `C:/Windows/Fonts/simsun.ttc` — 宋体
4. `assets/fonts/NotoSansSC-Regular.otf` — 若用户自行放置则使用

如果全部失败，**会降级为 ReportLab 默认 Helvetica 并打印警告**，中文会显示为方块，但功能不报错。

## 如果未来需要跨平台/无 GUI 环境

在 `assets/fonts/` 下放置 `NotoSansSC-Regular.otf` 或 `.ttf` 即可，代码会自动识别。
推荐来源：

- Google Noto Fonts：https://github.com/googlefonts/noto-cjk （License：SIL OFL 1.1）
- 思源黑体：https://github.com/adobe-fonts/source-han-sans

## 不 commit 二进制的原因

- 避免仓库体积膨胀（每个 TTF 通常 10MB+）
- Windows 用户环境已具备 msyh.ttc/simhei.ttf，无需重复
- 跨平台部署时再走构建脚本下载更合适