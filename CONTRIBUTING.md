# 🤝 贡献指南 · Contributing to AI4C

感谢你愿意让 **瞳观城康 / AI4C** 变得更好！我们欢迎任何形式的贡献：bug 报告、功能建议、
文档改进、PR、Demo 案例、Issue 中的讨论与思考。

> 提交贡献即视为你已阅读本指南、[`LICENSE`](./LICENSE) 与 [`README.md`](./README.md)，
> 并同意你的贡献以相同的 Research-Only 许可发布。

---

## 📋 提 Issue

提交 Issue 前请先 **搜索** 仓库是否已有同类问题；如确认是新问题，请：

1. **Bug 报告**：提供复现步骤、期望行为、实际行为、运行环境（OS / Python / GPU / 显存）、
   关键日志与截图。
2. **功能建议**：说明痛点、目标用户、你设想的交互或伪代码（可选）、是否愿意提交 PR。
3. **文档改进**：直接指出章节、错别字 / 含糊处 / 缺图。

> **安全相关** 问题请勿在公开 Issue 提，请直接联系仓库 Owner。

---

## 🔧 提 PR

### 基本流程

```bash
# 1. fork & clone
git clone https://github.com/<your-fork>/AI4C.git
cd AI4C

# 2. 切分支
git checkout -b feat/<short-name>     # 新功能
git checkout -b fix/<short-name>      # 修 bug
git checkout -b docs/<short-name>     # 仅文档

# 3. 改代码，确保本地能跑通
python scripts/check_env.py
python backend/test_backend.py        # 若改后端

# 4. 提交
git add -A
git commit -m "feat(frontend): xxx"
git push origin feat/<short-name>

# 5. 在 GitHub 上发起 PR
```

### Commit 信息风格

我们使用 **Conventional Commits**（与本仓库历史一致）：

```
<type>(<scope>): <subject>

<body (optional)>

<footer (optional)>
```

`type` 可选：`feat / fix / docs / style / refactor / perf / test / chore`  
`scope` 可选：`frontend / backend / algo / docs / ci` 等  
`subject` 用动词现在时，不超过 50 字，中英文皆可（建议英文）。

### 代码风格

- Python 遵循 **PEP 8**；4 空格缩进；UTF-8；LF 换行。
- 关键函数请加 **docstring**（中文或英文均可）。
- 提交前请确保 `python -m py_compile <changed_files>` 不报错。
- 不要 commit 大文件（`.pt / .pth / .zip / data/`）—— 仓库 `.gitignore` 已设防。

---

## 🧪 自我检查清单

PR 合并前请确认：

- [ ] 代码可在本地 `python scripts/check_env.py` 通过
- [ ] 改动的关键路径附了最小复现 / 单测
- [ ] 没引入新的 `print()` 调试输出（用 `utils/logger.py` 替代）
- [ ] 没引入新的第三方依赖（或在 PR 描述中说明）
- [ ] 文档同步更新（README / 对应子目录的 README）
- [ ] PR 描述清楚说明：动机 / 改动点 / 测试方式 / 截图（如有 UI）

---

## 🌟 第一次参与？推荐这些

- 🐛 找标签 [`good first issue`](https://github.com/mathMing/AI4C/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- 📖 改文档：发现拼写错误、错别字、表述不清之处直接 PR
- 🎨 美化：替换或新增 README / 子目录的截图、Demo 视频

---

## 🗣 社区准则

请保持 **友善、尊重、建设性**。我们禁止：

- 人身攻击、性别歧视、种族歧视、宗教歧视
- 发布与本项目无关的广告、垃圾信息
- 未经授权使用本项目进行商业宣传、医疗诊断宣称

任何违反上述准则的 Issue / PR / 评论将直接关闭，必要时将封禁账号。

---

## 📜 许可

你的贡献将 **以与本仓库相同的 Research-Only License 发布**（详见 [`LICENSE`](./LICENSE)）。
提交 PR 即表示你确认你拥有提交的代码 / 文档的版权，并同意这一许可条款。

---

<div align="center">

**Have fun & build something useful.  玩得开心，做点有用的事。**

</div>
