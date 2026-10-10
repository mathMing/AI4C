# -*- coding: utf-8 -*-
"""
? 性能监控 Dashboard
=====================
实时观察模型服务的健康度与吞吐：
- 5 个 KPI：TPS / p50 / p95 / 成功率 / 累计调用
- 60 秒 TPS + 平均延迟 双轴折线
- DR 等级分布柱状图
- 最近 20 次调用明细表
- 自动每 2 秒刷新

数据源：后端 /api/metrics（mock_backend.py 实现）
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.page_header import page_footer, page_header, section_divider, sidebar_nav
from utils.performance import (
    fetch_metrics,
    grade_color,
    grade_label,
    last_calls_to_dataframe,
    metrics_to_dataframe,
)
from utils.theme import apply_plotly_theme, inject_theme

inject_theme()
st.set_page_config(
    page_title="性能监控 Dashboard",
    page_icon="?",
    layout="wide",
)

sidebar_nav()
page_header(
    "?",
    "性能监控 Dashboard",
    "模型服务的实时 TPS / 延迟 / 成功率 / 错误分布 —— 答辩现场的工程线加分页",
    breadcrumbs=("首页", "性能监控"),
    accent="#0ea5e9",
)
section_divider()


# ====================================================================== #
# 控制条
# ====================================================================== #
ctrl_l, ctrl_r = st.columns([3, 1])
with ctrl_l:
    auto_refresh = st.toggle("? 自动刷新（每 2 秒）", value=True)
with ctrl_r:
    manual = st.button("? 立即刷新", use_container_width=True)

# 触发刷新（自增 session_state 计数，rerun 重新拉数据）
if "perf_tick" not in st.session_state:
    st.session_state["perf_tick"] = 0
if manual:
    st.session_state["perf_tick"] += 1
if auto_refresh:
    import time as _t
    _t.sleep(2)
    st.session_state["perf_tick"] += 1
    st.rerun()

# 抓取一次
data = fetch_metrics()
offline = data.get("_offline", False)

if offline:
    st.error(
        f"? 无法连接后端 API：{data.get('_error', '—')}  \n\n"
        f"请先启动 `uvicorn mock_backend:app --port 8000`"
    )
    page_footer("瞳观城康 | 性能监控 · 研究生智慧城市比赛 · 智慧医疗赛道")
    st.stop()

# 头部服务信息
hdr1, hdr2, hdr3 = st.columns(3)
with hdr1:
    st.success(f"? **{data.get('service', '—')}** · 在线")
with hdr2:
    st.caption(f"启动时长：**{data.get('uptime_sec', 0)}** 秒")
with hdr3:
    st.caption(f"采样时间：**{data.get('ts', '—')}**")

section_divider()


# ====================================================================== #
# 5 个 KPI 卡片
# ====================================================================== #
section_divider("? 核心指标", "?", color="#0ea5e9")
k1, k2, k3, k4, k5 = st.columns(5)

tps = data.get("tps", 0.0)
p50 = data.get("p50_ms", 0)
p95 = data.get("p95_ms", 0)
success_rate = data.get("success_rate", 0.0)
total_calls = data.get("total_calls", 0)
window_calls = data.get("window_calls", 0)

# TPS 用 delta（与上一帧对比，但简化：用窗口调用数变化量）
k1.metric("TPS（次/秒）", f"{tps:.2f}", delta=f"{window_calls} 次 / 60s", delta_color="off")
k2.metric("p50 延迟", f"{p50} ms")
k3.metric("p95 延迟", f"{p95} ms",
          delta="健康" if p95 < 1000 else "? 偏高",
          delta_color="normal" if p95 < 1000 else "inverse")
k4.metric("成功率", f"{success_rate * 100:.1f}%",
          delta="在线" if success_rate > 0.95 else "? 异常",
          delta_color="normal" if success_rate > 0.95 else "inverse")
k5.metric("累计调用", f"{total_calls}")

section_divider()


# ====================================================================== #
# 60 秒 TPS + 平均延迟 双轴折线
# ====================================================================== #
section_divider("? 实时吞吐 & 延迟（最近 60 秒）", "?", color="#0ea5e9")
df = metrics_to_dataframe(data.get("history", []))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["time"], y=df["calls"],
    name="TPS（每秒调用）",
    line=dict(color="#3b82f6", width=2),
    fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
    yaxis="y",
    mode="lines+markers",
    marker=dict(size=4),
))
fig.add_trace(go.Scatter(
    x=df["time"], y=df["avg_ms"],
    name="平均延迟 (ms)",
    line=dict(color="#dc2626", width=2, dash="dot"),
    yaxis="y2",
    mode="lines+markers",
    marker=dict(size=4),
))
fig.update_layout(
    xaxis=dict(title="时间", showgrid=False),
    yaxis=dict(title="调用次数 / 秒", side="left", showgrid=True),
    yaxis2=dict(title="延迟 (ms)", overlaying="y", side="right", showgrid=False),
    hovermode="x unified",
    height=360,
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
)
apply_plotly_theme(fig)
st.plotly_chart(fig, use_container_width=True)

section_divider()


# ====================================================================== #
# DR 等级分布 + 延迟分位对比
# ====================================================================== #
section_divider("? DR 等级分布 & 延迟分位", "?", color="#16a34a")
dist = data.get("dr_grade_dist", {0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
df_dist = pd.DataFrame([
    {"DR 等级": grade_label(g), "数量": dist.get(g, 0), "color": grade_color(g), "_g": g}
    for g in [0, 1, 2, 3, 4]
])

c1, c2 = st.columns(2)

with c1:
    fig = px.bar(
        df_dist,
        x="DR 等级", y="数量",
        color="DR 等级",
        color_discrete_map={row["DR 等级"]: row["color"] for _, row in df_dist.iterrows()},
        text="数量",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=320)
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # 延迟分位（p50 / p95 / p99 / avg / max）水平条形
    latencies = {
        "p50 (中位)": data.get("p50_ms", 0),
        "p95": data.get("p95_ms", 0),
        "p99": data.get("p99_ms", 0),
        "avg (平均)": data.get("avg_ms", 0),
        "max (最大)": data.get("max_ms", 0),
    }
    df_lat = pd.DataFrame([
        {"分位": k, "ms": v} for k, v in latencies.items()
    ])
    color_scale = ["#16a34a", "#84cc16", "#d97706", "#ea580c", "#dc2626"]
    fig = px.bar(
        df_lat, x="ms", y="分位", orientation="h",
        color="分位", color_discrete_sequence=color_scale,
        text="ms",
    )
    fig.update_traces(texttemplate="%{text} ms", textposition="outside")
    fig.update_layout(showlegend=False, height=320, yaxis=dict(categoryorder="array",
                                                                 categoryarray=list(latencies.keys())[::-1]))
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

section_divider()


# ====================================================================== #
# 最近 20 次调用明细
# ====================================================================== #
section_divider("? 最近 20 次调用明细", "?", color="#7c3aed")
df_last = last_calls_to_dataframe(data.get("last_calls", []))
if df_last.empty:
    st.info("暂无调用记录，去 ? 基层筛查工作台 跑一次 AI 诊断后这里就有数据了。")
else:
    # 美化：DR 等级加颜色标签
    df_last["DR 等级"] = df_last["dr_grade"].apply(grade_label)
    df_last["状态"] = df_last["status"].apply(
        lambda s: "? 成功" if s == "ok" else ("? 异常" if s == "bad_image" else "? 失败")
    )
    show_cols = ["ts", "latency_ms", "DR 等级", "状态", "model_version"]
    df_show = df_last[show_cols].rename(columns={
        "ts": "时间", "latency_ms": "耗时(ms)", "model_version": "模型版本",
    })
    st.dataframe(df_show, use_container_width=True, hide_index=True, height=420)

    # 导出 CSV
    csv = df_last.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="? 导出调用明细（CSV）",
        data=csv,
        file_name=f"api_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

section_divider()

# 解释
with st.expander("? 这些指标是怎么算的？"):
    st.markdown(
        """
        - **TPS**：最近 60 秒窗口内的调用数 ÷ 60
        - **p50 / p95 / p99**：50% / 95% / 99% 的请求都能在这个时间内完成（百分位延迟）
        - **成功率**：成功返回的请求占比
        - **DR 等级分布**：最近 60 秒内模型输出各级 DR 的次数
        - **环形缓冲**：后端保留最近 2000 条记录，超过则丢弃最早的
        - **自动刷新**：每 2 秒抓一次 `/api/metrics`，关掉开关可暂停

        答辩 Tips：
        - 切换到 ? 暗色模式看更酷炫
        - 跑到基层筛查工作台批量上传图片，这里 TPS 会立刻飙高
        - 截一张「TPS 飙高 + p95 平稳」的图，工程线直接拉满
        """
    )

page_footer("瞳观城康 | 性能监控 · 研究生智慧城市比赛 · 智慧医疗赛道")
