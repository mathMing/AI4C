# -*- coding: utf-8 -*-
"""
🏙 区域眼健康驾驶舱
===================
包装为"城市眼健康数字孪生大屏"：
- 顶部 KPI：总筛查量、异常检出率、高风险占比、覆盖社区
- 社区筛查量柱状图 + DR 分级饼图
- 社区疾病谱堆叠对比 + 转诊建议
- 趋势折线：DR 等级 vs 高血压风险
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_gen import generate_region_data
from utils.db import fetch_all_screenings
from utils.page_header import page_footer, page_header, section_divider, sidebar_nav
from utils.report import build_patient_list_csv, build_region_csv
from utils.theme import apply_plotly_theme, inject_theme

inject_theme()
st.set_page_config(page_title="区域眼健康驾驶舱", page_icon="🏙", layout="wide")

sidebar_nav()
page_header(
    "🏙",
    "区域眼健康驾驶舱",
    "从单个患者到整个城市，眼健康管理一屏掌握 —— 瞳观城康 · 数字孪生",
    breadcrumbs=("首页", "区域眼健康驾驶舱"),
    accent="#dc2626",
)
section_divider()

# ----------------------------- 数据源选择 ----------------------------- #
data_source = st.radio(
    "数据源",
    options=["真实入库数据 + 模拟补全", "全模拟演示"],
    horizontal=True,
    help="前端刚跑起来还没真实数据时，使用'全模拟演示'；有真实记录时优先用真实数据 + 模拟补全。",
)


def build_dataframe(real_df: pd.DataFrame) -> pd.DataFrame:
    """合并真实入库数据 + 模拟社区级数据。"""
    if data_source == "全模拟演示":
        return generate_region_data(seed=42)

    # 真实数据部分
    real_part = real_df.copy()
    if len(real_part) > 0:
        real_part["community"] = real_part["patient_id"].apply(
            lambda x: f"{(x or 'X')[0]}号社区" if isinstance(x, str) else "其他"
        )
        real_part["source"] = "真实"
    else:
        real_part = pd.DataFrame()

    sim_part = generate_region_data(seed=42)
    sim_part["source"] = "模拟"

    df = pd.concat([real_part, sim_part], ignore_index=True, sort=False)
    if "age" not in df.columns:
        df["age"] = np.random.randint(30, 80, size=len(df))
    return df


real_df = fetch_all_screenings()
df = build_dataframe(real_df)

# ----------------------------- 导出按钮组（三列）----------------------------- #
export_left, export_mid, export_right = st.columns([1, 1, 4])

with export_left:
    try:
        zip_bytes = build_region_csv(df)
        zip_name = f"region_data_{datetime.now().strftime('%Y%m%d')}.zip"
        st.download_button(
            label="📊 导出区域数据（CSV 打包）",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
            help="导出当前面板：社区 KPI / DR 分级 / 全身风险 三张 CSV，UTF-8 BOM，Excel 直接打开。",
        )
    except Exception as e:
        st.error(f"❌ CSV 生成失败：{e}")

with export_mid:
    has_real = data_source == "真实入库数据 + 模拟补全" and len(real_df) > 0
    try:
        csv_bytes = build_patient_list_csv(real_df if has_real else None)
        st.download_button(
            label="📋 导出真实患者清单（CSV）",
            data=csv_bytes,
            file_name=f"patient_list_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not has_real,
            help="仅从 SQLite 真实记录导出（不含模拟补全数据）。"
            if has_real
            else "请切换到「真实入库数据 + 模拟补全」，并至少写入一条筛查记录。",
        )
    except Exception as e:
        st.error(f"❌ CSV 生成失败：{e}")

with export_right:
    if has_real:
        st.caption(
            f"✅ 真实记录 {len(real_df)} 条，按最近筛查日期排序。"
            "不含模拟补全数据。"
        )
    else:
        st.caption(
            "⚠ 暂无真实记录。"
            "请先在「基层筛查工作台」或「患者随访」页面写入筛查数据。"
        )

# ----------------------------- 顶部 KPI ----------------------------- #
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("总筛查量", f"{len(df):,}")
k2.metric("异常检出率", f"{(df['dr_grade'] > 0).mean():.1%}")
k3.metric("心血管高风险占比", f"{(df['cv_risk'] == '高').mean():.1%}")
k4.metric("高血压高风险占比", f"{(df['htn_risk'] == '高').mean():.1%}")
k5.metric("覆盖社区数", f"{df['community'].nunique()}")

st.divider()

# ----------------------------- 第一行图：筛查量 + DR 分级饼图 ----------------------------- #
left, right = st.columns(2)
with left:
    st.subheader("📊 各社区筛查量")
    comm_stats = df.groupby("community").size().reset_index(name="count")
    fig = px.bar(
        comm_stats,
        x="community",
        y="count",
        color="count",
        color_continuous_scale="Blues",
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🥧 DR 分级分布")
    dr_counts = df["dr_grade"].value_counts().sort_index()
    fig = px.pie(
        values=dr_counts.values,
        names=[f"{i}级" for i in dr_counts.index],
        color_discrete_sequence=px.colors.sequential.Reds_r,
        hole=0.4,
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------- 第二行图：社区疾病谱 + 转诊 ----------------------------- #
left2, right2 = st.columns(2)
with left2:
    st.subheader("🏘 社区疾病谱对比（堆叠百分比）")
    community_dr = df.groupby(["community", "dr_grade"]).size().unstack(fill_value=0)
    pct = community_dr.div(community_dr.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for grade in pct.columns:
        fig.add_trace(
            go.Bar(
                name=f"DR {grade}级",
                x=pct.index,
                y=pct[grade],
                hovertemplate="%{x} · DR %{grade}级 · %{y:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.15),
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

with right2:
    st.subheader("⚠ 转诊建议")
    high = df[df["dr_grade"] >= 3]
    if len(high) > 0:
        st.warning(f"发现 **{len(high)}** 例需紧急转诊的患者（DR ≥ 3 级）")
        st.dataframe(
            high.groupby("community").size().reset_index(name="需转诊人数"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("暂无紧急转诊需求")

    # 全身风险社区排名
    cv_high = (
        df[df["cv_risk"] == "高"]
        .groupby("community")
        .size()
        .reset_index(name="心血管高风险数")
        .sort_values("心血管高风险数", ascending=False)
    )
    st.markdown("##### 🫀 心血管高风险社区排名")
    if len(cv_high) > 0:
        st.dataframe(cv_high, use_container_width=True, hide_index=True)
    else:
        st.info("暂无心血管高风险记录")

st.divider()

# ----------------------------- 第三行图：趋势折线 ----------------------------- #
st.subheader("📈 整体 DR 分级 vs 心血管风险 交叉趋势")

# 把 CV risk 映射为数字
cv_to_num = {"低": 1, "中": 2, "高": 3}
trend_df = df.copy()
trend_df["cv_risk_num"] = trend_df["cv_risk"].map(cv_to_num).fillna(0)

# 按 community 聚合均值
agg = (
    trend_df.groupby("community")
    .agg(avg_dr=("dr_grade", "mean"), avg_cv=("cv_risk_num", "mean"))
    .reset_index()
    .sort_values("avg_dr", ascending=False)
)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=agg["community"],
        y=agg["avg_dr"],
        name="平均 DR 等级",
        marker_color="indianred",
        yaxis="y",
    )
)
fig.add_trace(
    go.Scatter(
        x=agg["community"],
        y=agg["avg_cv"],
        name="平均心血管风险（1-3）",
        mode="lines+markers",
        marker=dict(size=10),
        line=dict(color="royalblue", width=3),
        yaxis="y2",
    )
)
fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="平均 DR 等级", range=[0, 4.5]),
    yaxis2=dict(title="心血管风险（1-3）", overlaying="y", side="right", range=[0, 3.5]),
    legend=dict(orientation="h", y=-0.2),
)
apply_plotly_theme(fig)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------- 数字孪生入口 ----------------------- #
st.info(
    "🌐 想看 3D 城市地图？"
    "**左侧栏 → 「🏙 数字孪生 3D 城市地图」**"
    "（每根 3D 柱代表一个社区，柱高=风险强度）"
)

page_footer("瞳观城康 | 数据为模拟生成 + 真实入库合并，仅供演示")