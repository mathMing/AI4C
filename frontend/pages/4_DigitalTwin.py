# -*- coding: utf-8 -*-
"""
🏙 数字孪生 3D 城市地图
========================
基于 pydeck 的 3D 城市可视化：
- 每个社区是一根 3D 柱（高度 = 基础高度 × 风险系数）
- 颜色按心血管/高血压高风险占比：红 > 橙 > 蓝
- 上方叠加患者散点（按 DR 等级配色）
- 提供热力图 / 散点 / 柱状 三种图层，可单独开关
- 鼠标悬停柱体显示详情：社区 / 筛查数 / 平均 DR / 高风险人数占比

依赖：
    - pydeck          : 3D 地图渲染（需 pip install pydeck）
    - utils.geo       : 社区经纬度
    - utils.region_3d : 数据聚合 + 图层 spec
    - utils.data_gen  : 模拟数据
    - utils.db        : 真实数据
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from utils.data_gen import generate_region_data
from utils.db import fetch_all_screenings
from utils.geo import BEIJING_CENTER
from utils.page_header import page_footer, page_header, section_divider, sidebar_nav
from utils.region_3d import (
    aggregate_by_community,
    make_building_layer,
    make_heatmap_layer,
    make_patient_points,
    make_patient_scatter_layer,
)
from utils.theme import inject_theme

inject_theme()
st.set_page_config(
    page_title="数字孪生 3D 城市地图",
    page_icon="🏙",
    layout="wide",
)

sidebar_nav()
page_header(
    "🌐",
    "数字孪生 3D 城市地图",
    "北京 10 社区 · 眼健康风险全景 · 柱高=风险强度 · 颜色=高风险占比",
    breadcrumbs=("首页", "数字孪生 3D 城市地图"),
    accent="#16a34a",
)
section_divider()

# ----------------------- pydeck 依赖探测 ----------------------- #
try:
    import pydeck as pdk
    _HAS_PYDECK = True
except ImportError:
    _HAS_PYDECK = False

if not _HAS_PYDECK:
    st.error(
        "❌ 未安装 `pydeck`。请在 conda 环境中运行：\n\n"
        "```\npip install pydeck\n```\n\n"
        "安装后刷新本页面即可。"
    )
    st.stop()

# ----------------------- 数据源 ----------------------- #
data_source = st.radio(
    "数据源",
    options=["真实入库数据 + 模拟补全", "全模拟演示"],
    horizontal=True,
    key="dt_data_source",
)
real_df = fetch_all_screenings()

if data_source == "全模拟演示":
    df = generate_region_data(seed=42)
else:
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
        import numpy as np
        df["age"] = np.random.randint(30, 80, size=len(df))

# ----------------------- 图层控制 ----------------------- #
st.subheader("🎛 图层 & 视角控制")
ccol1, ccol2, ccol3, ccol4 = st.columns(4)
with ccol1:
    show_buildings = st.checkbox("🏢 社区 3D 柱", value=True, key="ly_b")
    show_heatmap = st.checkbox("🔥 风险热力图", value=False, key="ly_h")
    show_patients = st.checkbox("👤 患者散点（按 DR 配色）", value=True, key="ly_p")
with ccol2:
    zoom = st.slider("🔍 缩放", 8.0, 12.0, 9.5, 0.1)
    pitch = st.slider("📐 俯仰角", 0, 70, 50, 5)
    bearing = st.slider("🧭 朝向", -180, 180, -20, 5)
with ccol3:
    color_mode = st.selectbox(
        "柱体配色",
        ["风险等级（红/橙/蓝）", "平均 DR 等级", "筛查量"],
        index=0,
        key="cm",
    )
with ccol4:
    map_style = st.selectbox(
        "底图风格",
        ["dark", "light", "satellite", "streets"],
        index=0,
        key="ms",
        help="dark 答辩最炫；satellite 看真实街景；streets 偏白底",
    )

# ----------------------- 数据聚合 ----------------------- #
agg_df = aggregate_by_community(df, community_col="community")

# 按 color_mode 重新上色（覆盖 aggregate_by_community 默认的"风险等级"配色）
if color_mode == "平均 DR 等级":
    def _dr_color(row):
        m = row["mean_dr"]
        if m >= 3.0:   return [220, 50, 47, 220]
        if m >= 2.0:   return [255, 140, 0, 220]
        if m >= 1.0:   return [255, 220, 100, 220]
        return [38, 139, 210, 220]
    agg_df = agg_df.copy()
    agg_df["color"] = agg_df.apply(_dr_color, axis=1)
elif color_mode == "筛查量":
    agg_df = agg_df.copy()
    nmax = max(agg_df["n"].max(), 1)
    norm = agg_df["n"] / nmax
    agg_df["color"] = [
        [
            int(38 + (220 - 38) * v),
            int(139 + (50 - 139) * v),
            int(210 + (47 - 210) * v),
            220,
        ]
        for v in norm
    ]

# 给 tooltip 准备一个友好版的占比字段
agg_tooltip_df = agg_df.copy()
agg_tooltip_df["high_risk_pct_pct"] = (
    (agg_tooltip_df["high_risk_pct"] * 100).round(1).astype(str) + "%"
)

# ----------------------- 患者散点（按需生成）----------------------- #
need_pts = show_patients or show_heatmap
pts_df = (
    make_patient_points(df, community_col="community", seed=42)
    if need_pts else pd.DataFrame(columns=["lon", "lat", "dr_grade", "cv_risk", "color"])
)

# ---------------------- 构建图层（顺序：柱 → 散点 → 热力图）----------------------- #
layers: list = []
if show_buildings:
    layers.append(pdk.Layer(**make_building_layer(agg_tooltip_df)))
if show_patients:
    layers.append(pdk.Layer(**make_patient_scatter_layer(pts_df)))
if show_heatmap:
    layers.append(pdk.Layer(**make_heatmap_layer(pts_df)))

# ----------------------- 渲染地图 ----------------------- #
view_state = pdk.ViewState(
    longitude=BEIJING_CENTER[0],
    latitude=BEIJING_CENTER[1],
    zoom=zoom,
    pitch=pitch,
    bearing=bearing,
)
tooltip = {
    "html": (
        "<b>{community}</b><br/>"
        "📊 筛查数: {n}<br/>"
        "🎯 平均 DR: {mean_dr}<br/>"
        "⚠ 高风险人数: {high_risk_n}<br/>"
        "📈 高风险占比: {high_risk_pct_pct}"
    ),
    "style": {"backgroundColor": "#0b1f44", "color": "white", "fontSize": "13px"},
}

deck = pdk.Deck(
    map_style=map_style,
    initial_view_state=view_state,
    layers=layers,
    tooltip=tooltip,
)

st.pydeck_chart(deck, use_container_width=True, height=620)

# ----------------------- 下方：图例 + 数据表 ----------------------- #
st.divider()
legend_l, table_r = st.columns([1, 2])

with legend_l:
    st.markdown("##### 🎨 图例")
    st.markdown(
        """
        **柱体颜色（风险等级模式）**
        - 🔴 红：高风险占比 ≥ 20%
        - 🟠 橙：8% ≤ 高风险占比 < 20%
        - 🔵 蓝：高风险占比 < 8%

        **散点颜色（按 DR 等级）**
        - 🔴 红：DR 4 级（增殖期）
        - 🟠 橙：DR 3 级（重度）
        - 🟡 黄：DR 2 级（中度）
        - 🔵 浅蓝：DR 1 级（轻度）
        - 🔵 深蓝：DR 0 级（无 DR）

        **柱高含义**
        - 基础高度 = 社区人口密度（演示值）
        - 风险系数 = 1.0 + 高风险占比×5 + 平均 DR/4
        """
    )

with table_r:
    st.markdown("##### 📋 社区聚合数据（驱动 3D 地图）")
    show_df = agg_tooltip_df[[
        "community", "n", "mean_dr", "high_risk_n", "high_risk_pct_pct", "elevation"
    ]].rename(columns={
        "community": "社区",
        "n": "筛查数",
        "mean_dr": "平均 DR",
        "high_risk_n": "高风险人数",
        "high_risk_pct_pct": "高风险占比",
        "elevation": "柱高",
    })
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    csv = show_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 导出 3D 地图数据（CSV）",
        data=csv,
        file_name=f"3D城市地图-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=False,
    )

# ----------------------- 底部说明 ----------------------- #
st.divider()
with st.expander("ℹ️ 地图数据说明"):
    st.markdown(
        """
        - **10 社区坐标**：基于北京二环-五环近似分布，**非真实地理位置**，仅作科研演示。
        - **柱体高度**：基础高度 × 风险系数（高风险占比 + 平均 DR 等级驱动）。
        - **颜色**：默认按"心血管+高血压高风险 + DR≥3 占比"分三档；可切换为"平均 DR"或"筛查量"配色。
        - **散点**：每位患者围绕社区中心做 ±0.003°（约 300m）随机抖动。
        - **底图风格**：dark 适合答辩；satellite 看真实街景。
        - **真实数据**：从 SQLite `screenings` 表读 `patient_id` 首字符映射到社区。

        ⚠ 本系统仅供科研演示，不作为临床诊断依据。
        """
    )

page_footer("瞳观城康 | 数字孪生 · 研究生智慧城市比赛 · 智慧医疗赛道")
