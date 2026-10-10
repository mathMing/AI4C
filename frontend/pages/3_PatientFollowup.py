# -*- coding: utf-8 -*-
"""
📋 患者随访档案
================
- 输入患者ID，调取该患者所有筛查记录
- 折线图展示 DR 分级变化趋势
- 表格展示每次筛查的详情
- 根据最新 DR 分级给出随访/转诊建议
- 提供"新增筛查"快捷按钮（演示场景）
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.db import add_screening, ensure_db, fetch_patient_history, list_patient_ids
from utils.page_header import info_card, page_footer, page_header, section_divider, sidebar_nav
from utils.report import build_patient_report
from utils.theme import apply_plotly_theme, inject_theme

ensure_db()
inject_theme()

st.set_page_config(page_title="患者随访档案", page_icon="📋", layout="wide")

sidebar_nav()
page_header(
    "📋",
    "患者随访档案",
    "筛查之后怎么办？—— 闭环随访，连接社区与城市",
    breadcrumbs=("首页", "患者随访档案"),
    accent="#7c3aed",
)
section_divider()
st.divider()

# ----------------------------- 侧栏：患者选择 ----------------------------- #
with st.sidebar:
    st.header("🔍 选择患者")
    ids = list_patient_ids()
    options = ["（输入新ID）"] + ids
    choice = st.selectbox("历史患者", options, ids.index("P001") + 1 if "P001" in ids else 0)
    if choice == "（输入新ID）":
        patient_id = st.text_input("患者ID", value="P001").strip()
    else:
        patient_id = choice

# 主体
if not patient_id:
    st.info("请输入或选择患者 ID。")
    st.stop()

df = fetch_patient_history(patient_id)

if len(df) == 0:
    st.warning(f"患者 **{patient_id}** 暂无筛查记录。")
    st.markdown("##### 🧪 为该患者插入一条模拟筛查记录（演示用）")
    with st.form(f"demo_add_{patient_id}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("姓名", value=f"演示患者_{patient_id}")
            dr_grade = st.selectbox("DR 分级", options=[0, 1, 2, 3, 4], index=0)
        with col2:
            cv_risk = st.selectbox("心血管风险", options=["低", "中", "高"], index=0)
            htn_risk = st.selectbox("高血压风险", options=["低", "中", "高"], index=0)
        with col3:
            ckd_risk = st.selectbox(
                "慢性肾病风险",
                options=["低", "中", "高", "不适用（非糖尿病患者）"],
                index=0,
            )
            diabetes = st.checkbox("糖尿病史", value=False)

        if st.form_submit_button("📥 插入模拟记录"):
            add_screening(
                patient_id=patient_id,
                patient_name=name,
                age=55,
                gender="男",
                diabetes=diabetes,
                dr_grade=int(dr_grade),
                cv_risk=cv_risk,
                htn_risk=htn_risk,
                ckd_risk=ckd_risk,
                diagnosis=f"DR {dr_grade} 级",
                confidence=round(float(np.random.uniform(0.75, 0.95)), 4),
                gradcam_path="",
            )
            st.success("已插入模拟记录，请刷新页面查看。")
            st.rerun()
    st.stop()

# ----------------------------- 概览 ----------------------------- #
patient_name = df["patient_name"].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("姓名", patient_name)
c2.metric("累计筛查", len(df))
latest = df.iloc[-1]
c3.metric("最新 DR 分级", int(latest["dr_grade"]))
c4.metric("最新心血管风险", latest["cv_risk"])

# 多病种联合卡片（v2.0）
g1, g2, g3, g4 = st.columns(4)
g1.metric("最新青光眼", str(latest.get("glaucoma_risk", "低风险")))
g2.metric("青光眼概率", f"{float(latest.get('glaucoma_prob', 0)):.0%}")
g3.metric("最新 AMD", str(latest.get("amd_risk", "无")))
g4.metric("AMD 概率", f"{float(latest.get('amd_prob', 0)):.0%}")

# 联合提示横幅（v2.0 复用）
dr_grade = int(latest["dr_grade"])
glaucoma_risk = str(latest.get("glaucoma_risk", "低风险"))
amd_risk = str(latest.get("amd_risk", "无"))
if dr_grade >= 3 or "高" in glaucoma_risk or amd_risk in ("中期", "晚期"):
    st.error(
        "🚨 **多病种联合提示**：最近一次筛查存在 ≥1 项需紧急转诊的指征，"
        "建议尽快到眼科专科进一步检查（OCT / 视野 / FFA）。"
    )
elif dr_grade >= 2 or "中" in glaucoma_risk or amd_risk == "早期":
    st.warning(
        "⚠ **多病种联合提示**：最近一次筛查存在 ≥1 项需关注的中度指征，"
        "建议 3-6 个月内复查。"
    )
else:
    st.success(
        "✅ **多病种联合提示**：三项均处于低风险，建议年度常规复查。"
    )

section_divider()

# ----------------------------- 趋势图 ----------------------------- #
st.subheader(f"📈 {patient_name} · DR 分级变化趋势")
fig = px.line(
    df,
    x="screening_date",
    y="dr_grade",
    markers=True,
    labels={"dr_grade": "DR 分级", "screening_date": "筛查日期"},
)
fig.update_yaxes(range=[-0.5, 4.5], dtick=1)
fig.update_traces(line=dict(color="indianred", width=3), marker=dict(size=10))
fig.update_layout(height=380)
apply_plotly_theme(fig)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------- 全身风险变化 ----------------------------- #
st.subheader("🫀 全身风险等级变化")
risk_to_num = {"低": 1, "中": 2, "高": 3, "不适用（非糖尿病患者）": 0}
df["cv_num"] = df["cv_risk"].map(risk_to_num).fillna(0)
df["htn_num"] = df["htn_risk"].map(risk_to_num).fillna(0)
df["ckd_num"] = df["ckd_risk"].map(risk_to_num).fillna(0)

import plotly.graph_objects as _go

fig2 = _go.Figure()
fig2.add_trace(
    _go.Scatter(
        x=df["screening_date"],
        y=df["cv_num"],
        mode="lines+markers",
        name="心血管风险",
        line=dict(color="royalblue", width=3),
        marker=dict(size=10),
    )
)
fig2.add_trace(
    _go.Scatter(
        x=df["screening_date"],
        y=df["htn_num"],
        mode="lines+markers",
        name="高血压风险",
        line=dict(color="darkorange", width=3),
        marker=dict(size=10),
    )
)
fig2.add_trace(
    _go.Scatter(
        x=df["screening_date"],
        y=df["ckd_num"],
        mode="lines+markers",
        name="慢性肾病风险",
        line=dict(color="seagreen", width=3),
        marker=dict(size=10),
    )
)
fig2.update_yaxes(
    tickvals=[0, 1, 2, 3],
    ticktext=["不适用", "低", "中", "高"],
    range=[-0.3, 3.5],
)
fig2.update_layout(height=380, hovermode="x unified")
apply_plotly_theme(fig2)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ----------------------------- 详情表 + 建议 ----------------------------- #
st.subheader("📜 全部筛查记录")
display_cols = [
    "screening_date",
    "diagnosis",
    "dr_grade",
    "cv_risk",
    "htn_risk",
    "ckd_risk",
    "glaucoma_risk",
    "glaucoma_prob",
    "amd_risk",
    "amd_prob",
    "confidence",
]
st.dataframe(
    df[display_cols].rename(
        columns={
            "screening_date": "筛查日期",
            "diagnosis": "诊断",
            "dr_grade": "DR 分级",
            "cv_risk": "心血管",
            "htn_risk": "高血压",
            "ckd_risk": "慢性肾病",
            "glaucoma_risk": "青光眼",
            "glaucoma_prob": "青光眼概率",
            "amd_risk": "AMD",
            "amd_prob": "AMD 概率",
            "confidence": "置信度",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# ---------------- 导出完整随访档案 PDF ---------------- #
st.divider()
st.markdown("##### 📄 一键导出随访档案")
try:
    pdf_bytes = build_patient_report(
        patient_df=df,
        patient_id=patient_id,
        patient_name=patient_name,
        model_version="mock-1.0",
    )
    safe_name = patient_name or "患者"
    file_name = f"随访档案-{safe_name}-{datetime.now().strftime('%Y%m%d')}.pdf"
    st.download_button(
        label="📄 导出完整随访档案（PDF）",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
        use_container_width=False,
    )
except Exception as e:
    st.error(f"❌ PDF 生成失败：{e}")

st.subheader("🩺 随访建议")
dr = int(latest["dr_grade"])
if dr >= 4:
    st.error("⚠ DR 4 级（增殖期），建议**立即转诊至眼科专科**，可能需激光光凝/玻切手术评估。")
    next_in = "1 个月"
elif dr == 3:
    st.error("⚠ DR 3 级（重度 NPDR），建议**尽快转诊眼科**，3 个月内复查。")
    next_in = "1 个月"
elif dr == 2:
    st.warning("中度 NPDR，建议**6 个月内复查**，并严格控制血糖/血压。")
    next_in = "6 个月"
elif dr == 1:
    st.info("轻度 NPDR，建议**12 个月内复查**。")
    next_in = "12 个月"
else:
    st.success("✅ 当前未见明显 DR 病变，建议**年度常规复查**。")
    next_in = "12 个月"

st.markdown(f"##### ⏰ 下次随访建议：**{next_in} 内**")

# ----------------------------- 一键新增筛查 ----------------------------- #
with st.expander("➕ 为该患者新增一条筛查记录（演示用）"):
    with st.form(f"add_{patient_id}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            dr_grade = st.selectbox(
                "DR 分级", options=[0, 1, 2, 3, 4], index=int(dr)
            )
            cv_risk = st.selectbox(
                "心血管风险", options=["低", "中", "高"], index=["低", "中", "高"].index(latest["cv_risk"]) if latest["cv_risk"] in ["低", "中", "高"] else 0,
            )
        with col2:
            htn_risk = st.selectbox(
                "高血压风险", options=["低", "中", "高"], index=0
            )
            ckd_risk = st.selectbox(
                "慢性肾病风险",
                options=["低", "中", "高", "不适用（非糖尿病患者）"],
                index=0,
            )
        with col3:
            confidence = st.slider("置信度", 0.5, 0.99, 0.85, 0.01)
        if st.form_submit_button("📥 保存"):
            add_screening(
                patient_id=patient_id,
                patient_name=patient_name,
                age=55,
                gender="男",
                diabetes=False,
                dr_grade=int(dr_grade),
                cv_risk=cv_risk,
                htn_risk=htn_risk,
                ckd_risk=ckd_risk,
                diagnosis=f"DR {dr_grade} 级",
                confidence=float(confidence),
                gradcam_path="",
            )
            st.success("已新增筛查记录。")
            st.rerun()

st.divider()
page_footer("瞳观城康 | 研究生智慧城市比赛 · 智慧医疗赛道 · B 同学工程线")