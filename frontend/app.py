# -*- coding: utf-8 -*-
"""
👁 瞳观城康 · 主入口
=====================
基于视网膜基础模型的眼底—全身慢病早筛与城市健康管理平台

副标题：
    一张眼底照，洞察眼与全身；
    一个平台，连接社区与城市。
"""
from __future__ import annotations

import datetime
import streamlit as st

from utils.api_client import get_api_url, health_check
from utils.db import ensure_db, fetch_all_screenings, seed_demo_data
from utils.page_header import info_card, page_footer, section_divider, sidebar_nav
from utils.performance import fetch_metrics
from utils.theme import inject_theme

ensure_db()
inject_theme()

st.set_page_config(
    page_title="瞳观城康",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

sidebar_nav()

# ----------------------- 顶部 Hero ----------------------- #
st.markdown(
    """
    <div class="tg-hero">
      <div style="font-size:34px; font-weight:800;">👁 瞳观城康</div>
      <div style="font-size:18px; opacity:0.92; margin-top:6px;">
        基于视网膜基础模型的眼底—全身慢病早筛与城市健康管理平台
      </div>
      <div style="font-size:14px; opacity:0.85; margin-top:14px;">
        一张眼底照，洞察眼与全身；一个平台，连接社区与城市。
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ----------------------- 平台定位 ----------------------- #
section_divider("🎯 平台定位", "🎯", color="var(--primary)")
st.markdown(
    """
    视网膜是全身微血管和神经的"窗口"。瞳观城康以 **RETFound 视网膜基础模型** 为核心，
    用一张眼底影像完成：
    - 🩺 **眼病分级**：糖尿病视网膜病变（DR）0-4 级
    - 🟣 **多病种联合诊断 v2.0**：DR + 青光眼 + AMD 一次出
    - 🫀 **全身慢病风险**：心血管 / 高血压 / 慢性肾病三分类
    - 🔥 **可解释 AI**：DR / 青光眼 / AMD 三张独立 Grad-CAM 热力图
    - 🏙 **城市驾驶舱**：从单个筛查到整城管理，区域眼健康一屏掌握
    - 🌐 **数字孪生 3D 城市地图**：北京 10 社区 3D 柱状全景，柱高=风险强度
    """
)

section_divider()

# ----------------------- 系统状态 + 数据初始化 ----------------------- #
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("##### 🔌 系统状态")
    h = health_check()
    if h.get("ok"):
        st.success(
            f"✅ 模型服务在线 · 端点：`{get_api_url()}` · "
            f"服务：{h.get('service', '—')} · 版本：{h.get('ts', '—')}"
        )
    else:
        st.error(f"❌ 模型服务不可达：`{get_api_url()}`\n\n原因：{h.get('error')}")
        st.info(
            "💡 A 同学的 FastAPI 还没启动？"
            "可先运行模拟后端 `uvicorn mock_backend:app --port 8000`。"
        )

with col2:
    st.markdown("##### 🧪 演示数据")
    if st.button("📥 插入演示数据（首次启动建议）", use_container_width=True):
        n = seed_demo_data(force=False)
        st.toast(f"✅ 已就绪，新增 {n} 条记录", icon="✅")
        st.success(f"完成，共插入 {n} 条演示记录。")

# 性能速览（首页瞥一眼）
perf = fetch_metrics(timeout=1.5)
if not perf.get("_offline"):
    section_divider("🚀 性能速览", "🚀", color="#0ea5e9")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("TPS", f"{perf.get('tps', 0):.2f}")
    p2.metric("p50", f"{perf.get('p50_ms', 0)} ms")
    p3.metric("p95", f"{perf.get('p95_ms', 0)} ms")
    p4.metric("成功率", f"{perf.get('success_rate', 0) * 100:.1f}%")
    p5.metric("累计调用", f"{perf.get('total_calls', 0)}",
              delta="查看详情 →", delta_color="off")
    st.page_link("pages/5_Performance.py", label="📊 进入性能监控 Dashboard",
                 icon="🚀")

section_divider()

# ----------------------- 数据导出 ----------------------- #
section_divider("📤 数据导出", "📤", color="#16a34a")
exp_left, exp_right = st.columns([1, 2])
with exp_left:
    try:
        from utils.report import build_patient_list_csv
        csv_bytes = build_patient_list_csv()
        st.download_button(
            label="📋 导出全部患者记录（CSV）",
            data=csv_bytes,
            file_name=f"patient_list_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="仅导出 SQLite 真实记录，不含模拟演示数据。UTF-8 BOM，Excel 直接打开。",
        )
    except Exception as e:
        st.error(f"❌ 导出失败：{e}")

with exp_right:
    try:
        real_n = len(fetch_all_screenings())
    except Exception:
        real_n = 0
    st.caption(
        f"当前数据库共有 **{real_n}** 条筛查记录。"
        "点击左侧按钮导出全部记录（仅 SQLite 真实数据，不含模拟）。"
    )

section_divider()

# ----------------------- 创新点四宫格 ----------------------- #
section_divider("💡 四大创新点", "💡", color="#dc2626")
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

with c1:
    info_card(
        title="视网膜基础模型",
        body="基于 RETFound 微调，不从头训练。用 160 万张眼底图预训练的基础模型，"
             "让基层筛查也有三甲级 AI 能力。",
        icon="🔬",
        color="#1f4287",
    )
with c2:
    info_card(
        title="眼底—全身多任务",
        body="DR + 青光眼 + AMD + 心血管 / 高血压 / 慢性肾病。一张眼底照看眼又看全身，"
             "v2.0 新增多病种联合诊断。",
        icon="🫀",
        color="#dc2626",
    )
with c3:
    info_card(
        title="可解释 AI",
        body="DR / 青光眼 / AMD 三张独立 Grad-CAM 热力图。AI 不是黑箱，"
             "医生看得见它在看哪里。",
        icon="🔥",
        color="#d97706",
    )
with c4:
    info_card(
        title="城市健康驾驶舱",
        body="区域筛查地图、疾病谱、转诊闭环、数字孪生 3D 城市地图。"
             "从单个患者到整个城市，眼健康管理一屏掌握。",
        icon="🏙",
        color="#16a34a",
    )

section_divider()

# ----------------------- 路由入口 ----------------------- #
section_divider("🧭 快速进入", "🧭", color="#7c3aed")
n1, n2, n3, n4, n5 = st.columns(5)
n1.page_link("pages/1_ScreenWorkbench.py", label="📤 基层筛查", icon="👁")
n2.page_link("pages/2_RegionCockpit.py", label="🏙 区域驾驶舱", icon="📊")
n3.page_link("pages/3_PatientFollowup.py", label="📋 患者随访", icon="📋")
n4.page_link("pages/4_DigitalTwin.py", label="🌐 数字孪生", icon="🌐")
n5.page_link("pages/5_Performance.py", label="🚀 性能监控", icon="🚀")

section_divider()

# ----------------------- 团队 / 声明 ----------------------- #
with st.expander("👥 项目信息与免责声明"):
    st.markdown(
        """
        **项目名称**：瞳观城康 —— 基于视网膜基础模型的眼底—全身慢病早筛与城市健康管理平台
        **赛道**：研究生智慧城市比赛 · 智慧医疗赛道
        **分工**：
        - A 同学：算法线（RETFound 推理/微调、Grad-CAM、RetiZero 零样本、AutoMorph 特征提取、FastAPI 模型服务）
        - B 同学：工程线（FastAPI 模拟后端、Streamlit 三页面 Web 工作台、SQLite 随访数据库、城市驾驶舱）

        ⚠ **本系统仅供科研演示，不作为临床诊断依据。** 任何筛查结果须由执业眼科医师确认。
        """
    )

page_footer()
