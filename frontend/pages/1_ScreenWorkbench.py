# -*- coding: utf-8 -*-
"""
👁 基层筛查工作台
=================
- Tab 1: 单张筛查（医生逐张精看 + Grad-CAM + PDF 报告）
- Tab 2: 批量筛查（基层一天几百张：上传 → 图像质控 → 批量诊断 → 一键入库/导出）

依赖：
    - utils.api_client : 与 mock_backend / 真 FastAPI 通信
    - utils.db         : SQLite 随访档案
    - utils.report     : PDF 报告
    - utils.image_qc   : 新增的离线图像质控
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# 项目内部
from utils.api_client import call_predict, get_gradcam_url, health_check
from utils.db import add_screening, ensure_db
from utils.image_qc import run_image_qc
from utils.page_header import page_footer, page_header, section_divider, sidebar_nav
from utils.report import build_screening_report
from utils.theme import inject_theme

ensure_db()
inject_theme()

st.set_page_config(
    page_title="基层筛查工作台",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

sidebar_nav()
page_header(
    "👁",
    "基层筛查工作台",
    "一张眼底照，洞察眼与全身 —— 瞳观城康 · 智慧医疗赛道",
    breadcrumbs=("首页", "基层筛查工作台"),
    accent="#1f4287",
)
section_divider()

# 侧边栏：患者信息（仅单张筛查用，批量筛查用表格批量填）
with st.sidebar:
    st.header("🧑 患者信息")
    col_a, col_b = st.columns(2)
    with col_a:
        patient_name = st.text_input("姓名", "张三", key="sb_name")
    with col_b:
        patient_id = st.text_input("患者ID", "P001", key="sb_pid")
    age = st.number_input("年龄", min_value=1, max_value=120, value=55, key="sb_age")
    gender = st.selectbox("性别", ["男", "女"], key="sb_gender")
    diabetes = st.checkbox("糖尿病史", value=False, key="sb_diab")
    st.caption("📎 仅供科研演示，不作为临床诊断依据")

# 健康检查（提示用户模型是否在线）
with st.expander("🔌 模型服务状态", expanded=False):
    if st.button("检查模型服务"):
        h = health_check()
        if h.get("ok"):
            st.success(f"✅ 服务可用 · 版本：{h.get('service', 'unknown')}")
        else:
            st.error(f"❌ 服务不可达：{h.get('error')}")


# ============================================================ #
# Tab 拆分：单张 vs 批量
# ============================================================ #
tab_single, tab_batch = st.tabs(["👁 单张筛查", "📦 批量筛查"])


# ============================================================ #
# Tab 1: 单张筛查（保留原逻辑）
# ============================================================ #
with tab_single:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("📤 上传眼底图像")
        uploaded = st.file_uploader(
            "拖拽或点击上传眼底图像",
            type=["jpg", "jpeg", "png"],
            key="single_upload",
            help="支持 JPG / JPEG / PNG 格式",
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="原始眼底图像", use_container_width=True)
            st.session_state["uploaded_bytes"] = uploaded.getvalue()
            st.session_state["uploaded_name"] = uploaded.name

    with right:
        st.subheader("📊 AI 分析结果")

        if "uploaded_bytes" not in st.session_state:
            st.info("请先在左侧上传眼底图像。")
        else:
            if st.button("🚀 开始分析", type="primary", use_container_width=True):
                with st.spinner("AI 分析中…"):
                    try:
                        res = call_predict(
                            st.session_state["uploaded_bytes"],
                            st.session_state["uploaded_name"],
                        )
                        st.session_state["last_result"] = res
                    except requests.exceptions.ConnectionError:
                        st.error(
                            "❌ 无法连接模型服务，请确认 mock_backend 或 A同学的 FastAPI 已启动。"
                        )
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ 分析失败：{e}")
                        st.stop()

            if "last_result" in st.session_state:
                res = st.session_state["last_result"]
                st.success(f"**{res.get('diagnosis', 'N/A')}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("DR 分级", res.get("dr_grade", "—"))
                c2.metric("置信度", f"{res.get('confidence', 0):.1%}")
                c3.metric(
                    "模型",
                    res.get("model_version", "—")[:14]
                    + ("…" if len(res.get("model_version", "")) > 14 else ""),
                )

                st.markdown("##### 🫀 全身风险提示")
                r1, r2, r3 = st.columns(3)
                r1.metric("心血管", res.get("cv_risk", "—"))
                r2.metric("高血压", res.get("htn_risk", "—"))
                r3.metric("慢性肾病", res.get("ckd_risk", "—"))

                # =============== 多病种联合诊断 v2.0 =============== #
                st.markdown("##### 👁 多病种联合诊断（DR / 青光眼 / AMD）")
                md1, md2, md3 = st.columns(3)

                # 1) DR 主病（已经在上面 c1.metric 显示）
                with md1:
                    st.markdown("**🩺 DR（糖尿病视网膜病变）**")
                    dr_color = (
                        "🟢" if res.get("dr_grade", 0) == 0
                        else "🟡" if res.get("dr_grade", 0) <= 2
                        else "🔴"
                    )
                    st.markdown(
                        f"{dr_color} **DR {res.get('dr_grade', 0)} 级** · "
                        f"置信 {res.get('confidence', 0):.0%}"
                    )

                # 2) 青光眼
                with md2:
                    st.markdown("**🟣 青光眼风险**")
                    glaucoma = res.get("glaucoma_risk", "低风险")
                    glaucoma_prob = float(res.get("glaucoma_prob", 0))
                    g_color = "🟢" if "低" in glaucoma else ("🟡" if "中" in glaucoma else "🔴")
                    st.markdown(f"{g_color} **{glaucoma}** · 概率 {glaucoma_prob:.0%}")

                # 3) AMD
                with md3:
                    st.markdown("**🟠 AMD（黄斑变性）**")
                    amd = res.get("amd_risk", "无")
                    amd_prob = float(res.get("amd_prob", 0))
                    a_color = "🟢" if amd == "无" else ("🟡" if amd == "早期" else "🔴")
                    st.markdown(f"{a_color} **{amd}** · 概率 {amd_prob:.0%}")

                # 联合诊断结论横幅
                dr_grade = int(res.get("dr_grade", 0))
                glaucoma_risk = res.get("glaucoma_risk", "低风险")
                amd_risk = res.get("amd_risk", "无")
                if dr_grade >= 3 or "高" in glaucoma_risk or amd_risk in ("中期", "晚期"):
                    st.error(
                        "🚨 **多病种联合提示**：存在 ≥1 项需紧急转诊的指征，"
                        "建议尽快到眼科专科进一步检查（OCT / 视野 / FFA）。"
                    )
                elif dr_grade >= 2 or "中" in glaucoma_risk or amd_risk == "早期":
                    st.warning(
                        "⚠ **多病种联合提示**：存在 ≥1 项需关注的中度指征，"
                        "建议 3-6 个月内复查。"
                    )
                else:
                    st.success(
                        "✅ **多病种联合提示**：三项均处于低风险，"
                        "建议年度常规复查。"
                    )

                with st.expander("📈 各类别概率明细", expanded=False):
                    probs = res.get("probs_dict") or res.get("all_probs") or {}
                    if probs:
                        df_p = pd.DataFrame(
                            [{"类别": k, "概率": float(v)} for k, v in probs.items()]
                        ).sort_values("概率", ascending=False)
                        st.dataframe(df_p, use_container_width=True, hide_index=True)

                gradcam_url = get_gradcam_url(res)
                if gradcam_url:
                    st.markdown("##### 🔥 AI 关注区域（Grad-CAM 三联）")
                    g1, g2, g3 = st.columns(3)
                    targets = [
                        (g1, gradcam_url, "DR 关注区域", "dr"),
                        (g2, get_gradcam_url({**res, "gradcam_url": res.get("gradcam_url_glaucoma", "")}),
                         "青光眼（视盘）", "glaucoma"),
                        (g3, get_gradcam_url({**res, "gradcam_url": res.get("gradcam_url_amd", "")}),
                         "AMD（黄斑）", "amd"),
                    ]
                    # 缓存所有热力图到 session，供 PDF 报告使用
                    for col, url, caption, key in targets:
                        with col:
                            if not url:
                                st.info(f"{caption}：暂不可用")
                                continue
                            try:
                                img_resp = requests.get(url, timeout=10)
                                if img_resp.status_code == 200:
                                    img_bytes = img_resp.content
                                    st.image(
                                        Image.open(io.BytesIO(img_bytes)),
                                        caption=caption,
                                        use_container_width=True,
                                    )
                                    st.session_state[f"last_gradcam_{key}"] = img_bytes
                                else:
                                    st.info(f"{caption}：加载中…")
                            except Exception:
                                st.info(f"{caption}：加载中…")

                st.divider()
                st.markdown("##### 📄 一键导出报告")
                patient_info = {
                    "patient_name": patient_name,
                    "patient_id": patient_id,
                    "age": age,
                    "gender": gender,
                    "diabetes": diabetes,
                }
                gradcam_bytes = st.session_state.get("last_gradcam_dr")
                gradcam_glaucoma_bytes = st.session_state.get("last_gradcam_glaucoma")
                gradcam_amd_bytes = st.session_state.get("last_gradcam_amd")
                try:
                    pdf_bytes = build_screening_report(
                        result=res,
                        patient_info=patient_info,
                        gradcam_bytes=gradcam_bytes,
                        gradcam_glaucoma_bytes=gradcam_glaucoma_bytes,
                        gradcam_amd_bytes=gradcam_amd_bytes,
                        model_version=res.get("model_version", "mock-1.0"),
                    )
                    file_name = (
                        f"筛查报告-{patient_name}-{datetime.now().strftime('%Y%m%d')}.pdf"
                    )
                    st.download_button(
                        label="📄 导出本次诊断报告（PDF）",
                        data=pdf_bytes,
                        file_name=file_name,
                        mime="application/pdf",
                        use_container_width=False,
                    )
                except Exception as e:
                    st.error(f"❌ PDF 生成失败：{e}")

    # 底部：一键入库（仅对单张筛查生效）
    st.divider()
    st.subheader("💾 保存到随访档案")
    if "last_result" in st.session_state and st.button(
        "📝 写入本次筛查记录", type="secondary", use_container_width=False, key="save_single"
    ):
        res = st.session_state["last_result"]
        record_id = add_screening(
            patient_id=patient_id,
            patient_name=patient_name,
            age=age,
            gender=gender,
            diabetes=diabetes,
            dr_grade=int(res.get("dr_grade", 0)),
            cv_risk=res.get("cv_risk", "—"),
            htn_risk=res.get("htn_risk", "—"),
            ckd_risk=res.get("ckd_risk", "—"),
            diagnosis=res.get("diagnosis", "—"),
            confidence=float(res.get("confidence", 0)),
            gradcam_path=res.get("gradcam_url", ""),
            glaucoma_risk=res.get("glaucoma_risk", "低风险"),
            glaucoma_prob=float(res.get("glaucoma_prob", 0)),
            amd_risk=res.get("amd_risk", "无"),
            amd_prob=float(res.get("amd_prob", 0)),
        )
        st.toast(f"✅ 已写入随访档案（记录 #{record_id}）", icon="✅")
        st.success(
            f"患者 **{patient_name}**（{patient_id}）的本次筛查已保存。"
            "可在「患者随访」页面查看趋势。"
        )


# ============================================================ #
# Tab 2: 批量筛查（新增）
# ============================================================ #
with tab_batch:
    st.subheader("📦 批量筛查")
    st.caption(
        "适用场景：基层一天集中筛查（社区义诊/单位体检/学校普查）。"
        "先做图像质控剔除废片，再批量调模型，最后一键入库或导出 CSV。"
    )

    # ---------- 1) 批量上传 ----------
    batch_files = st.file_uploader(
        "📤 选择多张眼底图像（JPG / JPEG / PNG，可按住 Ctrl/Shift 多选）",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if not batch_files:
        st.info("👆 请先上传 ≥1 张图像")
    else:
        st.success(f"已选择 {len(batch_files)} 张图像")

        # ---------- 2) 质控阈值可调 ----------
        with st.expander("⚙️ 质控阈值（高级，可保持默认）", expanded=False):
            qc1, qc2, qc3, qc4 = st.columns(4)
            thr_min_side = qc1.slider("最小边长 (px)", 128, 512, 256, 32)
            thr_dark = qc2.slider("过暗阈值（灰度均值）", 10, 80, 35, 5)
            thr_bright = qc3.slider("过曝阈值（灰度均值）", 180, 250, 220, 5)
            thr_blur = qc4.slider("模糊阈值（边缘能量）", 10, 200, 60, 10)

        # ---------- 3) 患者信息批量填写 ----------
        with st.expander("👥 批量患者信息（默认：批量用同一信息；如每张不同，请用文件名约定）", expanded=True):
            mp1, mp2, mp3, mp4 = st.columns(4)
            batch_default_name = mp1.text_input("默认姓名", "批量筛查", key="batch_name")
            batch_default_id_prefix = mp2.text_input(
                "默认患者ID前缀", "BATCH", key="batch_pid",
                help="每张图的 patient_id 会自动追加序号，如 BATCH-001、BATCH-002",
            )
            batch_default_age = mp3.number_input(
                "默认年龄", min_value=1, max_value=120, value=55, key="batch_age"
            )
            batch_default_gender = mp4.selectbox(
                "默认性别", ["男", "女"], key="batch_gender"
            )
            batch_default_diab = st.checkbox(
                "默认有糖尿病史", value=False, key="batch_diab"
            )
            naming_rule = st.radio(
                "命名规则",
                ["按上传顺序自动生成", "从文件名提取 (去后缀)"],
                horizontal=True,
                help="如选了「从文件名提取」，patient_id 和 patient_name 会按文件名生成",
            )

        # ---------- 4) 启动按钮 ----------
        run_batch = st.button(
            f"🚀 启动批量质控 + 诊断（共 {len(batch_files)} 张）",
            type="primary",
            use_container_width=True,
        )

        # ---------- 5) 阶段 A: 图像质控 ----------
        if run_batch:
            qc_results: List[Dict[str, Any]] = []
            qc_bar = st.progress(0.0, text="🔍 图像质控中…")
            qc_status = st.status("图像质控", expanded=True)
            for i, f in enumerate(batch_files, start=1):
                try:
                    img = Image.open(f).convert("RGB")
                except Exception as e:
                    qc_results.append({
                        "filename": f.name,
                        "qc_ok": False,
                        "qc_score": 0,
                        "qc_reason": f"无法读取: {e}",
                        "laterality": "UNKNOWN",
                    })
                else:
                    r = run_image_qc(
                        img,
                        min_side=thr_min_side,
                        dark_mean=thr_dark,
                        bright_mean=thr_bright,
                        blur_var=thr_blur,
                    )
                    qc_results.append({
                        "filename": f.name,
                        "qc_ok": r.ok,
                        "qc_score": r.score,
                        "qc_reason": r.reason,
                        "laterality": r.laterality,
                        "width": r.width,
                        "height": r.height,
                        "mean_gray": round(r.mean_gray, 1),
                        "blur_var": round(r.blur_var, 1),
                        "img_bytes": f.getvalue(),
                    })
                qc_bar.progress(i / len(batch_files), text=f"🔍 质控 {i}/{len(batch_files)}")

            passed = sum(1 for x in qc_results if x.get("qc_ok"))
            qc_status.update(
                label=f"✅ 质控完成：通过 {passed}/{len(qc_results)}",
                state="complete",
            )
            st.session_state["batch_qc_results"] = qc_results

        # ---------- 6) 阶段 B: 批量诊断（仅对质控通过者）----------
        if "batch_qc_results" in st.session_state:
            qc_results = st.session_state["batch_qc_results"]
            st.divider()
            st.markdown("##### 📋 质控结果")
            qc_df = pd.DataFrame(
                [
                    {k: v for k, v in r.items() if k != "img_bytes"}
                    for r in qc_results
                ]
            )
            st.dataframe(qc_df, use_container_width=True, hide_index=True)

            col_act1, col_act2 = st.columns([1, 3])
            with col_act1:
                run_diag = st.button(
                    "🧠 批量诊断（仅质控通过）",
                    type="primary",
                    use_container_width=True,
                )
            with col_act2:
                st.caption(
                    f"将通过 {sum(1 for x in qc_results if x.get('qc_ok'))} 张；"
                    f"跳过 {sum(1 for x in qc_results if not x.get('qc_ok'))} 张不合格图像"
                )

            if run_diag:
                diag_results: List[Dict[str, Any]] = []
                targets = [r for r in qc_results if r.get("qc_ok")]
                if not targets:
                    st.error("❌ 没有通过质控的图像，无法诊断")
                else:
                    bar = st.progress(0.0, text="🧠 批量诊断中…")
                    status = st.status("AI 诊断", expanded=True)
                    for i, r in enumerate(targets, start=1):
                        try:
                            res = call_predict(r["img_bytes"], r["filename"])
                            r["diagnosis"] = res.get("diagnosis", "—")
                            r["dr_grade"] = res.get("dr_grade", 0)
                            r["confidence"] = float(res.get("confidence", 0))
                            r["cv_risk"] = res.get("cv_risk", "—")
                            r["htn_risk"] = res.get("htn_risk", "—")
                            r["ckd_risk"] = res.get("ckd_risk", "—")
                            r["gradcam_url"] = res.get("gradcam_url", "")
                            r["model_version"] = res.get("model_version", "—")
                            r["probs_dict"] = res.get("probs_dict") or res.get("all_probs") or {}
                            # 多病种联合诊断（v2.0）
                            r["glaucoma_risk"] = res.get("glaucoma_risk", "低风险")
                            r["glaucoma_prob"] = float(res.get("glaucoma_prob", 0))
                            r["amd_risk"] = res.get("amd_risk", "无")
                            r["amd_prob"] = float(res.get("amd_prob", 0))
                            r["gradcam_url_glaucoma"] = res.get("gradcam_url_glaucoma", "")
                            r["gradcam_url_amd"] = res.get("gradcam_url_amd", "")
                            r["status"] = "ok"
                            r["error"] = ""
                        except Exception as e:
                            r["status"] = "fail"
                            r["error"] = str(e)
                        bar.progress(i / len(targets), text=f"🧠 诊断 {i}/{len(targets)}")
                    status.update(
                        label=f"✅ 批量诊断完成：{len(targets)} 张",
                        state="complete",
                    )
                    st.session_state["batch_results"] = qc_results
                    st.rerun()

        # ---------- 7) 阶段 C: 结果汇总表 + 详情查看 ----------
        if "batch_results" in st.session_state:
            results = st.session_state["batch_results"]
            diag_done = [r for r in results if r.get("status") == "ok"]

            st.divider()
            st.markdown("##### 📊 诊断结果汇总")
            summary_cols = [
                "filename", "laterality", "qc_score", "qc_ok",
                "diagnosis", "dr_grade", "confidence",
                "cv_risk", "htn_risk", "ckd_risk",
                "glaucoma_risk", "glaucoma_prob", "amd_risk", "amd_prob",
                "status",
            ]
            summary_df = pd.DataFrame(
                [
                    {k: r.get(k, "") for k in summary_cols}
                    for r in results
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # ---------- 8) 详情查看（点击某一行）----------
            st.markdown("##### 🔍 单张详情")
            if diag_done:
                sel = st.selectbox(
                    "选择要查看详情的图像",
                    options=[r["filename"] for r in diag_done],
                    key="batch_select",
                )
                hit = next(r for r in diag_done if r["filename"] == sel)
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**诊断卡片**")
                    st.success(f"**{hit['diagnosis']}**")
                    m1, m2 = st.columns(2)
                    m1.metric("DR 分级", hit["dr_grade"])
                    m2.metric("置信度", f"{hit['confidence']:.1%}")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("心血管", hit["cv_risk"])
                    r2.metric("高血压", hit["htn_risk"])
                    r3.metric("慢性肾病", hit["ckd_risk"])
                    st.markdown("**多病种联合**")
                    g1, g2 = st.columns(2)
                    g1.metric("青光眼", hit.get("glaucoma_risk", "低风险"))
                    g2.metric("AMD", hit.get("amd_risk", "无"))
                with d2:
                    st.markdown("**Grad-CAM 热力图（DR）**")
                    if hit.get("gradcam_url"):
                        full = get_gradcam_url(hit)
                        try:
                            img_resp = requests.get(full, timeout=10)
                            if img_resp.status_code == 200:
                                st.image(
                                    Image.open(io.BytesIO(img_resp.content)),
                                    caption=f"{sel} · DR Grad-CAM",
                                    use_container_width=True,
                                )
                            else:
                                st.info("热力图加载失败")
                        except Exception as e:
                            st.info(f"热力图加载失败: {e}")
                    else:
                        st.info("该图无热力图 URL")
                    # 青光眼 + AMD 热力图（如有）
                    gg1, gg2 = st.columns(2)
                    for col, key, label in [
                        (gg1, "gradcam_url_glaucoma", "青光眼热力图"),
                        (gg2, "gradcam_url_amd", "AMD 热力图"),
                    ]:
                        with col:
                            if hit.get(key):
                                try:
                                    img_resp = requests.get(
                                        get_gradcam_url({**hit, "gradcam_url": hit[key]}),
                                        timeout=10,
                                    )
                                    if img_resp.status_code == 200:
                                        st.image(
                                            Image.open(io.BytesIO(img_resp.content)),
                                            caption=label,
                                            use_container_width=True,
                                        )
                                    else:
                                        st.info(f"{label}：加载失败")
                                except Exception:
                                    st.info(f"{label}：加载失败")
                            else:
                                st.info(f"{label}：暂不可用")

            # ---------- 9) 批量入库 ----------
            st.divider()
            st.markdown("##### 💾 批量入库")
            save_btn = st.button(
                f"📝 一键写入所有通过诊断的记录（共 {len(diag_done)} 条）",
                type="secondary",
                use_container_width=False,
                key="save_batch",
            )
            if save_btn and diag_done:
                ok_count = 0
                for idx, r in enumerate(diag_done, start=1):
                    if naming_rule == "从文件名提取 (去后缀)":
                        stem = r["filename"].rsplit(".", 1)[0]
                        cur_name = stem
                        cur_id = stem
                    else:
                        cur_name = batch_default_name
                        cur_id = f"{batch_default_id_prefix}-{idx:03d}"
                    add_screening(
                        patient_id=cur_id,
                        patient_name=cur_name,
                        age=int(batch_default_age),
                        gender=batch_default_gender,
                        diabetes=bool(batch_default_diab),
                        dr_grade=int(r.get("dr_grade", 0)),
                        cv_risk=r.get("cv_risk", "—"),
                        htn_risk=r.get("htn_risk", "—"),
                        ckd_risk=r.get("ckd_risk", "—"),
                        diagnosis=r.get("diagnosis", "—"),
                        confidence=float(r.get("confidence", 0)),
                        gradcam_path=r.get("gradcam_url", ""),
                        glaucoma_risk=r.get("glaucoma_risk", "低风险"),
                        glaucoma_prob=float(r.get("glaucoma_prob", 0)),
                        amd_risk=r.get("amd_risk", "无"),
                        amd_prob=float(r.get("amd_prob", 0)),
                    )
                    ok_count += 1
                st.toast(f"✅ 已批量写入 {ok_count} 条记录", icon="✅")
                st.success(f"已批量写入 {ok_count} 条随访记录，可在「患者随访」查看。")

            # ---------- 10) 批量导出 CSV ----------
            st.divider()
            st.markdown("##### 📤 导出")
            if diag_done:
                export_df = pd.DataFrame(
                    [
                        {
                            "文件名": r["filename"],
                            "左右眼": r.get("laterality", "—"),
                            "质控评分": r.get("qc_score", 0),
                            "质控通过": "是" if r.get("qc_ok") else "否",
                            "DR 分级": r.get("dr_grade", 0),
                            "诊断": r.get("diagnosis", ""),
                            "置信度": round(r.get("confidence", 0), 4),
                            "心血管风险": r.get("cv_risk", ""),
                            "高血压风险": r.get("htn_risk", ""),
                            "慢性肾病风险": r.get("ckd_risk", ""),
                            "青光眼风险": r.get("glaucoma_risk", ""),
                            "青光眼概率": round(r.get("glaucoma_prob", 0), 4),
                            "AMD 分期": r.get("amd_risk", ""),
                            "AMD 概率": round(r.get("amd_prob", 0), 4),
                            "模型": r.get("model_version", ""),
                            "质控原因": r.get("qc_reason", ""),
                        }
                        for r in diag_done
                    ]
                )
                csv = export_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 导出本批结果为 CSV",
                    data=csv,
                    file_name=f"批量筛查-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=False,
                )


st.divider()
page_footer("瞳观城康 | 研究生智慧城市比赛 · 智慧医疗赛道 · B 同学工程线")
