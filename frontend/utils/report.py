# -*- coding: utf-8 -*-
"""
utils/report.py
=================
集中所有报告生成逻辑：基层筛查 PDF、患者随访 PDF、区域 CSV zip。

设计要点：
- 中文字体运行时探测：Windows 自带 msyh.ttc / simhei.ttf 优先，
  assets/fonts/*.otf 兜底，全部失败则用 Helvetica + 警告。
- matplotlib 用 Agg 后端（无 GUI），rcParams 全局生效一次。
- 所有函数返回 ``bytes``，方便前端 ``st.download_button`` 直接拿。
- 字段缺失统一兜底 ``"—"``，绝不抛 KeyError。
"""
from __future__ import annotations

import io
import os
import warnings
import zipfile
from datetime import datetime
from typing import Any, Mapping as _Mapping

import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端，避免在 streamlit / headless 环境报错
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ----------------------------- 全局常量 ----------------------------- #
_FOOTER_MODEL_VERSION = "mock-1.0"   # 默认值，会被传入值覆盖
_DASH = "—"                           # 缺字段占位

# 中文字体候选路径（按优先级排序）
_CJK_FONT_CANDIDATES = [
    ("CJK", r"C:\Windows\Fonts\msyh.ttc"),       # 微软雅黑 Regular
    ("CJK_BOLD", r"C:\Windows\Fonts\msyhbd.ttc"), # 微软雅黑 Bold
    ("CJK_FANGSONG", r"C:\Windows\Fonts\simfang.ttf"),  # 仿宋（备用）
]

_CJK_FALLBACK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "fonts",
)

_FALLBACK_TTF_NAMES = [
    "NotoSansSC-Regular.otf",
    "NotoSansSC-Regular.ttf",
    "SourceHanSansSC-Regular.otf",
]


# ----------------------------- 字体探测 ----------------------------- #
def register_cjk_font() -> str:
    """
    在 reportlab 中注册一个支持中文的字体，返回字体名。
    按优先级尝试：微软雅黑 → 仿宋 → assets/fonts/ 下的开源字体。
    全部失败则返回 'Helvetica' 并发出警告。
    """
    # 1) 优先：Windows 系统字体
    for name, path in _CJK_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception as e:
                warnings.warn(f"注册字体 {path} 失败：{e}", stacklevel=2)

    # 2) 兜底：assets/fonts/ 下的开源字体
    if os.path.isdir(_CJK_FALLBACK_DIR):
        for fname in _FALLBACK_TTF_NAMES:
            fpath = os.path.join(_CJK_FALLBACK_DIR, fname)
            if os.path.exists(fpath):
                try:
                    pdfmetrics.registerFont(TTFont("CJK", fpath))
                    return "CJK"
                except Exception as e:
                    warnings.warn(f"注册字体 {fpath} 失败：{e}", stacklevel=2)

    # 4) 全部失败：降级 Helvetica（中文会变方块，但功能不报错）
    warnings.warn(
        "未找到任何中文字体，PDF 中文将显示为方块（□）。"
        "可在 assets/fonts/ 放置 NotoSansSC-Regular.otf 解决。",
        stacklevel=2,
    )
    return "Helvetica"


def _setup_matplotlib_fonts() -> None:
    """配置 matplotlib 中文字体（无 GUI 环境下也安全）。"""
    # 探测顺序同 reportlab，但 matplotlib 需要的是字体族名
    candidates = [
        "Microsoft YaHei",
        "微软雅黑",
        "SimHei",
        "黑体",
        "SimSun",
        "宋体",
        "Noto Sans SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "DejaVu Sans",
    ]
    # 拿掉不存在的，只留候选
    try:
        from matplotlib.font_manager import findSystemFonts  # noqa: WPS433

        available = {os.path.basename(f).lower() for f in findSystemFonts()}
    except Exception:
        available = set()

    preferred = []
    if any("msyh" in a for a in available):
        preferred.append("Microsoft YaHei")
    if any("simhei" in a for a in available):
        preferred.append("SimHei")
    if any("simsun" in a for a in available):
        preferred.append("SimSun")
    preferred += [c for c in candidates if c not in preferred]

    matplotlib.rcParams["font.sans-serif"] = preferred
    matplotlib.rcParams["axes.unicode_minus"] = False


# 初始化时执行一次：保证 import 此模块即可用
_CJK_FONT_NAME = register_cjk_font()
_setup_matplotlib_fonts()


# ----------------------------- 样式工厂 ----------------------------- #
def _styles():
    """构造带中文字体的样式集合。"""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCJK",
            parent=base["Title"],
            fontName=_CJK_FONT_NAME,
            fontSize=18,
            leading=22,
            spaceAfter=10,
            alignment=1,  # 居中
            textColor=colors.HexColor("#1f3a8a"),
        ),
        "h2": ParagraphStyle(
            "H2CJK",
            parent=base["Heading2"],
            fontName=_CJK_FONT_NAME,
            fontSize=13,
            leading=18,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#1f3a8a"),
        ),
        "normal": ParagraphStyle(
            "NormalCJK",
            parent=base["BodyText"],
            fontName=_CJK_FONT_NAME,
            fontSize=10,
            leading=15,
        ),
        "footer": ParagraphStyle(
            "FooterCJK",
            parent=base["BodyText"],
            fontName=_CJK_FONT_NAME,
            fontSize=8,
            leading=11,
            textColor=colors.grey,
            alignment=1,
        ),
    }


def _table_style(header_color: str = "#1f3a8a") -> TableStyle:
    """构造通用表格样式（中文字体、表头蓝底白字、斑马纹）。"""
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), _CJK_FONT_NAME),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]
    )


def _safe_get(d: _Mapping, key: str, default: Any = _DASH) -> Any:
    """字典取值统一兜底，避免 KeyError。"""
    if d is None:
        return default
    val = d.get(key, default) if hasattr(d, "get") else default
    return default if val is None else val


def _safe_pct(x: Any) -> str:
    """0-1 浮点 → 百分号字符串；非数值返回 —。"""
    try:
        return f"{float(x):.1%}"
    except (TypeError, ValueError):
        return _DASH


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _file_date() -> str:
    return datetime.now().strftime("%Y%m%d")


# ===================================================================== #
# 0. 报告模板：印章 / Banner / 边框 / 报告编号（3.0 升级）
# ===================================================================== #
def _generate_seal_png(dr_grade: int, confidence: float) -> io.BytesIO:
    """
    用 matplotlib 生成一个红色圆形 AI 诊断印章 PNG。
    中心：DR 分级；外圈：瞳观城康 AI；底部：置信度。
    返回 BytesIO（reportlab Image 可以直接吃 BytesIO）。
    """
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=200)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # 外圈红
    outer = plt.Circle((0, 0), 1.0, fill=False, color="#dc2626", linewidth=4)
    inner = plt.Circle((0, 0), 0.92, fill=False, color="#dc2626", linewidth=1.2)
    ax.add_patch(outer)
    ax.add_patch(inner)

    # 顶部弧形文字（用普通文字近似）
    ax.text(0, 0.62, "瞳观城康 AI 诊断", ha="center", va="center",
            fontsize=14, color="#dc2626", weight="bold", fontfamily=_CJK_MPL_FONT)
    # 中心 DR 分级
    ax.text(0, 0.05, f"DR {int(dr_grade)}", ha="center", va="center",
            fontsize=36, color="#dc2626", weight="bold", fontfamily=_CJK_MPL_FONT)
    # 底部置信度
    ax.text(0, -0.55, f"置信度 {confidence:.0%}", ha="center", va="center",
            fontsize=11, color="#dc2626", fontfamily=_CJK_MPL_FONT)
    # 装饰小星
    ax.plot([-0.78, -0.72], [0, 0], color="#dc2626", linewidth=2)
    ax.plot([0.72, 0.78], [0, 0], color="#dc2626", linewidth=2)

    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05,
                transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


# 拿到 matplotlib 已配置的中文字体名
try:
    _CJK_MPL_FONT = matplotlib.rcParams["font.sans-serif"][0]
except Exception:
    _CJK_MPL_FONT = "DejaVu Sans"


def _report_header(patient_info: _Mapping, result: _Mapping) -> Table:
    """
    报告顶部 banner：左侧 Logo 文字 + 中间标题 + 右侧报告编号 / 印章。
    """
    report_no = (
        f"TG-{_file_date()}-"
        f"{str(_safe_get(patient_info, 'patient_id', 'X')).strip() or 'X'}-"
        f"{int(_safe_get(result, 'dr_grade', 0))}"
    )

    # 三列：左 logo 文字 / 中标题 / 右编号 + 印章
    left_cell = Paragraph(
        "<para align='left'>"
        "<font size=18 color='#1f3a8a'><b>👁 瞳观城康</b></font><br/>"
        "<font size=8 color='#64748b'>TongGuan · City Health</font>"
        "</para>",
        _styles()["normal"],
    )
    middle_cell = Paragraph(
        "<para align='center'>"
        "<font size=20 color='#0b1f44'><b>AI 眼底筛查诊断报告</b></font><br/>"
        f"<font size=9 color='#64748b'>生成时间：{_now_str()}</font>"
        "</para>",
        _styles()["normal"],
    )
    right_cell = Paragraph(
        "<para align='right'>"
        f"<font size=8 color='#64748b'>报告编号</font><br/>"
        f"<font size=11 color='#0b1f44'><b>{report_no}</b></font><br/>"
        f"<font size=7 color='#94a3b8'>模型 {_safe_get(result, 'model_version', 'mock-1.0')}</font>"
        "</para>",
        _styles()["normal"],
    )
    header_tbl = Table(
        [[left_cell, middle_cell, right_cell]],
        colWidths=[5 * cm, 7 * cm, 5 * cm],
    )
    header_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#1f3a8a")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return header_tbl


def _report_footer(canvas, doc):
    """每页底部：版权 + 页码。"""
    canvas.saveState()
    canvas.setFont(_CJK_FONT_NAME, 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        2 * cm, 1.2 * cm,
        "瞳观城康 © 2026 · 研究生智慧城市比赛 · 智慧医疗赛道 · 仅供科研演示",
    )
    canvas.drawRightString(
        A4[0] - 2 * cm, 1.2 * cm,
        f"第 {doc.page} 页",
    )
    # 上边框
    canvas.setStrokeColor(colors.HexColor("#1f3a8a"))
    canvas.setLineWidth(0.6)
    canvas.line(2 * cm, A4[1] - 1.5 * cm, A4[0] - 2 * cm, A4[1] - 1.5 * cm)
    canvas.restoreState()


# ===================================================================== #
# 1. 单次筛查报告 PDF
# ===================================================================== #
def build_screening_report(
    result: _Mapping,
    patient_info: _Mapping,
    gradcam_bytes: bytes | None = None,
    gradcam_glaucoma_bytes: bytes | None = None,
    gradcam_amd_bytes: bytes | None = None,
    model_version: str = _FOOTER_MODEL_VERSION,
) -> bytes:
    """
    生成单次 AI 筛查诊断报告 PDF（3.0 三栏 + 印章 + 边框 + 报告编号 + 多病种联合）。

    布局：
        - 顶部：banner（Logo + 标题 + 报告编号）
        - 第一行：左印章 + 右患者信息
        - 第二行：DR 主诊断卡片（彩色背景）
        - 第三行：全身风险三联色块
        - 第四行：多病种联合诊断 v2.0（DR / 青光眼 / AMD）
        - 第五行：类别概率条形图（matplotlib）
        - 第六行：Grad-CAM 热力图（DR 主图 + 青光眼 + AMD 副图）
        - 每页：边框 + 页脚版权 + 页码
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="瞳观城康 · 筛查诊断报告",
        author="瞳观城康 B 同学工程线",
    )
    styles = _styles()
    story: list = []

    # ============ 1. 顶部 Banner ============ #
    story.append(_report_header(patient_info, result))
    story.append(Spacer(1, 0.5 * cm))

    # ============ 2. 印章 + 患者信息（三栏：左 4cm 印章 / 中右 12cm 信息）============ #
    p_name = _safe_get(patient_info, "patient_name")
    p_id = _safe_get(patient_info, "patient_id")
    p_age = _safe_get(patient_info, "age")
    p_gender = _safe_get(patient_info, "gender")
    p_diabetes = "是" if patient_info.get("diabetes") else "否"

    # 左侧：印章图
    dr_grade = int(_safe_get(result, "dr_grade", 0))
    confidence_raw = result.get("confidence", 0)
    try:
        confidence_float = float(confidence_raw)
    except (TypeError, ValueError):
        confidence_float = 0.0
    try:
        seal_buf = _generate_seal_png(dr_grade, confidence_float)
        seal_img = Image(seal_buf, width=3.6 * cm, height=3.6 * cm)
    except Exception:
        seal_img = Paragraph("🔴<br/>AI<br/>诊断<br/>印章", styles["normal"])

    # 右侧：患者信息表
    info_rows = [
        ["姓 名", p_name, "患者 ID", p_id],
        ["年 龄", p_age, "性 别", p_gender],
        ["糖尿病史", p_diabetes, "报告类型", "单次 AI 筛查"],
    ]
    info_tbl = Table(
        info_rows,
        colWidths=[2.2 * cm, 4.5 * cm, 2.2 * cm, 4.5 * cm],
    )
    info_tbl.setStyle(_table_style(header_color="#0b1f44"))

    section1 = Table(
        [[seal_img, info_tbl]],
        colWidths=[4 * cm, 13.4 * cm],
    )
    section1.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f8fafc")),
                ("BOX", (1, 0), (1, 0), 0.8, colors.HexColor("#0b1f44")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(section1)
    story.append(Spacer(1, 0.4 * cm))

    # ============ 3. 主诊断卡片（大字彩色背景）============ #
    story.append(Paragraph("一、AI 诊断主结果", styles["h2"]))
    diagnosis = _safe_get(result, "diagnosis")

    # 主诊断表：3 行 4 列
    main_rows = [
        ["诊断结论", diagnosis, "DR 分级", f"DR {dr_grade} 级"],
        ["置信度", _safe_pct(confidence_raw), "模型版本",
         _safe_get(result, "model_version", model_version)],
    ]
    main_tbl = Table(
        main_rows,
        colWidths=[2.5 * cm, 5.5 * cm, 2.5 * cm, 5.5 * cm],
    )
    # 关键单元格用红色背景（诊断结论）+ 浅蓝背景（DR 分级）
    main_tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), _CJK_FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("FONTNAME", (1, 0), (1, 0), _CJK_FONT_NAME),
                ("FONTSIZE", (1, 0), (1, 0), 13),
                ("FONTNAME", (3, 0), (3, 0), _CJK_FONT_NAME),
                ("FONTSIZE", (3, 0), (3, 0), 13),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
                ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor("#b91c1c")),
                ("TEXTCOLOR", (3, 0), (3, 0), colors.HexColor("#b91c1c")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f1f5f9")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(main_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ============ 4. 全身风险三联（彩色色块）============ #
    story.append(Paragraph("二、全身风险提示", styles["h2"]))
    cv = str(_safe_get(result, "cv_risk"))
    htn = str(_safe_get(result, "htn_risk"))
    ckd = str(_safe_get(result, "ckd_risk"))

    def _risk_color(r: str) -> colors.Color:
        if "高" in r: return colors.HexColor("#fee2e2")
        if "中" in r: return colors.HexColor("#fef3c7")
        if "低" in r: return colors.HexColor("#dcfce7")
        return colors.HexColor("#e2e8f0")

    def _risk_fg(r: str) -> colors.Color:
        if "高" in r: return colors.HexColor("#b91c1c")
        if "中" in r: return colors.HexColor("#b45309")
        if "低" in r: return colors.HexColor("#15803d")
        return colors.HexColor("#475569")

    risk_cell1 = Paragraph(
        f"<para align='center'><font size=9 color='#64748b'>心血管风险</font><br/>"
        f"<font size=18 color='{_risk_fg(cv).hexval()}'><b>{cv}</b></font></para>",
        styles["normal"],
    )
    risk_cell2 = Paragraph(
        f"<para align='center'><font size=9 color='#64748b'>高血压风险</font><br/>"
        f"<font size=18 color='{_risk_fg(htn).hexval()}'><b>{htn}</b></font></para>",
        styles["normal"],
    )
    risk_cell3 = Paragraph(
        f"<para align='center'><font size=9 color='#64748b'>慢性肾病风险</font><br/>"
        f"<font size=14 color='{_risk_fg(ckd).hexval()}'><b>{ckd}</b></font></para>",
        styles["normal"],
    )
    risk_tbl = Table(
        [[risk_cell1, risk_cell2, risk_cell3]],
        colWidths=[5.8 * cm, 5.8 * cm, 5.8 * cm],
    )
    risk_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), _risk_color(cv)),
                ("BACKGROUND", (1, 0), (1, 0), _risk_color(htn)),
                ("BACKGROUND", (2, 0), (2, 0), _risk_color(ckd)),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#0b1f44")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    story.append(risk_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ============ 4.5 多病种联合诊断 v2.0 ============ #
    story.append(Paragraph("三、多病种联合诊断（DR / 青光眼 / AMD）", styles["h2"]))

    glaucoma_risk = str(_safe_get(result, "glaucoma_risk", "低风险"))
    glaucoma_prob = float(_safe_get(result, "glaucoma_prob", 0))
    amd_risk = str(_safe_get(result, "amd_risk", "无"))
    amd_prob = float(_safe_get(result, "amd_prob", 0))

    def _multi_disease_color(label: str) -> colors.Color:
        if "高" in label or label in ("中期", "晚期"):
            return colors.HexColor("#fee2e2")
        if "中" in label or label == "早期":
            return colors.HexColor("#fef3c7")
        return colors.HexColor("#dcfce7")

    def _multi_disease_fg(label: str) -> str:
        if "高" in label or label in ("中期", "晚期"):
            return "#b91c1c"
        if "中" in label or label == "早期":
            return "#b45309"
        return "#15803d"

    md1 = Paragraph(
        f"<para align='center'><font size=9 color='#64748b'>DR 分级</font><br/>"
        f"<font size=18 color='{_multi_disease_fg(f'DR {dr_grade} 级')}'><b>DR {dr_grade} 级</b></font></para>",
        styles["normal"],
    )
    md2 = Paragraph(
        f"<para align='center'><font size=9 color='#64748b'>青光眼风险</font><br/>"
        f"<font size=16 color='{_multi_disease_fg(glaucoma_risk)}'><b>{glaucoma_risk}</b></font><br/>"
        f"<font size=8 color='#64748b'>概率 {glaucoma_prob:.0%}</font></para>",
        styles["normal"],
    )
    md3 = Paragraph(
        f"<para align='center'><font size=9 color='#64748b'>AMD 分期</font><br/>"
        f"<b><font size=16 color='{_multi_disease_fg(amd_risk)}'>{amd_risk}</font></b><br/>"
        f"<font size=8 color='#64748b'>概率 {amd_prob:.0%}</font></para>",
        styles["normal"],
    )
    multi_tbl = Table(
        [[md1, md2, md3]],
        colWidths=[5.8 * cm, 5.8 * cm, 5.8 * cm],
    )
    multi_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), _multi_disease_color(f"DR {dr_grade} 级")),
                ("BACKGROUND", (1, 0), (1, 0), _multi_disease_color(glaucoma_risk)),
                ("BACKGROUND", (2, 0), (2, 0), _multi_disease_color(amd_risk)),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#0b1f44")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(multi_tbl)

    # 联合诊断结论
    if dr_grade >= 3 or "高" in glaucoma_risk or amd_risk in ("中期", "晚期"):
        conclusion = (
            "<b><font color='#b91c1c'>联合提示：≥1 项需紧急转诊指征，"
            "建议尽快到眼科专科进一步检查（OCT / 视野 / FFA）。</font></b>"
        )
    elif dr_grade >= 2 or "中" in glaucoma_risk or amd_risk == "早期":
        conclusion = (
            "<b><font color='#b45309'>联合提示：≥1 项需关注的中度指征，"
            "建议 3-6 个月内复查。</font></b>"
        )
    else:
        conclusion = (
            "<b><font color='#15803d'>联合提示：三项均处于低风险，"
            "建议年度常规复查。</font></b>"
        )
    story.append(Paragraph(conclusion, styles["normal"]))
    story.append(Spacer(1, 0.4 * cm))

    # ============ 5. 类别概率（水平条形图）============ #
    probs = result.get("probs_dict") or result.get("all_probs") or {}
    if probs:
        story.append(Paragraph("四、5 类 DR 概率分布", styles["h2"]))
        try:
            sorted_probs = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
        except (TypeError, ValueError):
            sorted_probs = list(probs.items())
        labels = [str(k) for k, _ in sorted_probs]
        values = [float(v) for _, v in sorted_probs]

        fig, ax = plt.subplots(figsize=(11, 2.6), dpi=200)
        colors_bar = [
            "#dc2626" if i == 0 else
            "#f97316" if i == 1 else
            "#eab308" if i == 2 else
            "#3b82f6" if i == 3 else
            "#10b981"
            for i in range(len(labels))
        ]
        ax.barh(range(len(labels))[::-1], values, color=colors_bar)
        ax.set_yticks(range(len(labels))[::-1])
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("概率", fontfamily=_CJK_MPL_FONT)
        ax.set_title("类别概率分布（自上而下概率递减）", fontfamily=_CJK_MPL_FONT, fontsize=11)
        for i, v in enumerate(values):
            ax.text(v + 0.01, len(labels) - 1 - i, f"{v:.1%}", va="center", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        bar_buf = io.BytesIO()
        plt.savefig(bar_buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        bar_buf.seek(0)
        story.append(Image(bar_buf, width=16 * cm, height=4.2 * cm))
        story.append(Spacer(1, 0.4 * cm))

    # ============ 6. Grad-CAM 热力图 ============ #
    if gradcam_bytes:
        story.append(Paragraph("五、AI 关注区域（DR Grad-CAM 热力图）", styles["h2"]))
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(gradcam_bytes))
            pil_img.load()
            img_buf = io.BytesIO()
            pil_img.save(img_buf, format="PNG")
            img_buf.seek(0)
            story.append(Image(img_buf, width=12 * cm, height=12 * cm))
            story.append(Paragraph(
                "<para align='center'><font size=8 color='#64748b'>"
                "DR 模型关注区域（红=高关注，蓝=低关注）"
                "</font></para>",
                styles["footer"],
            ))
        except Exception as e:
            story.append(Paragraph(f"（热力图加载失败：{e}）", styles["normal"]))

    # ============ 6.5 多病种副热力图（青光眼 + AMD）============ #
    if gradcam_glaucoma_bytes or gradcam_amd_bytes:
        story.append(Paragraph(
            "六、多病种联合 Grad-CAM（青光眼视盘 / AMD 黄斑）",
            styles["h2"],
        ))
        from PIL import Image as PILImage
        sub_cells = []
        sub_captions = []
        for b, label in [
            (gradcam_glaucoma_bytes, "青光眼视盘（视盘偏鼻侧）"),
            (gradcam_amd_bytes, "AMD 黄斑（黄斑中心凹）"),
        ]:
            if not b:
                sub_cells.append(Paragraph(
                    f"<para align='center'><font size=9 color='#64748b'>{label}：暂不可用</font></para>",
                    styles["normal"],
                ))
                sub_captions.append("")
                continue
            try:
                pil_img = PILImage.open(io.BytesIO(b))
                pil_img.load()
                ib = io.BytesIO()
                pil_img.save(ib, format="PNG")
                ib.seek(0)
                sub_cells.append(Image(ib, width=7.5 * cm, height=7.5 * cm))
                sub_captions.append(label)
            except Exception as e:
                sub_cells.append(Paragraph(f"加载失败：{e}", styles["normal"]))
                sub_captions.append("")

        sub_tbl = Table(
            [sub_cells],
            colWidths=[8.2 * cm, 8.2 * cm],
        )
        sub_tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(sub_tbl)

    # ============ 7. 底部免责 ============ #
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "⚠ 本报告由瞳观城康 AI 模型自动生成，"
        "仅供科研演示与辅助参考，不作为临床诊断依据。"
        "任何筛查结果须由执业眼科医师确认。",
        styles["footer"],
    ))

    doc.build(story, onFirstPage=_report_footer, onLaterPages=_report_footer)
    return buf.getvalue()


# ===================================================================== #
# 2. 患者随访完整档案 PDF
# ===================================================================== #
def _line_chart_png(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    yticks: list[float] | None = None,
    yticklabels: list[str] | None = None,
) -> bytes:
    """用 matplotlib 画单线趋势图，返回 PNG bytes。"""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", fontsize=14)
    else:
        ax.plot(
            df[x_col],
            df[y_col],
            marker="o",
            linewidth=2.4,
            markersize=7,
            color="#b91c1c",
        )
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("筛查日期")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if yticks is not None and yticklabels is not None:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
    fig.autofmt_xdate()
    fig.tight_layout()
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    png_buf.seek(0)
    return png_buf.getvalue()


def _multi_risk_chart_png(df: pd.DataFrame) -> bytes:
    """全身风险趋势（三线）PNG bytes。"""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", fontsize=14)
    else:
        risk_to_num = {"低": 1, "中": 2, "高": 3, "不适用（非糖尿病患者）": 0}
        for col, label, color in [
            ("cv_risk", "心血管", "#2563eb"),
            ("htn_risk", "高血压", "#ea580c"),
            ("ckd_risk", "慢性肾病", "#16a34a"),
        ]:
            nums = (
                df[col].map(risk_to_num).fillna(0).astype(float)
                if col in df.columns
                else pd.Series([0] * len(df))
            )
            ax.plot(
                df["screening_date"],
                nums,
                marker="o",
                linewidth=2.2,
                markersize=6,
                label=label,
                color=color,
            )
        ax.set_title("全身风险等级变化趋势", fontsize=13, pad=10)
        ax.set_xlabel("筛查日期")
        ax.set_ylabel("风险等级")
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["不适用", "低", "中", "高"])
        ax.set_ylim(-0.3, 3.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    png_buf.seek(0)
    return png_buf.getvalue()


def build_patient_report(
    patient_df: pd.DataFrame,
    patient_id: str,
    patient_name: str | None = None,
    model_version: str = _FOOTER_MODEL_VERSION,
) -> bytes:
    """
    生成单患者完整随访档案 PDF。

    参数
    ----
    patient_df : pd.DataFrame
        该患者全部筛查记录，至少包含列：
        screening_date, dr_grade, cv_risk, htn_risk, ckd_risk,
        diagnosis, confidence 等。
    patient_id : str
        患者 ID。
    patient_name : str | None
        患者姓名；为 None 时取 patient_df 第 1 行。
    """
    df = patient_df.copy() if patient_df is not None else pd.DataFrame()
    if patient_name is None and len(df) > 0 and "patient_name" in df.columns:
        patient_name = str(df["patient_name"].iloc[0])
    elif patient_name is None:
        patient_name = "—"

    # 标准化日期列
    if "screening_date" in df.columns:
        df["screening_date"] = pd.to_datetime(df["screening_date"], errors="coerce")
        df = df.sort_values("screening_date")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"随访档案 · {patient_name}",
        author="B 同学工程线",
    )
    styles = _styles()
    story = []

    # 标题
    story.append(Paragraph(f"瞳观城康 · 患者随访档案", styles["title"]))
    story.append(Paragraph(
        f"生成时间：{_now_str()} &nbsp;&nbsp; 模型版本：{model_version}",
        styles["footer"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # 基本信息
    story.append(Paragraph("一、患者基本信息", styles["h2"]))
    age = _safe_get(df.iloc[0], "age") if len(df) > 0 else _DASH
    gender = _safe_get(df.iloc[0], "gender") if len(df) > 0 else _DASH
    diabetes = "—" if len(df) == 0 else ("是" if df.iloc[0].get("diabetes") else "否")
    info_rows = [
        ["姓名", patient_name, "患者 ID", patient_id],
        ["年龄", str(age), "性别", gender],
        ["糖尿病史", diabetes, "累计记录数", str(len(df))],
    ]
    info_tbl = Table(info_rows, colWidths=[2.5 * cm, 5 * cm, 2.5 * cm, 5 * cm])
    info_tbl.setStyle(_table_style())
    story.append(info_tbl)
    story.append(Spacer(1, 0.4 * cm))

    if len(df) == 0:
        story.append(Paragraph("该患者暂无筛查记录。", styles["normal"]))
        doc.build(story)
        return buf.getvalue()

    # 最新一次诊断
    latest = df.iloc[-1]
    story.append(Paragraph("二、最新一次诊断", styles["h2"]))
    latest_rows = [
        ["筛查日期", str(latest.get("screening_date", _DASH))[:10],
         "诊断", _safe_get(latest, "diagnosis")],
        ["DR 分级", str(_safe_get(latest, "dr_grade", 0)),
         "置信度", _safe_pct(latest.get("confidence"))],
        ["心血管风险", _safe_get(latest, "cv_risk"),
         "高血压风险", _safe_get(latest, "htn_risk")],
        ["慢性肾病风险", _safe_get(latest, "ckd_risk"),
         "模型版本", _safe_get(latest, "model_version", model_version)],
    ]
    latest_tbl = Table(latest_rows, colWidths=[2.5 * cm, 5 * cm, 2.5 * cm, 5 * cm])
    latest_tbl.setStyle(_table_style(header_color="#b91c1c"))
    story.append(latest_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # DR 趋势图
    story.append(Paragraph("三、DR 分级变化趋势", styles["h2"]))
    dr_png = _line_chart_png(
        df,
        x_col="screening_date",
        y_col="dr_grade",
        title=f"{patient_name} · DR 分级变化",
        ylabel="DR 分级（0-4）",
        ylim=(-0.5, 4.5),
        yticks=[0, 1, 2, 3, 4],
        yticklabels=["0", "1", "2", "3", "4"],
    )
    story.append(Image(io.BytesIO(dr_png), width=15 * cm, height=6 * cm))
    story.append(Spacer(1, 0.4 * cm))

    # 全身风险趋势图
    story.append(Paragraph("四、全身风险等级变化", styles["h2"]))
    risk_png = _multi_risk_chart_png(df)
    story.append(Image(io.BytesIO(risk_png), width=15 * cm, height=6.5 * cm))
    story.append(PageBreak())

    # 详情表
    story.append(Paragraph("五、全部筛查记录", styles["h2"]))
    display_cols = ["screening_date", "diagnosis", "dr_grade",
                    "cv_risk", "htn_risk", "ckd_risk", "confidence"]
    detail_rows = [["筛查日期", "诊断", "DR", "心血管", "高血压", "慢性肾病", "置信度"]]
    for _, r in df.iterrows():
        detail_rows.append([
            str(r.get("screening_date", _DASH))[:10],
            _safe_get(r, "diagnosis"),
            str(_safe_get(r, "dr_grade", 0)),
            _safe_get(r, "cv_risk"),
            _safe_get(r, "htn_risk"),
            _safe_get(r, "ckd_risk"),
            _safe_pct(r.get("confidence")),
        ])
    detail_tbl = Table(
        detail_rows,
        colWidths=[2.4 * cm, 2.8 * cm, 1.2 * cm, 1.8 * cm, 1.8 * cm, 2.4 * cm, 1.8 * cm],
        repeatRows=1,
    )
    detail_tbl.setStyle(_table_style())
    story.append(detail_tbl)
    story.append(Spacer(1, 0.6 * cm))

    # 随访建议
    story.append(Paragraph("六、随访与转诊建议", styles["h2"]))
    try:
        dr_latest = int(latest.get("dr_grade", 0))
    except (TypeError, ValueError):
        dr_latest = 0
    if dr_latest >= 4:
        advice = "DR 4 级（增殖期），建议立即转诊至眼科专科，可能需激光光凝 / 玻切手术评估。下次随访：1 个月内。"
    elif dr_latest == 3:
        advice = "DR 3 级（重度 NPDR），建议尽快转诊眼科，3 个月内复查。下次随访：1 个月内。"
    elif dr_latest == 2:
        advice = "中度 NPDR，建议 6 个月内复查，并严格控制血糖 / 血压。下次随访：6 个月内。"
    elif dr_latest == 1:
        advice = "轻度 NPDR，建议 12 个月内复查。下次随访：12 个月内。"
    else:
        advice = "当前未见明显 DR 病变，建议年度常规复查。下次随访：12 个月内。"
    story.append(Paragraph(advice, styles["normal"]))

    # 页脚
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "⚠ 本报告由瞳观城康 AI 模型自动生成，仅供科研演示，不作为临床诊断依据。",
        styles["footer"],
    ))
    story.append(Paragraph(
        f"© {_now_str()[:4]} 瞳观城康 · 研究生智慧城市比赛 · 智慧医疗赛道 · B 同学工程线",
        styles["footer"],
    ))

    doc.build(story)
    return buf.getvalue()


# ===================================================================== #
# 3. 区域驾驶舱 CSV zip
# ===================================================================== #
def build_region_csv(df: pd.DataFrame) -> bytes:
    """
    把区域驾驶舱当前面板数据导出成 zip 字节流，
    内含 3 个 UTF-8 BOM CSV（Excel 打开不乱码）：
        - kpi_by_community.csv    社区 KPI 汇总
        - dr_grade_distribution.csv   DR 分级分布
        - systemic_risk.csv       全身风险分布

    参数
    ----
    df : pd.DataFrame
        区域驾驶舱当前 dataframe，需含 community / dr_grade /
        cv_risk / htn_risk / ckd_risk 等列。
    """
    src = df.copy() if df is not None else pd.DataFrame()

    # 1) 社区 KPI
    kpi_rows: list[dict] = []
    if "community" in src.columns:
        for comm, sub in src.groupby("community"):
            kpi_rows.append({
                "社区": comm,
                "抓样量": int(len(sub)),
                "异常检出率": f"{((sub['dr_grade'] > 0).mean()):.2%}" if len(sub) else "0.00%",
                "平均 DR 分级": f"{sub['dr_grade'].mean():.2f}" if len(sub) else "0.00",
                "心血管高风险占比": f"{(sub['cv_risk'] == '高').mean():.2%}" if len(sub) else "0.00%",
                "高血压高风险占比": f"{(sub['htn_risk'] == '高').mean():.2%}" if len(sub) else "0.00%",
            })
    kpi_df = pd.DataFrame(kpi_rows)

    # 2) DR 分级分布
    if "dr_grade" in src.columns:
        dr_dist = (
            src["dr_grade"]
            .value_counts()
            .sort_index()
            .rename_axis("DR 分级")
            .reset_index(name="人数")
        )
        dr_dist["占比"] = (dr_dist["人数"] / max(len(src), 1) * 100).round(2).astype(str) + "%"
    else:
        dr_dist = pd.DataFrame(columns=["DR 分级", "人数", "占比"])

    # 3) 全身风险分布
    risk_rows: list[dict] = []
    for col, label in [("cv_risk", "心血管"), ("htn_risk", "高血压"), ("ckd_risk", "慢性肾病")]:
        if col in src.columns:
            vc = src[col].value_counts().reset_index()
            vc.columns = ["等级", "人数"]
            for _, r in vc.iterrows():
                risk_rows.append({
                    "风险类型": label,
                    "等级": str(r["等级"]),
                    "人数": int(r["人数"]),
                    "占比": f"{int(r['人数']) / max(len(src), 1):.2%}",
                })
    risk_df = pd.DataFrame(risk_rows)

    # 打包成 zip（手动 encode utf-8-sig，让 Excel 直接打开）
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("kpi_by_community.csv", kpi_df.to_csv(index=False).encode("utf-8-sig"))
        z.writestr("dr_grade_distribution.csv", dr_dist.to_csv(index=False).encode("utf-8-sig"))
        z.writestr("systemic_risk.csv", risk_df.to_csv(index=False).encode("utf-8-sig"))
    return buf.getvalue()


# ===================================================================== #
# 4. 全部患者筛查记录（SQLite flat CSV）
# ===================================================================== #
def build_patient_list_csv(df: pd.DataFrame | None = None) -> bytes:
    """
    把 SQLite 随访数据库里所有筛查记录导出为单 CSV（UTF-8 BOM，Excel 直接打开）。

    参数
    ----
    df : pd.DataFrame | None
        若传入 DataFrame，直接使用；为 None 则内部调用 fetch_all_screenings() 读取。

    返回
    ----
    bytes : UTF-8 BOM CSV，一行一条筛查记录，按筛查日期倒序。
    """
    if df is None:
        from utils.db import fetch_all_screenings
        df = fetch_all_screenings()

    src = df.copy() if df is not None else pd.DataFrame()

    # 标准化：screening_date 字符串化，diabetes 转中文
    if "screening_date" in src.columns:
        src["screening_date"] = src["screening_date"].astype(str)

    if "diabetes" in src.columns:
        src["糖尿病史"] = src["diabetes"].apply(
            lambda x: "是" if int(x) in (1, True) else "否" if pd.notna(x) else "—"
        )

    # 列映射（中文化列名，始终包含全部定义字段，缺失的用 NaN 填充）
    col_map = {
        "id": "记录 ID",
        "screening_date": "筛查日期",
        "patient_id": "患者 ID",
        "patient_name": "姓名",
        "age": "年龄",
        "gender": "性别",
        "糖尿病史": "糖尿病史",
        "diagnosis": "诊断",
        "dr_grade": "DR 分级",
        "cv_risk": "心血管风险",
        "htn_risk": "高血压风险",
        "ckd_risk": "慢性肾病风险",
        "confidence": "置信度",
        "gradcam_path": "热力图路径",
    }

    # 补齐所有定义列（缺失的列变成 NaN），再重命名
    all_src_cols = list(col_map.keys())
    out = src.reindex(columns=all_src_cols).rename(columns=col_map)

    # 排序：筛查日期 DESC，id DESC（最近在前）
    date_col = "筛查日期"
    if date_col in out.columns:
        out = out.sort_values(date_col, ascending=False)

    # 空表兜底：空 DataFrame 也有表头，to_csv 正常返回只有表头的字符串
    return out.to_csv(index=False).encode("utf-8-sig")