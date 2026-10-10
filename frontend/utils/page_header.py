# -*- coding: utf-8 -*-
"""
page_header.py
==============
瞳观城康 · 统一页面头部组件 + 全局布局助手

5 个页面统一使用：
    from utils.page_header import page_header, section_divider, info_card, page_footer

设计：
    - 标题区：图标 + 大标题 + 副标题 + 面包屑（可选）
    - section_divider()：统一间距分隔
    - info_card(title, text, icon)：4 创新点那种"卡片"统一
    - page_footer()：页脚版权
    - sidebar_nav()：侧边栏快速导航（4 子页 + 主题切换器）
"""
from __future__ import annotations

import streamlit as st

from utils.theme import render_theme_switcher


# ===================================================================== #
# 1. 统一页面头部
# ===================================================================== #
def page_header(
    icon: str,
    title: str,
    subtitle: str = "",
    breadcrumbs: tuple = (),
    accent: str = "var(--primary)",
) -> None:
    """
    统一页面头部：图标 + 标题 + 副标题 + 面包屑。

    用法：
        page_header("👁", "基层筛查工作台", "一张眼底照，洞察眼与全身")

    Args:
        icon: 主图标（emoji）
        title: 大标题
        subtitle: 副标题
        breadcrumbs: 面包屑，如 ("首页", "基层筛查工作台")
        accent: 主色，CSS 变量
    """
    bc_html = ""
    if breadcrumbs:
        items = []
        for i, item in enumerate(breadcrumbs):
            sep = "›" if i > 0 else ""
            color = "var(--primary)" if i == len(breadcrumbs) - 1 else "var(--fg-muted)"
            items.append(
                f"<span style='color:{color};font-size:13px'>{sep} {item}</span>"
            )
        bc_html = "<div style='margin-bottom:6px'>" + " ".join(items) + "</div>"

    html = f"""
<div class="tg-page-header" style="border-left:5px solid {accent};">
  {bc_html}
  <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
    <span style="font-size:36px;line-height:1">{icon}</span>
    <h1 style="margin:0;color:{accent};font-size:32px;font-weight:800;line-height:1.2">{title}</h1>
  </div>
  {f'<div style="margin-top:6px;color:var(--fg-soft);font-size:14px">{subtitle}</div>' if subtitle else ''}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def section_divider(
    text: str = "",
    icon: str = "",
    color: str = "var(--primary)",
) -> None:
    """
    分区分隔条。比 st.divider() 更醒目。
    """
    if text:
        html = f"""
<div class="tg-section-divider" style="border-left:4px solid {color}">
  <span style="color:{color};font-weight:700;font-size:15px">
    {icon + ' ' if icon else ''}{text}
  </span>
</div>
"""
    else:
        html = "<hr style='border:0;border-top:1px solid var(--border);margin:14px 0'>"
    st.markdown(html, unsafe_allow_html=True)


# ===================================================================== #
# 2. 统一卡片
# ===================================================================== #
def info_card(
    title: str,
    body: str = "",
    icon: str = "",
    color: str = "var(--primary)",
    footer: str = "",
) -> None:
    """
    单个信息卡片（左侧色条 + 标题 + 正文 + 底部）。
    """
    html = f"""
<div class="tg-card" style="border-left:4px solid {color}">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
    <span style="font-size:20px">{icon}</span>
    <h4 style="margin:0;color:{color};font-size:16px">{title}</h4>
  </div>
  {f'<div style="color:var(--fg-soft);font-size:14px;line-height:1.6">{body}</div>' if body else ''}
  {f'<div style="margin-top:8px;color:var(--fg-muted);font-size:12px">{footer}</div>' if footer else ''}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ===================================================================== #
# 3. 统一页脚
# ===================================================================== #
def page_footer(text: str = "瞳观城康 | 研究生智慧城市比赛 · 智慧医疗赛道") -> None:
    """统一页脚：版权 + 项目名。"""
    st.markdown(
        f"<div style='text-align:center;color:var(--fg-muted);font-size:12px;"
        f"padding:24px 0 8px;border-top:1px solid var(--border);margin-top:32px'>{text}</div>",
        unsafe_allow_html=True,
    )


# ===================================================================== #
# 4. 侧边栏统一（4 页面之间快速跳转 + 主题）
# ===================================================================== #
def sidebar_nav() -> None:
    """
    侧边栏统一组件：Logo + 4 页面跳转 + 主题切换器。
    在每个 page 顶部调用一次。
    """
    with st.sidebar:
        st.markdown(
            "<div style='padding:8px 0 16px'>"
            "<div style='font-size:28px;font-weight:800;color:var(--primary)'>👁 瞳观城康</div>"
            "<div style='color:var(--fg-muted);font-size:11px;margin-top:2px'>"
            "眼底—全身慢病早筛 · 城市健康管理</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("##### 🧭 快速跳转")
        # 用 page_link 保持 streamlit 自带导航一致
        try:
            st.page_link("app.py", label="🏠 首页", icon="🏠")
            st.page_link("pages/1_ScreenWorkbench.py", label="👁 基层筛查", icon="👁")
            st.page_link("pages/2_RegionCockpit.py", label="🏙 区域驾驶舱", icon="🏙")
            st.page_link("pages/3_PatientFollowup.py", label="📋 患者随访", icon="📋")
            st.page_link("pages/4_DigitalTwin.py", label="🌐 数字孪生", icon="🌐")
        except Exception:
            # 兼容老 streamlit
            st.markdown(
                "<div style='color:var(--fg-soft);font-size:13px;line-height:1.9'>"
                "🏠 <a href='/'>首页</a><br/>"
                "👁 <a href='/ScreenWorkbench'>基层筛查</a><br/>"
                "🏙 <a href='/RegionCockpit'>区域驾驶舱</a><br/>"
                "📋 <a href='/PatientFollowup'>患者随访</a><br/>"
                "🌐 <a href='/DigitalTwin'>数字孪生</a>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        render_theme_switcher()
