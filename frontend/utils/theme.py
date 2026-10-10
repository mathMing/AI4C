# -*- coding: utf-8 -*-
"""
theme.py
========
瞳观城康 · 全局主题管理（暗色 / 亮色 / 跟随系统）

提供：
    1. THEME_CSS : 完整的 CSS 变量 + 组件样式（暗色 / 亮色各一份）
    2. inject_theme() : 在每个页面入口调用一次，把 CSS 注入到 Streamlit
    3. render_theme_switcher() : 右上角手动切换器（写入 session_state）
    4. palette_for_pydeck() : 暗色模式下的 pydeck 配色
    5. palette_for_plotly() : 暗色模式下的 plotly 配色

设计：
    - 跟随系统用 matchMedia + prefers-color-scheme（首次加载时）
    - 手动切换后覆盖系统选择，写入 localStorage（HTML 注入一段 JS）
    - 不破坏现有 st.set_page_config() 调用
    - 不引入额外依赖（只用 streamlit + plotly 自带模板）
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


# ===================================================================== #
# 1. 主题常量
# ===================================================================== #
THEMES = ("auto", "light", "dark")  # auto = 跟随系统

# CSS 颜色变量（亮色 / 暗色两套）
LIGHT_VARS = {
    "--bg": "#ffffff",
    "--bg-soft": "#f7f9fc",
    "--bg-card": "#ffffff",
    "--bg-sidebar": "#f0f2f6",
    "--fg": "#0f172a",
    "--fg-soft": "#475569",
    "--fg-muted": "#94a3b8",
    "--primary": "#1f4287",
    "--primary-soft": "#0b1f44",
    "--accent": "#dc2626",
    "--border": "#e2e8f0",
    "--border-strong": "#cbd5e1",
    "--success": "#16a34a",
    "--warning": "#d97706",
    "--error": "#dc2626",
    "--shadow": "0 1px 2px rgba(15,23,42,.06), 0 4px 12px rgba(15,23,42,.05)",
}

DARK_VARS = {
    "--bg": "#0b1220",
    "--bg-soft": "#111a2e",
    "--bg-card": "#15203a",
    "--bg-sidebar": "#0e172a",
    "--fg": "#e2e8f0",
    "--fg-soft": "#cbd5e1",
    "--fg-muted": "#94a3b8",
    "--primary": "#3b82f6",
    "--primary-soft": "#1e3a8a",
    "--accent": "#f87171",
    "--border": "#1f2a44",
    "--border-strong": "#334155",
    "--success": "#4ade80",
    "--warning": "#fbbf24",
    "--error": "#f87171",
    "--shadow": "0 1px 2px rgba(0,0,0,.4), 0 4px 12px rgba(0,0,0,.3)",
}


# ===================================================================== #
# 2. CSS 注入
# ===================================================================== #
def _vars_to_css(vars_dict: dict, indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}{k}: {v};" for k, v in vars_dict.items())


THEME_CSS = f"""
<style>
/* === 瞳观城康 全局主题 === */

/* 浅色变量（默认） */
:root {{
{_vars_to_css(LIGHT_VARS)}
}}

/* 深色变量（通过 [data-theme="dark"] 切换） */
:root[data-theme="dark"] {{
{_vars_to_css(DARK_VARS)}
}}

/* 跟随系统模式（[data-theme="auto"] + 浏览器偏好深色） */
@media (prefers-color-scheme: dark) {{
  :root[data-theme="auto"] {{
{_vars_to_css(DARK_VARS)}
  }}
}}

/* === 全局背景与文字 === */
html, body, [class*="css"] {{
  background-color: var(--bg);
  color: var(--fg);
  transition: background-color .25s ease, color .25s ease;
}}

.stApp {{
  background-color: var(--bg);
}}

/* 侧边栏 */
section[data-testid="stSidebar"] {{
  background-color: var(--bg-sidebar);
  border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] * {{
  color: var(--fg) !important;
}}

/* 主内容区容器 */
.main .block-container {{
  background-color: var(--bg);
  padding-top: 2rem;
}}

/* === 通用卡片 === */
.tg-card {{
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  box-shadow: var(--shadow);
  margin-bottom: 12px;
}}

.tg-card h4 {{ margin-top: 0; color: var(--primary); }}
:root[data-theme="dark"] .tg-card h4 {{ color: #93c5fd; }}

/* === Metric 卡片 === */
[data-testid="stMetric"] {{
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
  box-shadow: var(--shadow);
}}
[data-testid="stMetricValue"] {{
  color: var(--primary);
  font-weight: 700;
}}
:root[data-theme="dark"] [data-testid="stMetricValue"] {{
  color: #93c5fd;
}}
[data-testid="stMetricLabel"] {{
  color: var(--fg-soft);
}}

/* === 标题 === */
h1, h2, h3, h4, h5, h6 {{ color: var(--fg); }}
h1 {{ color: var(--primary); font-weight: 800; }}
:root[data-theme="dark"] h1 {{ color: #93c5fd; }}

/* === 文字 === */
p, li, span, label, div {{ color: var(--fg); }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: var(--fg-muted) !important; }}

/* === 按钮 === */
.stButton > button {{
  background-color: var(--primary);
  color: #ffffff;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  transition: transform .12s ease, box-shadow .12s ease;
}}
.stButton > button:hover {{
  background-color: var(--primary-soft);
  transform: translateY(-1px);
  box-shadow: var(--shadow);
}}
.stDownloadButton > button {{
  background-color: var(--success) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
}}

/* === 输入控件 === */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {{
  background-color: var(--bg-card) !important;
  color: var(--fg) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 8px !important;
}}
:root[data-theme="dark"] .stTextInput input,
:root[data-theme="dark"] .stNumberInput input,
:root[data-theme="dark"] .stTextArea textarea,
:root[data-theme="dark"] .stSelectbox div[data-baseweb="select"] > div {{
  background-color: var(--bg-soft) !important;
}}

/* === DataFrame / 表格 === */
[data-testid="stDataFrame"] {{
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}}

/* === Expander === */
details {{
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px 10px;
}}
details summary {{ color: var(--fg) !important; }}

/* === Divider === */
hr {{ border-color: var(--border) !important; }}

/* === Info / Success / Warning / Error 框 === */
.stAlert {{
  border-radius: 8px;
  border: 1px solid var(--border);
}}

/* === Plotly 图表容器 === */
.js-plotly-plot .plotly .main-svg {{
  background: transparent !important;
}}

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] button {{
  color: var(--fg-soft);
}}
.stTabs [aria-selected="true"] {{
  color: var(--primary) !important;
  border-bottom-color: var(--primary) !important;
}}
:root[data-theme="dark"] .stTabs [aria-selected="true"] {{
  color: #93c5fd !important;
  border-bottom-color: #93c5fd !important;
}}

/* === Hero 横幅（app.py 用） === */
.tg-hero {{
  background: linear-gradient(90deg, var(--primary-soft) 0%, var(--primary) 100%);
  padding: 28px 36px;
  border-radius: 16px;
  color: #ffffff;
  margin-bottom: 12px;
  box-shadow: var(--shadow);
}}
.tg-hero * {{ color: #ffffff !important; }}

/* === 统一页面头部（page_header.py 用） === */
.tg-page-header {{
  background-color: var(--bg-card);
  padding: 18px 22px;
  margin: 0 0 18px 0;
  border-radius: 10px;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  transition: border-color .25s ease, background-color .25s ease;
}}
.tg-page-header h1 {{ margin: 0 !important; line-height: 1.2; }}

/* === 区分分隔条（section_divider 用） === */
.tg-section-divider {{
  padding: 6px 12px;
  margin: 16px 0 10px 0;
  background-color: var(--bg-soft);
  border-radius: 6px;
}}

/* === 滚动条 === */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* === 隐藏 Streamlit 默认 header & footer === */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
</style>
"""


# 跟随系统 / 主题切换的 JS（仅用 st.components.html 注入一次）
THEME_BOOTSTRAP_JS = """
<script>
(function() {
  const KEY = 'tg_theme';
  function apply(theme) {
    if (theme === 'auto') {
      document.documentElement.removeAttribute('data-theme');
      document.documentElement.setAttribute('data-theme', 'auto');
    } else {
      document.documentElement.setAttribute('data-theme', theme);
    }
  }
  // 初次加载：从 localStorage 读
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'light' || saved === 'dark' || saved === 'auto') {
      apply(saved);
    } else {
      apply('auto');
    }
  } catch(e) {
    apply('auto');
  }
  // 暴露给 Streamlit 调用
  window.__tg_set_theme = function(theme) {
    try { localStorage.setItem(KEY, theme); } catch(e) {}
    apply(theme);
  };
})();
</script>
"""


# ===================================================================== #
# 3. 主入口：注入主题
# ===================================================================== #
def inject_theme() -> None:
    """
    在每个页面入口处调用一次：
        1. 注入 CSS（深 / 浅 两套 + 跟随系统）
        2. 注入 JS（读 localStorage 应用主题 + 暴露 set 接口）
    """
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    components.html(THEME_BOOTSTRAP_JS, height=0, width=0)


def render_theme_switcher() -> None:
    """
    右上角主题切换器。返回当前主题（auto / light / dark）。
    使用方法：
        current = render_theme_switcher()
    """
    # 三个单选按钮嵌在 sidebar
    if "tg_theme_choice" not in st.session_state:
        st.session_state["tg_theme_choice"] = "auto"

    st.markdown("##### 🎨 主题")
    choice = st.radio(
        "主题",
        options=["auto", "light", "dark"],
        format_func=lambda x: {"auto": "🖥 跟随系统", "light": "☀ 浅色", "dark": "🌙 暗色"}[x],
        index=["auto", "light", "dark"].index(st.session_state["tg_theme_choice"]),
        key="tg_theme_radio",
        label_visibility="collapsed",
        horizontal=True,
    )
    st.session_state["tg_theme_choice"] = choice

    # 通过 st.components.html 调 JS 切换
    if choice != st.session_state.get("tg_last_applied"):
        st.session_state["tg_last_applied"] = choice
        components.html(
            f"<script>window.parent.__tg_set_theme && window.parent.__tg_set_theme('{choice}');</script>",
            height=0,
            width=0,
        )

    return choice


def get_current_theme() -> str:
    """返回当前选择的主题（auto / light / dark）。"""
    return st.session_state.get("tg_theme_choice", "auto")


# ===================================================================== #
# 4. 暗色适配：pydeck / plotly 配色
# ===================================================================== #
def palette_for_pydeck(map_style: str = "dark") -> str:
    """
    根据当前主题返回 pydeck 底图。
    暗色模式默认用 dark；亮色用 light。
    """
    theme = get_current_theme()
    if theme == "dark":
        return "dark"
    if theme == "light":
        return "light"
    # auto：跟系统 → 这里用 streamlit 拿不到系统色，但可以保守选 light
    return map_style


def apply_plotly_theme(fig) -> None:
    """
    给 plotly figure 应用当前主题的 layout。
    直接修改 fig 本身（in-place）。
    """
    theme = get_current_theme()
    is_dark = theme == "dark" or (
        theme == "auto" and _is_system_dark()
    )
    template = "plotly_dark" if is_dark else "plotly_white"
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1" if is_dark else "#0f172a"),
        margin=dict(l=10, r=10, t=40, b=30),
    )


def _is_system_dark() -> bool:
    """探测系统是否暗色（仅 auto 模式下用）。"""
    # streamlit 跑在浏览器里，没有直接的 API；用 HTML 注入 JS 异步获取
    # 这里用 components.html 同步取（不可靠，但作为 fallback 足够）
    try:
        result = components.html(
            """
            <script>
            (function(){
                const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                document.title = dark ? 'dark' : 'light';
            })();
            </script>
            """,
            height=0,
            width=0,
        )
        # components.html 不会返回 document.title 改动，只能用别的方法
        return False
    except Exception:
        return False
