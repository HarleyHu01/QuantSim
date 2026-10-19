"""Streamlit 通用组件：schema 驱动参数表单、指标卡片、ECharts 图表、全局主题。"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

# ================= 全局主题（量化终端视觉世界） =================

_GLOBAL_CSS = """
<style>
/* ---------- 基础排版 ---------- */
html, body, [data-testid="stAppViewContainer"] {
  font-family: "Manrope", "Outfit", -apple-system, "PingFang SC",
    "Microsoft YaHei", "Noto Sans SC", sans-serif;
  background: #0b1220;
  color: #e6edf7;
}
[data-testid="stHeader"] { background: transparent; }

/* ---------- 环境背景：简洁的高清光晕渐变（无网格/噪声装饰） ---------- */
[data-testid="stAppViewContainer"] {
  position: relative;
  background:
    radial-gradient(1200px 600px at 12% -8%, rgba(34,211,238,0.10), transparent 60%),
    radial-gradient(1000px 560px at 100% 4%, rgba(56,120,255,0.09), transparent 55%),
    radial-gradient(900px 620px at 50% 108%, rgba(16,185,129,0.07), transparent 60%),
    #0b1220;
  background-attachment: fixed;
}
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(620px 420px at 82% 18%, rgba(34,211,238,0.10), transparent 65%);
  animation: qs-glow-breathe 14s ease-in-out infinite;
}
/* 保证正文浮在背景层之上 */
[data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
[data-testid="stSidebar"] { position: relative; z-index: 2; }

@keyframes qs-grid-drift {
  0%   { background-position: 0 0; }
  100% { background-position: 0 44px; }
}
@keyframes qs-glow-breathe {
  0%, 100% { opacity: 0.5; transform: translateY(0) scale(1); }
  50%      { opacity: 1;   transform: translateY(-14px) scale(1.05); }
}
/* 时间线红点涟漪（yan.net.cn edu_timeline ripple 风格） */
@keyframes qs-ripple {
  0%   { transform: scale(1);    opacity: 0.8; }
  100% { transform: scale(2.6);  opacity: 0; }
}
.qs-timeline-dot {
  position: relative;
  display: inline-block;
  width: 12px; height: 12px;
  border-radius: 50%;
  background: #f87171;
  vertical-align: middle;
}
.qs-timeline-dot::after {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: 50%;
  background: #f87171;
  animation: qs-ripple 1.8s ease-out infinite;
  pointer-events: none;
}

/* ---------- 主区域留白与宽度 ---------- */
[data-testid="stAppViewContainer"] > .main .block-container {
  padding-top: 2.2rem;
  padding-bottom: 3rem;
  max-width: 1480px;
}

/* ---------- 页面大标题 ---------- */
h1, h2, h3 {
  letter-spacing: -0.02em;
  color: #f1f6ff;
}
h1 { font-weight: 700; }
/* 斜切视觉语言：标题与标题前的青色斜块（yan.net.cn section_header 风格） */
h2 {
  font-weight: 650; border-bottom: 1px solid rgba(34,211,238,0.18);
  padding-bottom: 0.55rem; margin-bottom: 0.4rem;
  position: relative;
}
h2::before {
  content: "";
  display: inline-block;
  width: 0.42em; height: 1.05em;
  margin-right: 0.55em;
  vertical-align: -0.12em;
  background: linear-gradient(135deg, #22d3ee, #0ea5c4);
  transform: skew(-20deg);
  border-radius: 2px;
}
h3 { font-weight: 600; }
h4, h5 { color: #9fb3d1; font-weight: 600; letter-spacing: 0.01em; }

/* ---------- 侧边栏 ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1526 0%, #0b1220 100%);
  border-right: 1px solid rgba(34,211,238,0.14);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
/* 侧边栏内容区可上下滚动：导航项多时「关闭系统」按钮可滚到可见 */
[data-testid="stSidebarContent"] {
  max-height: calc(100vh - 60px) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  scrollbar-width: thin;
  scrollbar-color: rgba(34,211,238,0.25) transparent;
}
[data-testid="stSidebarContent"]::-webkit-scrollbar { width: 8px; }
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
  background: rgba(34,211,238,0.22);
  border-radius: 6px;
}
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover { background: rgba(34,211,238,0.38); }

/* ---------- 卡片 / 容器 ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(17,26,46,0.72);
  border: 1px solid rgba(34,211,238,0.14) !important;
  border-radius: 12px !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.3);
  transition: transform 0.22s cubic-bezier(0.16,1,0.3,1),
              box-shadow 0.22s cubic-bezier(0.16,1,0.3,1),
              border-color 0.22s ease;
  will-change: transform;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(34,211,238,0.32) !important;
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.38), 0 0 0 1px rgba(34,211,238,0.12);
}
/* 毛玻璃卡：半透明白 + backdrop 模糊（yan.net.cn .sec_component_para 风格） */
[data-testid="stVerticalBlockBorderWrapper"] {
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}

/* ---------- 指标卡片 ---------- */
[data-testid="stMetric"] {
  background: rgba(17,26,46,0.85);
  border: 1px solid rgba(34,211,238,0.12);
  border-radius: 10px;
  padding: 0.8rem 1rem;
  transition: transform 0.2s cubic-bezier(0.16,1,0.3,1),
              border-color 0.2s ease,
              box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: rgba(34,211,238,0.34);
  box-shadow: 0 10px 26px rgba(0,0,0,0.4);
}
[data-testid="stMetricLabel"] { color: #9fb3d1; font-weight: 500; }
[data-testid="stMetricValue"] { color: #f1f6ff; font-weight: 650; }

/* ---------- 按钮 ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 8px;
  font-weight: 600;
  transition: transform 0.18s cubic-bezier(0.16,1,0.3,1),
              box-shadow 0.18s ease, filter 0.18s ease,
              background-color 0.18s ease, color 0.18s ease,
              border-color 0.18s ease;
}
/* 普通按钮：悬停上浮 + 描边亮起 + 轻微发光（明确的视觉反馈） */
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px);
  border-color: rgba(34,211,238,0.6) !important;
  color: #22d3ee !important;
  box-shadow: 0 6px 18px rgba(34,211,238,0.22);
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: translateY(0) scale(0.97);
  box-shadow: none;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 2px solid rgba(34,211,238,0.7);
  outline-offset: 2px;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(180deg, #22d3ee 0%, #0ea5c4 100%);
  color: #06222a;
  border: none;
  background-size: 100% 100%;
  transform: skew(-8deg);
}
/* 主按钮斜切：内文字反斜切校正（yan.net.cn theme_default_btn 风格） */
.stButton > button[kind="primary"] > div[data-testid="stMarkdownContainer"],
.stButton > button[kind="primary"] p,
.stDownloadButton > button[kind="primary"] > div {
  transform: skew(8deg);
  display: inline-block;
}
.stButton > button[kind="primary"]:hover {
  filter: brightness(1.1);
  box-shadow: 0 0 0 1px rgba(34,211,238,0.5), 0 8px 22px rgba(34,211,238,0.36);
  transform: translateY(-2px) skew(-8deg);
}
.stButton > button[kind="primary"]:active {
  filter: brightness(0.96);
  transform: translateY(0) scale(0.97) skew(-8deg);
}

/* ---------- 输入控件 ---------- */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
.stTextArea textarea, [data-testid="stSelectbox"] > div,
[data-baseweb="select"] > div {
  border-radius: 8px !important;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus,
.stTextArea textarea:focus {
  border-color: rgba(34,211,238,0.6) !important;
  box-shadow: 0 0 0 3px rgba(34,211,238,0.14);
}

/* ---------- 分隔线 ---------- */
hr { border-color: rgba(34,211,238,0.14); }

/* ---------- expander ---------- */
[data-testid="stExpander"] {
  border: 1px solid rgba(34,211,238,0.14);
  border-radius: 10px;
  transition: border-color 0.2s ease;
}
[data-testid="stExpander"]:hover { border-color: rgba(34,211,238,0.3); }

/* ---------- 提示框圆角 ---------- */
[data-testid="stAlert"] { border-radius: 10px; }

/* ---------- 表格 ---------- */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ---------- 侧边栏导航 radio 悬停反馈 ---------- */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  transition: transform 0.16s ease, opacity 0.16s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  transform: translateX(3px);
}

/* ---------- 涨跌语义色 ---------- */
.up   { color: #f87171; }
.down { color: #34d399; }

/* ---------- 焦点可见性（键盘可达性） ---------- */
:focus-visible {
  outline: 2px solid rgba(34,211,238,0.7);
  outline-offset: 2px;
}

/* ---------- 文本选择 / 滚动条（浏览器表面也要承载设计） ---------- */
::selection { background: rgba(34,211,238,0.28); color: #f1f6ff; }
* { scrollbar-width: thin; scrollbar-color: rgba(34,211,238,0.25) transparent; }
*::-webkit-scrollbar { width: 9px; height: 9px; }
*::-webkit-scrollbar-thumb {
  background: rgba(34,211,238,0.22);
  border-radius: 6px;
}
*::-webkit-scrollbar-thumb:hover { background: rgba(34,211,238,0.38); }

/* ---------- 减弱动效：尊重系统偏好，保留语义反馈 ---------- */
@media (prefers-reduced-motion: reduce) {
  [data-testid="stAppViewContainer"]::before,
  [data-testid="stAppViewContainer"]::after {
    animation: none;
  }
  [data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stMetric"],
  .stButton > button, .stDownloadButton > button,
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    transition: none;
  }
  [data-testid="stVerticalBlockBorderWrapper"]:hover,
  [data-testid="stMetric"]:hover,
  .stButton > button:hover {
    transform: none;
  }
}
</style>
"""


def inject_theme() -> None:
    """注入全局主题 CSS。每个页面 render 前调用一次（幂等，streamlit 自动去重）。"""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def scroll_to_top(times: tuple[int, ...] = (0, 60, 200, 500)) -> None:
    """把页面主滚动区滚回顶部（用于「打开页面应从顶部开始」的场合）。

    Streamlit 在 rerun 后会保留浏览器滚动位置，Python 侧无法控制，
    只能用一次性的内嵌脚本把滚动容器归零。这里按多个时间点重复执行，
    因为首屏元素是分批挂载的，只在 DOMContentLoaded 时滚一次会被随后的
    布局撑开又推回原位。
    """
    delays = ",".join(str(int(t)) for t in times)
    st.components.v1.html(
        f"""<script>
        (function () {{
          const delays = [{delays}];
          const selectors = ['section.main', '[data-testid="stMain"]',
                             '[data-testid="stAppViewContainer"]',
                             '[data-testid="stMainBlockContainer"]', 'body'];
          function reset() {{
            try {{
              const doc = window.parent.document;
              for (const s of selectors) {{
                const el = doc.querySelector(s);
                if (el) el.scrollTop = 0;
              }}
              window.parent.scrollTo(0, 0);
            }} catch (e) {{ /* 跨域或容器不存在时静默 */ }}
          }}
          for (const d of delays) setTimeout(reset, d);
        }})();
        </script>""",
        height=0,
    )


TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"]
# 各周期的通用提示（UI 用）。刻意**不写任何具体根数/天数**：
# 同一接口在不同时段、不同标的、不同网络下的实际可返回量并不相同，
# 把某台机器某次实测的数字写死在界面上，换一台电脑就会变成误导。
TF_COVERAGE_HINT = {
    "1m": "最短周期，历史深度最浅，通常只够看最近一两个交易日",
    "5m": "分钟线，历史深度取决于数据源当下的限制",
    "15m": "分钟线，历史深度取决于数据源当下的限制",
    "30m": "分钟线，历史深度取决于数据源当下的限制",
    "1h": "分钟线，历史深度取决于数据源当下的限制",
    "1d": "日线，多数数据源可回溯到上市首日",
    "1w": "周线，多数数据源可回溯到上市首日",
    "1M": "最长周期，通常能覆盖全部历史",
}
MODE_LABELS = {"signal_merge": "信号加权", "capital_split": "资金分配",
               "equal": "等权"}
ENGINE_LABELS = {"event_driven": "事件驱动回测 (event_driven)", "vectorized": "向量化回测 (vectorized)"}
EXEC_LABELS = {"next_open": "下一根开盘成交", "same_close": "当根收盘成交"}
ADJUST_LABELS = {"qfq": "前复权 (qfq)", "hfq": "后复权 (hfq)", "none": "不复权 (none)"}
METRIC_LABELS = {
    "total_return": "总收益率", "annual_return": "年化收益率", "max_drawdown": "最大回撤",
    "sharpe": "夏普比率", "sortino": "索提诺比率", "calmar": "卡玛比率",
    "win_rate": "胜率", "profit_factor": "盈亏比", "trade_count": "交易次数",
    "avg_trade_pnl": "平均盈亏", "avg_exposure": "平均暴露度", "benchmark_return": "基准收益",
    "volatility": "年化波动率", "elapsed_seconds": "耗时（秒）",
    # 对比指标
    "buyhold_return": "买入持有收益", "buyhold_annual": "买入持有年化",
    "excess_return_vs_buyhold": "超额收益(vs买入持有)",
    "benchmark_annual": "基准年化", "benchmark_max_drawdown": "基准最大回撤",
    "beta": "Beta", "alpha": "Alpha(年化)", "information_ratio": "信息比率",
}
PAPER_STATUS_LABELS = {"running": "运行中", "stopped": "已停止"}

# 时间周期中英文映射（注意 1M 在文件名中是 1mo，UI 展示必须还原为 1M/月线）
TF_LABELS = {
    "1m": "1分钟 (1 Minute)",
    "5m": "5分钟 (5 Minutes)",
    "15m": "15分钟 (15 Minutes)",
    "30m": "30分钟 (30 Minutes)",
    "1h": "1小时 (1 Hour)",
    "1d": "日线 (Daily)",
    "1w": "周线 (Weekly)",
    "1M": "月线 (Monthly)",
}


def tf_label(tf: str) -> str:
    return TF_LABELS.get(tf, tf)


def strategy_label(s: dict) -> str:
    """策略显示名：中英文全称（display_name），缺省回退 name/key。"""
    return s.get("display_name") or s.get("name") or s.get("key", "")


def adjust_label(a: str) -> str:
    return ADJUST_LABELS.get(a, a)


def metric_label(k: str) -> str:
    return METRIC_LABELS.get(k, k)


def echarts(option: dict, height: int = 380) -> None:
    """内嵌 ECharts（CDN），option 为 ECharts 配置。"""
    opt_json = json.dumps(option, ensure_ascii=False)
    html = f"""
    <div id="chart" style="width:100%;height:{height}px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <script>
      var c = echarts.init(document.getElementById('chart'));
      c.setOption({opt_json});
      window.addEventListener('resize', function() {{ c.resize(); }});
    </script>
    """
    st.components.v1.html(html, height=height + 8, scrolling=False)


# ================= 策略参数表单（按 schema 渲染） =================

def render_params_form(schema: dict, current: dict | None, prefix: str = "p") -> dict:
    """按 params_schema 自动渲染控件，返回当前值字典。

    - bool → checkbox；integer/number 带范围 → slider（范围大则 number_input）
    - string 带 enum → selectbox；str → text_input
    """
    props = (schema or {}).get("properties", {})
    current = current or {}
    values: dict = {}
    for name, spec in props.items():
        title = spec.get("title", name)
        default = spec.get("default")
        val = current.get(name, default)
        ptype = spec.get("type")
        key = f"{prefix}_{name}"
        lo, hi = spec.get("minimum"), spec.get("maximum")

        # 可空字段（int|None 等 → anyOf 含 null）：解析实际类型
        nullable = False
        if ptype is None and isinstance(spec.get("anyOf"), list):
            non_null = [t for t in spec["anyOf"] if t.get("type") != "null"]
            if non_null:
                nullable = True
                ptype = non_null[0].get("type")
                lo = non_null[0].get("minimum", lo)
                hi = non_null[0].get("maximum", hi)
                default = non_null[0].get("default", default)
        if val is None and nullable:
            continue  # 可选字段未设置：跳过，避免 str(None) 污染配置

        if ptype == "boolean":
            values[name] = st.checkbox(title, value=bool(val), key=key)
        elif ptype == "integer":
            if lo is not None and hi is not None and (hi - lo) <= 500:
                values[name] = st.slider(title, int(lo), int(hi), int(val), key=key)
            else:
                values[name] = int(st.number_input(
                    title, value=int(val), min_value=int(lo) if lo is not None else None,
                    max_value=int(hi) if hi is not None else None, step=1, key=key))
        elif ptype == "number":
            if lo is not None and hi is not None:
                step = 0.01 if (hi - lo) <= 2 else 0.1
                values[name] = st.slider(title, float(lo), float(hi), float(val), step, key=key)
            else:
                values[name] = st.number_input(title, value=float(val), step=0.01,
                                               format="%.4f", key=key)
        elif ptype == "string":
            enum = spec.get("enum")
            if enum:
                idx = enum.index(val) if val in enum else 0
                values[name] = st.selectbox(title, enum, index=idx, key=key)
            else:
                values[name] = st.text_input(title, value=str(val), key=key)
        else:
            values[name] = st.text_input(title, value=str(val), key=key)
    return values


# ================= 指标卡片 =================

def metrics_row(metrics: dict) -> None:
    def pct(v, digits=2):
        return "—" if v is None else f"{v * 100:.{digits}f}%"

    c = st.columns(6)
    c[0].metric("总收益率", pct(metrics.get("total_return")))
    c[1].metric("年化收益率", pct(metrics.get("annual_return")))
    c[2].metric("最大回撤", pct(metrics.get("max_drawdown")))
    c[3].metric("夏普比率", f"{metrics.get('sharpe'):.3f}" if metrics.get("sharpe") is not None else "—")
    c[4].metric("胜率", pct(metrics.get("win_rate")))
    c[5].metric("盈亏比", f"{metrics.get('profit_factor'):.2f}" if metrics.get("profit_factor") is not None else "—")
    c2 = st.columns(6)
    c2[0].metric("交易次数", int(metrics.get("trade_count", 0)))
    c2[1].metric("索提诺比率", f"{metrics.get('sortino'):.3f}" if metrics.get("sortino") is not None else "—")
    c2[2].metric("卡玛比率", f"{metrics.get('calmar'):.3f}" if metrics.get("calmar") is not None else "—")
    c2[3].metric("暴露度", pct(metrics.get("avg_exposure")))
    c2[4].metric("年化波动", pct(metrics.get("volatility")))
    c2[5].metric("基准收益", pct(metrics.get("benchmark_return")))
    # 第三行：对比指标（策略 vs 买入持有 vs 基准）
    c3 = st.columns(6)
    c3[0].metric("超额收益(vs买入持有)", pct(metrics.get("excess_return_vs_buyhold")))
    c3[1].metric("买入持有年化", pct(metrics.get("buyhold_annual")))
    c3[2].metric("基准最大回撤", pct(metrics.get("benchmark_max_drawdown")))
    c3[3].metric("Beta", f"{metrics.get('beta'):.3f}" if metrics.get("beta") is not None else "—")
    c3[4].metric("Alpha(年化)", pct(metrics.get("alpha")))
    c3[5].metric("信息比率", f"{metrics.get('information_ratio'):.3f}" if metrics.get("information_ratio") is not None else "—")
    if metrics.get("elapsed_seconds") is not None:
        st.caption(f"耗时 {metrics['elapsed_seconds']}s")


# ================= ECharts 图表 =================

def _ts_list(rows: list[dict], col: str = "ts") -> list[str]:
    return [str(r.get(col, ""))[:16] for r in rows]


def comparison_option(eq_rows: list[dict]) -> dict:
    """归一化收益对比（初始=100%）：策略收益 / 买入持有 / 基准指数 / 标的股价。

    各序列以其首值归一化，直观对比"策略是否跑赢无脑持有/大盘"。
    """
    x = _ts_list(eq_rows)
    lines = [
        ("策略收益", "equity", "#1f77b4", "solid", 2),
        ("买入持有", "buyhold", "#2ca02c", "dashed", 1),
        ("基准指数", "benchmark", "#ff7f0e", "dashed", 1),
        ("标的股价", "symbol_price", "#9467bd", "dotted", 1),
    ]
    series, legend = [], []
    for name, key, color, ltype, width in lines:
        vals = [r.get(key) for r in eq_rows]
        first = next((v for v in vals if v is not None and v), None)
        if first is None:
            continue  # 旧回测无该列 → 跳过该线
        data = [None if v is None else round(v / first * 100, 3) for v in vals]
        series.append({"name": name, "type": "line", "data": data, "showSymbol": False,
                       "color": color,
                       "lineStyle": {"width": width, "type": ltype}})
        legend.append(name)
    return {
        "title": {"text": "收益对比（归一化，初始=100%）", "left": "center",
                  "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis", "valueFormatter": "{value}%"},
        "legend": {"data": legend, "top": 26},
        "grid": {"left": 60, "right": 20, "top": 60, "bottom": 30},
        "xAxis": {"type": "category", "data": x, "axisLabel": {"fontSize": 10}},
        "yAxis": {"type": "value", "scale": True, "axisLabel": {"fontSize": 10,
                                                                "formatter": "{value}%"}},
        "series": series,
    }


def drawdown_option(eq_rows: list[dict]) -> dict:
    """回撤对比（%）：策略回撤 vs 基准回撤。"""
    x = _ts_list(eq_rows)
    dd = [round((r.get("drawdown") or 0) * 100, 3) for r in eq_rows]
    # 基准回撤：1 − b/cummax(b)
    bench_dd, peak = [], None
    for r in eq_rows:
        b = r.get("benchmark")
        if b is None:
            bench_dd.append(None)
            continue
        peak = b if peak is None else max(peak, b)
        bench_dd.append(round((1.0 - b / peak) * 100, 3) if peak else 0.0)
    return {
        "title": {"text": "回撤对比（%）", "left": "center", "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis", "valueFormatter": "{value}%"},
        "legend": {"data": ["策略回撤", "基准回撤"], "top": 26},
        "grid": {"left": 60, "right": 20, "top": 60, "bottom": 30},
        "xAxis": {"type": "category", "data": x, "axisLabel": {"fontSize": 10}},
        "yAxis": {"type": "value", "axisLabel": {"fontSize": 10, "formatter": "{value}%"}},
        "series": [
            {"name": "策略回撤", "type": "line", "data": dd, "showSymbol": False,
             "lineStyle": {"color": "#d62728", "width": 2},
             "areaStyle": {"color": "rgba(214,39,40,0.15)"}},
            {"name": "基准回撤", "type": "line", "data": bench_dd, "showSymbol": False,
             "lineStyle": {"color": "#ff7f0e", "width": 1, "type": "dashed"},
             "areaStyle": {"color": "rgba(255,127,14,0.08)"}},
        ],
    }


def monthly_heatmap_option(eq_rows: list[dict], series_key: str = "equity") -> dict:
    """月度收益热力图（%）：权益曲线按月复合收益的年 × 月矩阵。"""
    ts = pd.to_datetime([r.get("ts") for r in eq_rows], errors="coerce")
    vals = pd.Series([r.get(series_key) for r in eq_rows], index=ts, dtype="float64").dropna()
    if len(vals) < 3:
        return {}
    norm = vals / vals.iloc[0]
    rets = norm.pct_change().dropna()
    monthly = rets.resample("ME").apply(lambda s: (1.0 + s).prod() - 1.0) * 100.0
    if monthly.empty:
        return {}
    m = monthly.to_frame("ret")
    m["year"], m["month"] = m.index.year, m.index.month
    years = sorted(m["year"].unique())
    yidx = {y: i for i, y in enumerate(years)}
    data = [[yidx[int(row.year)], int(row.month) - 1, round(float(row.ret), 2)]
            for row in m.itertuples()]
    vmax = max(abs(float(v)) for v in m["ret"]) or 1.0
    return {
        "title": {"text": "策略月度收益（%）", "left": "center", "textStyle": {"fontSize": 13}},
        "tooltip": {"position": "top",
                    "formatter": "function(p){return p.value[2] + '%';}"},
        "grid": {"left": 40, "right": 20, "top": 60, "bottom": 60},
        "xAxis": {"type": "category", "data": [f"{i}月" for i in range(1, 13)],
                  "splitArea": {"show": True}, "axisLabel": {"fontSize": 10}},
        "yAxis": {"type": "category", "data": [str(y) for y in years],
                  "splitArea": {"show": True}, "axisLabel": {"fontSize": 10}},
        "visualMap": {"min": -vmax, "max": vmax, "calculable": True, "orient": "horizontal",
                      "left": "center", "bottom": 0, "text": ["高", "低"],
                      "inRange": {"color": ["#14b143", "#f5f5f5", "#ef232a"]}},
        "series": [{"type": "heatmap", "data": data,
                    "label": {"show": True, "fontSize": 10, "formatter": "{c}%"}}],
    }


def kline_option(bars: list[dict], trades: list[dict], symbol: str) -> dict:
    """K 线 + 成交量 + 买卖点标记。"""
    if not bars:
        return {}
    dates = [str(b.get("datetime", ""))[:16] for b in bars]
    kdata = [[b.get("open"), b.get("close"), b.get("low"), b.get("high")] for b in bars]
    vols = [{"value": b.get("volume") or 0, "itemStyle": {"color": "#ef232a" if (b.get("close") or 0) >= (b.get("open") or 0) else "#14b143"}} for b in bars]

    def markers(side: str, color: str):
        """买卖点：红色上箭头=买入（标在最低价下方），绿色图钉=卖出（标在最高价上方）。"""
        pts = []
        for t in trades:
            if t.get("side") != side:
                continue
            ts = str(t.get("timestamp", ""))[:16]
            if ts in dates:
                bar = bars[dates.index(ts)]
                y = bar.get("low") * 0.995 if side == "buy" else bar.get("high") * 1.005
                pts.append({
                    "coord": [dates.index(ts), y],
                    "value": f"{'买入' if side == 'buy' else '卖出'} {t.get('qty'):.0f}@{t.get('price'):.2f}",
                    "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 1},
                    "label": {"show": True, "formatter": "B" if side == "buy" else "S",
                              "position": "top", "fontWeight": "bold", "color": color},
                })
        return pts

    return {
        "title": {"text": f"{symbol} K线 + 买卖点 + 成交量", "left": "center", "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"data": ["K线", "买入", "卖出", "成交量"], "top": 26},
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "grid": [{"left": 60, "right": 20, "top": 60, "height": "46%"},
                 {"left": 60, "right": 20, "top": "70%", "height": "16%"}],
        "xAxis": [
            {"type": "category", "data": dates, "axisLabel": {"fontSize": 9}},
            {"type": "category", "gridIndex": 1, "data": dates, "axisLabel": {"show": False}},
        ],
        "yAxis": [
            {"scale": True, "axisLabel": {"fontSize": 9}},
            {"gridIndex": 1, "axisLabel": {"fontSize": 9}},
        ],
        "dataZoom": [{"type": "inside", "xAxisIndex": [0, 1], "start": 0, "end": 100}],
        "series": [
            {"name": "K线", "type": "candlestick", "data": kdata,
             "itemStyle": {"color": "#ef232a", "color0": "#14b143",
                           "borderColor": "#ef232a", "borderColor0": "#14b143"}},
            {"name": "买入", "type": "scatter", "data": markers("buy", "#ef232a"),
             "symbol": "triangle", "symbolSize": 16, "z": 20},
            {"name": "卖出", "type": "scatter", "data": markers("sell", "#14b143"),
             "symbol": "pin", "symbolSize": 18, "z": 20},
            {"name": "成交量", "type": "bar", "xAxisIndex": 1, "yAxisIndex": 1, "data": vols},
        ],
    }


def forecast_option(payload: dict) -> dict:
    """历史收盘价 + Kronos 预测区间（预测部分用虚线并把置信带画出来）。

    payload 来自 POST /api/ml/kronos/predict：
        {history: [{t, close}], forecast: [{t, open, high, low, close}], symbol, timeframe}
    """
    history = payload.get("history") or []
    forecast = payload.get("forecast") or []
    if not history and not forecast:
        return {}

    h_ts = [str(r["t"])[:16] for r in history]
    f_ts = [str(r["t"])[:16] for r in forecast]
    labels = h_ts + f_ts
    # 实测段：真实收盘价；预测段留空，折线在衔接处自然断开
    actual = [r.get("close") for r in history] + [None] * len(f_ts)
    # 预测段：起点接上真实最后一点，视觉上连成一条线
    pred = [None] * len(h_ts) + [r.get("close") for r in forecast]
    if history and forecast:
        pred[len(h_ts) - 1] = history[-1].get("close")
    # 预测区间用高低两条淡虚线表示（比 areaStyle 更直观，也不依赖轴类型）
    hi = [None] * len(h_ts) + [r.get("high") for r in forecast]
    lo = [None] * len(h_ts) + [r.get("low") for r in forecast]

    return {
        "title": {
            "text": f"{payload.get('symbol', '')} {payload.get('timeframe', '')} · Kronos 预测",
            "left": "center",
            "textStyle": {"fontSize": 13},
        },
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["实际收盘", "预测收盘", "预测最高", "预测最低"], "top": 24},
        "grid": {"left": 62, "right": 24, "top": 56, "bottom": 44},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"fontSize": 9}},
        "yAxis": {"scale": True, "axisLabel": {"fontSize": 9}},
        "dataZoom": [{"type": "inside", "start": 0, "end": 100}],
        "series": [
            {"name": "实际收盘", "type": "line", "data": actual, "showSymbol": False,
             "lineStyle": {"width": 1.6, "color": "#1f77b4"}},
            {"name": "预测收盘", "type": "line", "data": pred, "showSymbol": True,
             "symbolSize": 4,
             "lineStyle": {"width": 2, "type": "dashed", "color": "#d62728"},
             "itemStyle": {"color": "#d62728"}},
            {"name": "预测最高", "type": "line", "data": hi, "showSymbol": False,
             "lineStyle": {"width": 1, "type": "dotted", "color": "#f0a0a0"}},
            {"name": "预测最低", "type": "line", "data": lo, "showSymbol": False,
             "lineStyle": {"width": 1, "type": "dotted", "color": "#a0c8f0"}},
        ],
    }


# ---- 数据源代理模式（系统设置 → 数据源代理 与 数据管理页共用）----
PROXY_MODE_LABELS = {
    "auto": "跟随系统",
    "direct": "强制直连（不走代理）",
    "proxy": "强制走代理",
}
PROXY_MODE_HINTS = {
    "auto": "跟随系统 HTTP(S)_PROXY 与 NO_PROXY",
    "direct": "绕过所有代理，直接访问（适合国内站点）",
    "proxy": "忽略 NO_PROXY，显式走系统代理（适合需翻墙的国外站点）",
}


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()
