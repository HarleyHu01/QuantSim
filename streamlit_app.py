"""QuantSim Streamlit 前端入口。

启动：python -m streamlit run streamlit_app.py（或 start.bat）
后端 API 需先运行：python -m uvicorn quant.api.app:app --port 8000

导航分四个模块（外加系统设置）：

1. 数据管理 —— 标的、行情下载、数据状态
2. 策略管理 —— AI 对话生成、模型预测、手工创建、策略库、策略融合、组合配置
3. 策略回测 —— 单策略 / 多策略组合回测
4. 交易模拟 —— 模拟盘

页面之间用 ``st.session_state["nav_target"]`` 做程序化跳转（值为**页面名**，
不是模块名），由本文件在 radio 实例化前换算成 radio 的选中值。
"""
from __future__ import annotations

import streamlit as st

from web.streamlit.api import ApiClient
from web.streamlit.components import inject_theme
from web.streamlit.pages import (ai_page, backtest_page, data_page,
                                 data_settings_page, fusion_page, kronos_page,
                                 paper_page, portfolio_page, settings_page,
                                 strategies_page, strategy_create_page,
                                 strategy_settings_page)

st.set_page_config(page_title="QuantSim 量化交易模拟系统", page_icon="📈", layout="wide")
inject_theme()

# 四个模块 → 页面名（顺序即侧边栏展示顺序）
MODULES: dict[str, list[str]] = {
    "数据管理": ["数据管理"],
    "策略管理": ["AI 策略", "模型预测", "策略创建", "策略管理", "策略融合", "组合配置"],
    "策略回测": ["策略回测"],
    "交易模拟": ["交易模拟"],
    "系统设置": ["数据设置", "交易设置", "策略设置"],
}
MODULE_ICONS = {
    "数据管理": "🗂️",
    "策略管理": "🧠",
    "策略回测": "🧪",
    "交易模拟": "💼",
    "系统设置": "⚙️",
}

PAGE_MODULE = {p: m for m, pages in MODULES.items() for p in pages}
PAGES: dict[str, object] = {
    "数据管理": data_page,
    "AI 策略": ai_page,
    "模型预测": kronos_page,
    "策略创建": strategy_create_page,
    "策略管理": strategies_page,
    "策略融合": fusion_page,
    "组合配置": portfolio_page,
    "策略回测": backtest_page,
    "交易模拟": paper_page,
    "数据设置": data_settings_page,
    "交易设置": settings_page,
    "策略设置": strategy_settings_page,
}

# API 客户端（可配置 base url）
if "api" not in st.session_state:
    st.session_state["api"] = ApiClient()
api: ApiClient = st.session_state["api"]

with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:2px'>"
        "<div style='width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#22d3ee,#0ea5c4);"
        "display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#06222a'>Q</div>"
        "<div><div style='font-size:18px;font-weight:700;color:#f1f6ff;letter-spacing:-0.01em'>QuantSim</div>"
        "<div style='font-size:11.5px;color:#7d90b3'>本地量化交易模拟系统</div></div></div>",
        unsafe_allow_html=True)
    st.divider()

    # 程序化跳转：其他页面设置 nav_target（页面名）→ 换算成模块 + 子页选中值
    nav_target = st.session_state.pop("nav_target", None)
    if nav_target is not None and nav_target in PAGES:
        st.session_state["module"] = PAGE_MODULE[nav_target]
        st.session_state["subpage"] = nav_target
        st.session_state["nav"] = nav_target

    module = st.selectbox("模块", list(MODULES.keys()),
                          format_func=lambda m: f"{MODULE_ICONS[m]} {m}",
                          key="module")
    page = st.radio("页面", MODULES[module], key="subpage")

    try:
        env = api.health().get("environment", "local")
        st.success("● API 已连接")
    except Exception:  # noqa: BLE001
        env = "local"
        st.error("● API 未连接\n请先运行 start.bat 或后端服务")
    env_label = "本地运行" if env == "local" else "沙箱运行"
    env_dot = "#34d399" if env == "local" else "#fbbf24"
    st.markdown(
        f"<div style='background:rgba(17,26,46,0.85);padding:8px 12px;border-radius:9px;"
        f"border:1px solid rgba(34,211,238,0.14);font-size:12.5px;color:#c7d4ea'>"
        f"<span style='color:{env_dot};font-size:10px;margin-right:6px'>●</span>"
        f"运行环境：<b style='color:#f1f6ff'>{env_label}</b><span style='color:#7d90b3'>（{env}）</span></div>",
        unsafe_allow_html=True)
    st.caption("API: " + str(api.base) + "\n自动刷新默认关闭，间隔默认 5s（数据管理/模拟盘页可开启）")

    st.divider()
    if st.button("关闭系统", type="primary", use_container_width=True, key="shutdown_btn"):
        st.session_state["confirm_shutdown"] = True
        st.rerun()
    if st.session_state.get("confirm_shutdown"):
        st.warning("确认关闭系统？将停止 API 服务和 UI 界面。"
                   "运行中的模拟盘账户将自动停止并保存。")
        sc1, sc2 = st.columns(2)
        if sc1.button("确认关闭", type="primary", key="shutdown_confirm"):
            try:
                api.shutdown()  # 后端先停模拟盘账户并落库，随后关闭 API 进程
            except Exception:  # noqa: BLE001
                pass  # API 已开始关闭，连接可能中断
            st.info("系统已关闭，可以安全关闭此窗口")
            import os
            import threading

            threading.Timer(2.0, lambda: os._exit(0)).start()  # 关闭 UI 自身进程
            st.stop()
        if sc2.button("取消", key="shutdown_cancel"):
            st.session_state.pop("confirm_shutdown", None)
            st.rerun()

# 记录本次 rerun 是否发生了「换页」：页面模块据此做一次性的首屏动作
# （例如 AI 策略页需要主动滚回顶部）。用 pop 语义由页面自行消费。
prev_page = st.session_state.get("_active_page")
st.session_state["_page_changed"] = prev_page != page
st.session_state["_active_page"] = page

PAGES[page].render(api)
