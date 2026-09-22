"""QuantSim Streamlit 前端入口（第一版）。

启动：python -m streamlit run streamlit_app.py（或 start.bat）
后端 API 需先运行：python -m uvicorn quant.api.app:app --port 8000
"""
from __future__ import annotations

import streamlit as st

from web.streamlit.api import ApiClient
from web.streamlit.pages import (backtest_page, data_page, paper_page,
                                 portfolio_page, settings_page, strategies_page,
                                 strategy_create_page)

st.set_page_config(page_title="QuantSim 量化交易模拟系统", page_icon="", layout="wide")

PAGES = {
    "数据管理": data_page,
    "策略创建": strategy_create_page,
    "策略管理": strategies_page,
    "组合配置": portfolio_page,
    "交易设置": settings_page,
    "回测": backtest_page,
    "模拟盘": paper_page,
}

# API 客户端（可配置 base url）
if "api" not in st.session_state:
    st.session_state["api"] = ApiClient()
api: ApiClient = st.session_state["api"]

with st.sidebar:
    st.title(" QuantSim")
    st.caption("本地量化交易模拟系统")
    # 程序化跳转：其他页面设置 nav_target → 在 radio 实例化前写入 nav
    nav_target = st.session_state.pop("nav_target", None)
    if nav_target is not None and nav_target in PAGES:
        st.session_state["nav"] = nav_target
    page = st.radio("导航", list(PAGES.keys()), key="nav")
    try:
        env = api.health().get("environment", "local")
        st.success("API 已连接")
    except Exception:  # noqa: BLE001
        env = "local"
        st.error("API 未连接\n请先运行 start.bat 或后端服务")
    env_label = "本地运行" if env == "local" else "沙箱运行"
    bg = "#e6f4ea" if env == "local" else "#fff3e0"
    st.markdown(
        f"<div style='background:{bg};padding:6px 10px;border-radius:6px;"
        f"border:1px solid #ccc;font-size:13px'>运行环境：<b>{env_label}</b>（{env}）</div>",
        unsafe_allow_html=True)
    st.divider()
    st.caption(f"API: {api.base}\n自动刷新默认关闭，间隔默认 5s（数据管理/模拟盘页可开启）")

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

PAGES[page].render(api)
