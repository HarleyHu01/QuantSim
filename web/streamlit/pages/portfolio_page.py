"""组合配置页：模式选择、权重滑杆（-100~100 步长 5）、信号分解预览。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from web.streamlit.components import (MODE_LABELS, inject_theme, metrics_row,
                                      strategy_label, to_dataframe)


def render(api) -> None:
    inject_theme()
    st.header("组合配置")
    strategies = api.strategies()
    saved = api.portfolio_get()
    mode_keys = list(MODE_LABELS.keys())
    mode = st.selectbox("组合模式", mode_keys,
                        format_func=lambda k: f"{MODE_LABELS[k]}（{k}）",
                        index=mode_keys.index(saved["mode"]) if saved["mode"] in mode_keys else 0,
                        key="pf_mode")

    st.markdown("#### 策略权重（-100% ~ 100%，步长 5%，负值 = 反向/做空）")
    weights: dict[str, int] = {}
    cols = st.columns(2)
    for i, s in enumerate(strategies):
        w = int(saved["weights"].get(s["key"], 0))
        weights[s["key"]] = cols[i % 2].slider(
            f"{strategy_label(s)}（{s['key']}）", -100, 100, w, 5, key=f"w_{s['key']}")

    c1, c2 = st.columns(2)
    threshold = c1.slider("信号阈值 θ（目标仓位变化小于该值不交易）", 0.0, 0.5,
                          float(saved.get("signal_threshold", 0.0)), 0.01, key="pf_th")
    freq = c2.selectbox("再平衡频率", ["none", "daily", "weekly", "monthly"],
                        index=["none", "daily", "weekly", "monthly"].index(saved.get("rebalance_freq", "none")),
                        key="pf_freq")

    if st.button(" 保存组合配置", type="primary"):
        api.portfolio_put({"mode": mode, "weights": weights,
                           "rebalance_freq": freq, "signal_threshold": threshold})
        st.success("已保存（下次回测/模拟盘默认使用）")

    st.divider()
    st.markdown("#### 组合信号分解预览")
    st.caption("用当前组合配置跑一次快速回测，展示组合指标 + 各策略独立指标 + 最近合成信号")
    active = [s for s in strategies if weights.get(s["key"], 0) != 0]
    if st.button(" 运行预览回测", key="pf_run"):
        status = api.data_status()
        symbols = [m["symbol"] for m in status.get("symbols", []) if m.get("rows")]
        if not symbols or not active:
            st.warning("需要至少一个有数据的标的和一个非零权重策略")
        else:
            with st.spinner("回测中..."):
                body = {"symbols": symbols[:2], "timeframe": "1d",
                        "strategy_ids": [s["key"] for s in active],
                        "portfolio": {"mode": mode, "weights": {k: float(v) for k, v in weights.items()},
                                      "signal_threshold": threshold}}
                r = api.backtest_combo(body)
            st.session_state["pf_preview"] = r

    r = st.session_state.get("pf_preview")
    if r:
        metrics_row(r["metrics"])
        st.markdown("##### 各策略独立指标（同设置单跑）")
        pm = pd.DataFrame(r["per_strategy_metrics"]).T
        if not pm.empty:
            show = [c for c in ["total_return", "max_drawdown", "sharpe", "win_rate", "trade_count"] if c in pm.columns]
            st.dataframe(pm[show], width='stretch')
        st.markdown("##### 最近合成信号（组合目标仓位）")
        for sym, tail in (r.get("combined_signal_tail") or {}).items():
            st.caption(f"标的 {sym}")
            st.dataframe(to_dataframe(tail), width='stretch', height=200)
