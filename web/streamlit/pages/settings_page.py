"""交易设置页：TradingSettings 全字段表单（按类别分组），保存到 /api/settings/trading。"""
from __future__ import annotations

import streamlit as st

from web.streamlit.components import EXEC_LABELS, TIMEFRAMES, tf_label

PCT_FIELDS = {
    "max_position_pct": (0.0, 1.0, "单标的最大仓位（权益比例）"),
    "max_exposure_pct": (0.0, 2.0, "总敞口上限"),
    "max_drawdown_halt": (0.0, 1.0, "最大回撤熔断（0 = 关闭）"),
    "stop_loss_pct": (0.0, 1.0, "单笔止损（0 = 关闭）"),
    "take_profit_pct": (0.0, 5.0, "单笔止盈（0 = 关闭）"),
    "daily_loss_limit_pct": (0.0, 1.0, "每日最大亏损（0 = 关闭）"),
    "margin_rate": (0.0, 1.0, "做空保证金率"),
    "margin_call_threshold": (0.0, 1.0, "强制回补阈值（亏损/冻结）"),
}


def render(api) -> None:
    st.header("交易设置")
    s = api.settings_trading()

    st.markdown("#### 资金")
    s["initial_capital"] = st.number_input("初始资金", 1_000.0, 1e12,
                                           float(s.get("initial_capital", 1_000_000.0)),
                                           step=10_000.0, format="%.0f", key="ts_cap")

    st.markdown("#### 手续费")
    c = st.columns(3)
    s["commission_rate"] = c[0].number_input("比例佣金（万分之）", 0.0, 100.0,
                                             float(s.get("commission_rate", 0.00025)) * 10000,
                                             0.1, key="ts_cr") / 10000
    s["commission_fixed"] = c[1].number_input("固定佣金/笔（0=按比例）", 0.0, 1000.0,
                                              float(s.get("commission_fixed", 0.0)), 0.5,
                                              key="ts_cf")
    s["commission_min"] = c[2].number_input("最低佣金", 0.0, 1000.0,
                                            float(s.get("commission_min", 5.0)), 0.5, key="ts_cm")
    c = st.columns(3)
    s["stamp_tax_rate"] = c[0].number_input("卖出印花税（万分之，0=关）", 0.0, 100.0,
                                            float(s.get("stamp_tax_rate", 0.0005)) * 10000,
                                            0.1, key="ts_st") / 10000
    s["transfer_fee_rate"] = c[1].number_input("过户费（万分之，0=关）", 0.0, 100.0,
                                               float(s.get("transfer_fee_rate", 0.00001)) * 10000,
                                               0.1, key="ts_tf") / 10000
    s["slippage_bps"] = c[2].number_input("滑点（bp）", 0.0, 1000.0,
                                          float(s.get("slippage_bps", 1.0)), 0.5, key="ts_sp")

    st.markdown("#### 交易频度")
    c = st.columns(3)
    tf_idx = TIMEFRAMES.index(s.get("timeframe", "1d")) if s.get("timeframe") in TIMEFRAMES else 8
    s["timeframe"] = c[0].selectbox("周期", TIMEFRAMES, index=tf_idx,
                                    format_func=tf_label, key="ts_tf0")
    s["execution"] = c[1].selectbox("成交时点", list(EXEC_LABELS.keys()),
                                    format_func=lambda k: EXEC_LABELS[k],
                                    index=0 if s.get("execution") != "same_close" else 1, key="ts_ex")
    s["rebalance_freq"] = c[2].selectbox("再平衡", ["none", "daily", "weekly", "monthly"],
                                         key="ts_rb")
    c = st.columns(3)
    s["min_trade_interval_bars"] = int(c[0].number_input("最小交易间隔（根）", 1, 1000,
                                                         int(s.get("min_trade_interval_bars", 1)),
                                                         key="ts_mi"))
    s["cooldown_bars"] = int(c[1].number_input("成交冷却（根）", 0, 1000,
                                               int(s.get("cooldown_bars", 0)), key="ts_cd"))
    s["signal_threshold"] = c[2].number_input("信号阈值 θ", 0.0, 0.5,
                                              float(s.get("signal_threshold", 0.0)), 0.01,
                                              key="ts_th")

    st.markdown("#### 仓位与规则")
    c = st.columns(2)
    s["max_position_pct"] = c[0].slider("单标的最大仓位", 0.0, 1.0,
                                        float(s.get("max_position_pct", 0.3)), 0.05, key="ts_mp")
    s["max_exposure_pct"] = c[1].slider("总敞口上限", 0.0, 2.0,
                                        float(s.get("max_exposure_pct", 1.0)), 0.05, key="ts_me")
    c = st.columns(2)
    s["allow_short"] = c[0].checkbox("允许做空", value=bool(s.get("allow_short", False)),
                                     key="ts_as")
    s["t_plus_enabled"] = c[1].checkbox("启用 T+0/T+1（按标的属性）",
                                        value=bool(s.get("t_plus_enabled", True)), key="ts_t1")

    st.markdown("#### 做空保证金")
    c = st.columns(2)
    s["margin_rate"] = c[0].slider("保证金率", 0.0, 1.0, float(s.get("margin_rate", 0.5)),
                                   0.05, key="ts_mr")
    s["margin_call_threshold"] = c[1].slider("强制回补阈值（亏损/冻结）", 0.0, 1.0,
                                             float(s.get("margin_call_threshold", 0.8)),
                                             0.05, key="ts_mc")

    st.markdown("#### 风控（0 = 关闭）")
    c = st.columns(3)
    s["max_drawdown_halt"] = c[0].slider("最大回撤熔断", 0.0, 1.0,
                                         float(s.get("max_drawdown_halt") or 0.0), 0.01, key="ts_dd")
    s["stop_loss_pct"] = c[1].slider("单笔止损", 0.0, 1.0,
                                     float(s.get("stop_loss_pct") or 0.0), 0.01, key="ts_sl")
    s["take_profit_pct"] = c[2].slider("单笔止盈", 0.0, 5.0,
                                       float(s.get("take_profit_pct") or 0.0), 0.05, key="ts_tp")
    c = st.columns(3)
    s["daily_loss_limit_pct"] = c[0].slider("每日最大亏损", 0.0, 1.0,
                                            float(s.get("daily_loss_limit_pct") or 0.0), 0.01,
                                            key="ts_dl")
    s["max_positions"] = c[1].number_input("最大持仓数（0=不限）", 0, 100,
                                           int(s.get("max_positions") or 0), key="ts_mx")
    s["periods_per_year"] = int(c[2].number_input("年化周期数", 1, 100000,
                                                  int(s.get("periods_per_year", 252)), key="ts_pp"))

    st.markdown("#### 基准")
    st.caption("基准用于对比策略是否跑赢大盘（缺失时回测会自动拉取）")
    bench_presets = ["SPY", "QQQ", "000300.SS", "000905.SS", "BTC-USD"]
    current_bs = s.get("benchmark_symbol", "SPY") or "SPY"
    options = bench_presets + ["自定义..."]
    idx = bench_presets.index(current_bs) if current_bs in bench_presets else len(bench_presets)
    bench_choice = st.selectbox("基准标的（下拉预置或自定义）", options, index=idx, key="ts_bs")
    if bench_choice == "自定义...":
        s["benchmark_symbol"] = st.text_input("自定义基准 symbol", value=current_bs,
                                              key="ts_bs_custom")
    else:
        s["benchmark_symbol"] = bench_choice

    # 风控 0 → None（关闭）
    for f in ["max_drawdown_halt", "stop_loss_pct", "take_profit_pct", "daily_loss_limit_pct"]:
        s[f] = s[f] if s.get(f) else None
    s["max_positions"] = s["max_positions"] if s.get("max_positions") else None

    if st.button(" 保存交易设置", type="primary"):
        api.settings_trading_put({k: v for k, v in s.items()})
        st.success("已保存（回测/模拟盘默认使用）")
