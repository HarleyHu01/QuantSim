"""回测页：选标的/周期/策略/引擎 → 运行 → 指标卡片 + 图表 + 明细。"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from web.streamlit.components import (ENGINE_LABELS, MODE_LABELS, TIMEFRAMES,
                                      comparison_option, drawdown_option, echarts,
                                      inject_theme, kline_option, metric_label,
                                      metrics_row, monthly_heatmap_option,
                                      strategy_label, tf_label, to_dataframe)


def _default_range() -> tuple[str, str]:
    """默认回测区间：过去一年（今天往前推 365 天 → 今天）。"""
    today = date.today()
    return (today - timedelta(days=365)).isoformat(), today.isoformat()


def render(api) -> None:
    inject_theme()
    st.header("回测")
    status = api.data_status()
    items = [m for m in status.get("symbols", []) if m.get("rows")]
    symbols = list(dict.fromkeys(m["symbol"] for m in items))
    # 已覆盖的 (symbol, timeframe) 集合：判断所选周期数据是否已存在（避免"有日线但无 5m 不自动拉取"）
    covered = {(m["symbol"], m["timeframe"]) for m in items}
    strategies = api.strategies()
    strategy_keys = [s["key"] for s in strategies]
    strategy_labels = {s["key"]: strategy_label(s) for s in strategies}

    c1, c2 = st.columns(2)
    sel_symbols = c1.multiselect("标的（从已有数据选择）", symbols, default=symbols[:1], key="bt_sym")
    manual = st.text_input("或手动输入 symbol（逗号分隔，将自动拉取数据）：如 600519.SH,000001.SZ,AAPL",
                           "", key="bt_manual")
    tf_idx = TIMEFRAMES.index("1d")
    timeframe = c2.selectbox("周期", TIMEFRAMES, index=tf_idx,
                              format_func=tf_label, key="bt_tf")

    engine = st.radio("回测引擎", list(ENGINE_LABELS.keys()), horizontal=True,
                      format_func=lambda k: f"{ENGINE_LABELS[k]}", index=0, key="bt_engine")
    try:
        bench_sym = api.settings_trading().get("benchmark_symbol", "SPY")
        st.caption(f"基准：{bench_sym}（缺失时自动拉取，用于对比策略是否跑赢大盘）")
    except Exception:  # noqa: BLE001
        pass
    sel_keys = st.multiselect("策略（可多选=组合）", strategy_keys,
                              format_func=lambda k: strategy_labels.get(k, k),
                              default=strategy_keys[:1] if strategy_keys else [],
                              key="bt_st")
    d_start, d_end = _default_range()
    c3, c4 = st.columns(2)
    start = c3.text_input("开始日期（YYYY-MM-DD）", d_start, key="bt_start")
    end = c4.text_input("结束日期（YYYY-MM-DD）", d_end, key="bt_end")
    st.caption(f"默认区间为**过去一年**（{d_start} ~ {d_end}）；"
               "清空两框则使用全部历史数据。")

    mode = st.selectbox("组合模式", list(MODE_LABELS.keys()),
                        format_func=lambda k: f"{MODE_LABELS[k]}（{k}）",
                        key="bt_mode")
    weights: dict[str, int] = {}
    if len(sel_keys) > 1:
        with st.expander("组合权重（步长 5%，留 0 = 等权）"):
            for k in sel_keys:
                weights[k] = st.slider(k, -100, 100, 0, 5, key=f"btw_{k}")

    if st.button("运行回测", type="primary", width='stretch', key="bt_run"):
        # 合并手动输入
        if manual.strip():
            manual_syms = [s.strip().upper() for s in manual.split(",") if s.strip()]
            sel_symbols = list(dict.fromkeys(sel_symbols + manual_syms))
        if not sel_symbols or not sel_keys:
            st.warning("请选择标的和策略")
        else:
            # 缺失数据自动拉取：按 (symbol, timeframe) 判断（不只看 symbol——
            # 有日线但缺 5m 时也要自动拉取该周期数据）
            missing = [s for s in sel_symbols if (s, timeframe) not in covered]
            if missing:
                with st.spinner(f"正在拉取 {missing} 的 {timeframe} 数据..."):
                    api.data_update(symbols=missing, timeframes=[timeframe])
                st.caption(f"已尝试拉取 {missing} {timeframe}（失败标的会在结果中提示）")
            body = {"symbols": sel_symbols, "timeframe": timeframe,
                    "strategy_ids": sel_keys, "engine": engine}
            if start.strip():
                body["start"] = start.strip()
            if end.strip():
                body["end"] = end.strip()
            if any(weights.values()):
                body["portfolio"] = {"mode": mode, "weights": {k: float(v) for k, v in weights.items()}}
            elif mode != "signal_merge":
                body["portfolio"] = {"mode": mode, "weights": {}}
            with st.spinner("回测运行中..."):
                try:
                    r = api.backtest_combo(body)
                except Exception as e:  # noqa: BLE001
                    st.error(f"回测失败: {e}")
                    st.stop()
            st.session_state["bt_result"] = r

    r = st.session_state.get("bt_result")
    if r is None:
        st.info("选择参数后点击「运行回测」")
        return

    st.divider()
    st.markdown(f"#### 回测结果 `#{r.get('run_id')}` · {r['engine']} · {','.join(r['symbols'])} {tf_label(r['timeframe'])} "
                f"（{r.get('start')} ~ {r.get('end')}）")
    metrics_row(r["metrics"])

    try:
        eq = api.backtest_equity(r["run_id"])
        if eq:
            # 图1：归一化收益对比（策略/买入持有/基准指数/标的股价）；图2：回撤对比
            c1, c2 = st.columns(2)
            with c1:
                echarts(comparison_option(eq), 320)
            with c2:
                echarts(drawdown_option(eq), 320)
            # 图4：月度收益热力图（可选增强）
            heat = monthly_heatmap_option(eq)
            if heat:
                echarts(heat, 300)
    except Exception:  # noqa: BLE001
        st.warning("权益曲线获取失败")

    sym = sel_symbols[0] if sel_symbols else None
    if sym:
        try:
            bars = api.data_bars(sym, timeframe, start if start.strip() else None,
                                 end if end.strip() else None)["data"]
            trades = api.backtest_trades(r["run_id"])
            if bars:
                echarts(kline_option(bars, trades, sym), 420)
        except Exception:  # noqa: BLE001
            st.warning("K 线图获取失败")

    st.markdown("##### 各策略独立指标（对比）")
    pm = pd.DataFrame(r.get("per_strategy_metrics", {})).T
    if not pm.empty:
        show = [c for c in ["total_return", "max_drawdown", "sharpe", "win_rate",
                            "profit_factor", "trade_count"] if c in pm.columns]
        st.dataframe(pm[show].rename(columns=metric_label), width='stretch')

    st.markdown("##### 交易明细")
    try:
        trades_df = to_dataframe(api.backtest_trades(r["run_id"]))
        if not trades_df.empty:
            st.dataframe(trades_df, width='stretch', height=260)
    except Exception:  # noqa: BLE001
        pass

    if r.get("warnings"):
        with st.expander(f" 回测警告（{len(r['warnings'])} 条）"):
            st.write(r["warnings"])
