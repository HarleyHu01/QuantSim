""""模拟盘页：账户状态卡片、start/stop、子账户资金池、最新成交（st.empty 轮询刷新）。

分钟级账户会额外展示：
- 交易时段状态（交易中 / 午间休市 / 已收盘 / 周末休市）与下次开盘倒计时
- 数据新鲜度（本地最后一根 bar 距现在多久）—— 这就是「延迟」的真实来源
- 最近一次 tick 是否被门控跳过（非交易时段不空转拉数）
"""
from __future__ import annotations

import time

import streamlit as st

from web.streamlit.components import (MODE_LABELS, PAPER_STATUS_LABELS, TIMEFRAMES,
                                      TF_COVERAGE_HINT, inject_theme,
                                      strategy_label, tf_label, to_dataframe)


def _fmt_secs(secs: float | None) -> str:
    if secs is None:
        return "—"
    secs = int(secs)
    if secs < 60:
        return f"{secs} 秒"
    if secs < 3600:
        return f"{secs // 60} 分 {secs % 60} 秒"
    return f"{secs // 3600} 小时 {(secs % 3600) // 60} 分"


def _render_realtime(rt: dict) -> None:
    """渲染分钟级账户的时段与延迟面板。"""
    if not rt or not rt.get("is_minute"):
        return
    syms: dict = rt.get("symbols") or {}
    any_open = rt.get("market_open")
    if not any_open:
        # 找一个最近要开盘的标的，给用户一个明确等待时间
        pending = [(s, v) for s, v in syms.items() if v.get("seconds_to_next_open")]
        if pending:
            s, v = min(pending, key=lambda kv: kv[1]["seconds_to_next_open"])
            st.info(
                f"当前非交易时段（{v.get('session')}，{s}）——"
                f"距下次开盘约 {_fmt_secs(v['seconds_to_next_open'])}。"
                "此期间不会拉取数据，也不会产生成交。"
            )
    rows = []
    for sym, v in syms.items():
        lag = v.get("bar_lag_seconds")
        rows.append({
            "标的": sym,
            "市场": "A股" if v.get("market") == "cn" else "美股",
            "时段": v.get("session"),
            "最新K线延迟": _fmt_secs(lag) if lag is not None else "无数据",
            "最近tick": (v.get("last_tick_at") or "—")[11:19],
            "本tick": v.get("skipped") or "已处理",
        })
    if rows:
        st.dataframe(to_dataframe(rows), width="stretch", height=140, hide_index=True)
        st.caption("「最新K线延迟」= 本地最后一根 K 线距现在的时间。"
                   f"当前周期 {rt.get('timeframe')}，一根 K 线内的延迟属正常。")


def render(api) -> None:
    inject_theme()
    st.header("模拟盘")
    try:
        accounts = api.paper_status()["accounts"]
    except Exception:  # noqa: BLE001
        accounts = []
    acc_labels = {a["id"]: f"#{a['id']} {a['name']}（"
                  f"{PAPER_STATUS_LABELS.get(a['status'], a['status'])}）" for a in accounts}
    acc_ids = [a["id"] for a in accounts]

    sel = st.selectbox("账户", acc_ids, format_func=lambda i: acc_labels.get(i, str(i)),
                       key="pp_acc") if acc_ids else None

    # ---- 新建/启动 ----
    with st.expander(" 新建账户 / 启动模拟盘"):
        c1, c2 = st.columns(2)
        name = c1.text_input("账户名", "模拟盘", key="pp_name")
        capital = c2.number_input("初始资金", 1_000.0, 1e12, 1_000_000.0, step=10_000.0,
                                  format="%.0f", key="pp_cap")
        symbols = st.multiselect("标的", [s["symbol"] for s in api.data_symbols()], key="pp_sym")
        strategies = api.strategies()
        sel_st = st.multiselect("策略", [s["key"] for s in strategies],
                                format_func=lambda k: strategy_label(next(
                                    (s for s in strategies if s["key"] == k), {"key": k})), key="pp_st")
        tf = st.selectbox("周期", TIMEFRAMES, index=TIMEFRAMES.index("1d"),
                          format_func=tf_label, key="pp_tf")
        if TF_COVERAGE_HINT.get(tf):
            st.caption(f"ⓘ {TF_COVERAGE_HINT[tf]}")
        is_minute = tf in ("1m", "5m", "15m", "30m", "1h")
        if is_minute:
            st.caption("分钟级账户在非交易时段自动暂停拉取，恢复交易后继续，无需手动重启。")
        mode = st.selectbox("组合模式", list(MODE_LABELS.keys()),
                            format_func=lambda k: f"{MODE_LABELS[k]}（{k}）", key="pp_mode")
        weights: dict[str, int] = {}
        if len(sel_st) > 1:
            for k in sel_st:
                weights[k] = st.slider(f"权重 {k}", -100, 100, 0, 5, key=f"ppw_{k}")
        poll = st.number_input(
            "轮询间隔（秒，0 = 按周期自动）", 0, 3600, 0, key="pp_poll",
            help="分钟级默认按 K 线边界对齐触发（5m 周期即每 5 分钟一次），"
                 "填 0 用自动值；日线/周线/月线按收盘后 cron 触发，此值不生效。")
        if st.button(" 启动（新账户）", type="primary"):
            if not symbols or not sel_st:
                st.warning("请选择标的和策略")
            else:
                body = {"name": name, "initial_capital": float(capital), "symbols": symbols,
                        "timeframe": tf, "strategy_ids": sel_st}
                if poll:
                    body["poll_seconds"] = int(poll)
                if any(weights.values()):
                    body["portfolio"] = {"mode": mode, "weights": {k: float(v) for k, v in weights.items()}}
                elif mode != "signal_merge":
                    body["portfolio"] = {"mode": mode, "weights": {}}
                r = api.paper_start(body)
                st.success(f"已启动账户 #{r['account']['id']}")
                st.rerun()

    # ---- 删除账户 ----
    with st.expander("删除账户"):
        if accounts:
            del_id = st.selectbox("选择账户（运行中账户需先停止）", acc_ids,
                                  format_func=lambda i: acc_labels.get(i, str(i)), key="pp_del")
            del_acct = next((a for a in accounts if a["id"] == del_id), None)
            running = bool(del_acct and del_acct["status"] == "running")
            if running:
                st.caption("该账户运行中，需先停止才能删除")
            cleanup = st.checkbox("同时清理持仓/成交/净值记录", value=True, key="pp_del_cleanup")
            if st.button("删除账户", disabled=running, key="pp_del_btn"):
                api.delete_account(del_id, cleanup=cleanup)
                st.success(f"已删除账户 #{del_id}")
                st.rerun()
        else:
            st.caption("暂无账户")

    if sel is None:
        st.info("暂无模拟盘账户，请在上方「新建账户 / 启动模拟盘」创建")
        return

    c1, c2 = st.columns([1, 5])
    if c1.button(" 停止", width='stretch'):
        api.paper_stop(sel)
        st.rerun()
    with c2:
        st.caption("停止后持仓/成交保留；重启服务时 running 账户自动恢复")

    auto = st.toggle(" 自动刷新", value=False, key="pp_auto")
    interval = st.slider("刷新间隔（秒）", 1, 60, 5, key="pp_iv")
    placeholder = st.empty()

    def status_panel() -> None:
        try:
            st_ = api.paper_status(sel)
        except Exception as e:  # noqa: BLE001
            st.error(f"获取状态失败: {e}")
            return
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("状态", PAPER_STATUS_LABELS.get(st_.get("status", ""), st_.get("status", "?")))
        m2.metric("总权益", f"{st_.get('equity', 0):,.0f}")
        m3.metric("现金", f"{st_.get('cash', 0):,.0f}")
        m4.metric("持仓数", len(st_.get("positions", [])))
        if st_.get("mode") == "capital_split":
            subs = st_.get("sub_brokers", [])
            st.caption("子账户资金池：" + "、".join(f"{v:,.0f}" for v in subs))
        if st_.get("pending_orders"):
            st.warning(f"挂单 {st_['pending_orders']} 笔（下一 bar 开盘成交）")
        if st_.get("realtime"):
            _render_realtime(st_["realtime"])
        pos = to_dataframe(st_.get("positions", []))
        if not pos.empty:
            st.markdown("##### 持仓")
            st.dataframe(pos, width='stretch', height=160)
        trades = to_dataframe(st_.get("trades_tail", []))
        if not trades.empty:
            st.markdown("##### 最新成交")
            st.dataframe(trades, width='stretch', height=200)

    if auto:
        while True:
            with placeholder.container():
                status_panel()
            time.sleep(interval)
            st.rerun()
    else:
        with placeholder.container():
            status_panel()
