"""策略创建页：手动编写声明式规则策略（指标白名单 + 条件组 + 仓位 + 实时 JSON 预览）。

- 与未来 LLM 生成共用同一套 schema / 校验链（后端 POST /api/strategies/rule）
- 保存为草稿（enabled=0）→ 可立即回测 → 回测通过后在策略管理页激活
- 从策略管理页可跳转本页编辑已保存的规则策略（预填）
"""
from __future__ import annotations

import streamlit as st

from web.streamlit.components import (TIMEFRAMES, metrics_row, tf_label)

IND_LABELS = {
    "sma": "简单移动平均线 (SMA)", "ema": "指数移动平均线 (EMA)", "macd": "MACD",
    "rsi": "相对强弱指标 (RSI)", "boll": "布林带 (BOLL)", "atr": "平均真实波幅 (ATR)",
    "roc": "变动率 (ROC)", "zscore": "Z-Score", "donchian": "唐奇安通道 (Donchian)",
    "volume_sma": "成交量均线 (Volume SMA)", "vwap": "成交量加权均价 (VWAP)",
}
OP_LABELS = {
    ">": "大于 (>)", "<": "小于 (<)", ">=": "大于等于 (>=)", "<=": "小于等于 (<=)",
    "==": "等于 (==)", "between": "介于 (between)", "not_between": "不介于 (not_between)",
    "crosses_above": "上穿 (crosses_above)", "crosses_below": "下穿 (crosses_below)",
}
SRC_LABELS = {
    "close": "收盘价 (close)", "high": "最高价 (high)", "low": "最低价 (low)",
    "volume": "成交量 (volume)", "position": "当前持仓 (position)",
    "indicator": "指标 (indicator)",
}
POS_LABELS = {
    "fixed_pct": "固定比例仓位", "fixed_amount": "固定金额仓位",
    "atr_pct": "ATR 动态仓位", "strength_mapped": "信号强度映射",
}


def _row_defaults() -> dict:
    return {"src": "close", "ind": "sma", "op": ">", "rhs_type": "value",
            "rhs_value": 0.0, "rhs_ref": "close", "lo": 0.0, "hi": 1.0,
            "ind_params": {"window": 20}}


def _init_rows(group: str, edit_rows: list | None) -> list:
    if f"rc_{group}" not in st.session_state:
        st.session_state[f"rc_{group}"] = edit_rows if edit_rows is not None else []
    return st.session_state[f"rc_{group}"]


def _compute_as_map(groups: dict[str, list]) -> dict[tuple[str, int], str]:
    """全局指标输出命名：{指标}_{序号}；编辑加载时保留原 as 名。"""
    counters: dict[str, int] = {}
    out: dict[tuple[str, int], str] = {}
    for gname, rows in groups.items():
        for i, r in enumerate(rows):
            if r.get("src") == "indicator":
                if r.get("_as_override"):
                    out[(gname, i)] = r["_as_override"]
                else:
                    k = r.get("ind", "sma")
                    counters[k] = counters.get(k, 0) + 1
                    out[(gname, i)] = f"{k}_{counters[k]}"
    return out


def _all_refs(meta, groups, as_map) -> tuple[list[str], dict[str, str]]:
    """引用选项 = 全部指标输出（含本表单定义的）+ 内置价格/持仓字段。"""
    inds = {i["key"]: i for i in meta["indicators"]}
    refs = ["close", "high", "low", "volume", "position"]
    labels = {k: SRC_LABELS[k] for k in refs}
    for (gname, i), asn in as_map.items():
        r = groups[gname][i]
        spec = inds[r["ind"]]
        for tpl in spec["outputs"]:
            name = tpl.format(**{"as": asn})
            refs.append(name)
            labels[name] = name
    return refs, labels


def _condition_editor(meta, rows: list, title: str, gname: str,
                      as_map, refs: list, ref_labels: dict) -> None:
    """渲染一个条件组编辑器（行可增删）。"""
    st.markdown(f"##### {title}")
    inds = {i["key"]: i for i in meta["indicators"]}
    src_options = list(SRC_LABELS.keys())
    logic = st.radio("条件组合（内部）", ["全部满足 (all)", "任一满足 (any)"],
                     horizontal=True, key=f"rc_{gname}_logic")
    rows_meta = st.session_state.get(f"rc_{gname}_logic_val", logic)

    deletes: list[int] = []
    for i, r in enumerate(rows):
        c = st.columns([2.1, 1.7, 1.3, 1.9, 0.5])
        src = c[0].selectbox(
            f"数据源 {i + 1}", src_options,
            format_func=lambda s: SRC_LABELS[s],
            index=src_options.index(r.get("src", "close"))
            if r.get("src") in src_options else 0,
            key=f"rc_{gname}_{i}_src")
        r["src"] = src
        if src == "indicator":
            keys = list(inds.keys())
            r["ind"] = c[1].selectbox(
                "指标", keys,
                format_func=lambda k: inds[k]["label"],
                index=keys.index(r.get("ind", "sma")) if r.get("ind") in keys else 0,
                key=f"rc_{gname}_{i}_ind")
            params = inds[r["ind"]]["params"]
            if params:
                c[2].caption("参数")
                cols = st.columns(len(params))
                for pi, p in enumerate(params):
                    cur = r.get("ind_params", {}).get(p["name"], p["default"])
                    if p["type"] == "integer":
                        cur = cols[pi].number_input(p["name"], int(p["min"]), int(p["max"]),
                                                    int(cur), key=f"rc_{gname}_{i}_p{p['name']}")
                    else:
                        cur = cols[pi].number_input(p["name"], float(p["min"]), float(p["max"]),
                                                    float(cur), key=f"rc_{gname}_{i}_p{p['name']}")
                    r.setdefault("ind_params", {})[p["name"]] = cur
            else:
                c[2].caption("—")
        else:
            c[1].caption("—")
            c[2].caption("—")

        op_keys = list(OP_LABELS.keys())
        r["op"] = c[3].selectbox(
            "操作符", op_keys, format_func=lambda k: OP_LABELS[k],
            index=op_keys.index(r.get("op", ">")) if r.get("op") in op_keys else 0,
            key=f"rc_{gname}_{i}_op")
        op = r["op"]

        c4 = st.columns(2)
        if op in ("between", "not_between"):
            r["lo"] = c4[0].number_input("下限", value=float(r.get("lo", 0.0)),
                                         key=f"rc_{gname}_{i}_lo")
            r["hi"] = c4[1].number_input("上限", value=float(r.get("hi", 1.0)),
                                         key=f"rc_{gname}_{i}_hi")
        elif op in ("crosses_above", "crosses_below"):
            r["rhs_type"] = "ref"
            r["rhs_ref"] = c4[0].selectbox(
                "参照（指标/字段）", refs, format_func=lambda k: ref_labels.get(k, k),
                index=refs.index(r.get("rhs_ref", "close")) if r.get("rhs_ref") in refs else 0,
                key=f"rc_{gname}_{i}_rhsref")
        else:
            r["rhs_type"] = c4[0].radio("右值", ["数值 (value)", "引用 (ref)"],
                                        horizontal=True, key=f"rc_{gname}_{i}_rt")
            if r["rhs_type"] == "ref":
                r["rhs_ref"] = c4[1].selectbox(
                    "引用", refs, format_func=lambda k: ref_labels.get(k, k),
                    index=refs.index(r.get("rhs_ref", "close")) if r.get("rhs_ref") in refs else 0,
                    key=f"rc_{gname}_{i}_rhsref2")
            else:
                r["rhs_value"] = c4[1].number_input("数值", value=float(r.get("rhs_value", 0.0)),
                                                    key=f"rc_{gname}_{i}_rhsv")

        if c[4].button("删除", key=f"rc_{gname}_{i}_del"):
            deletes.append(i)
    for i in sorted(deletes, reverse=True):
        rows.pop(i)
    if deletes:
        st.rerun()


def _build_config(meta, groups: dict[str, list], as_map, name_zh, name_en, desc,
                  allow_short, only_long, max_hold, pos_mode, pos_value,
                  risk, stop_mult, max_pos) -> dict:
    inds = {i["key"]: i for i in meta["indicators"]}
    indicators: list[dict] = []
    seen: set[str] = set()
    for gname, rows in groups.items():
        for i, r in enumerate(rows):
            if r.get("src") != "indicator":
                continue
            asn = as_map[(gname, i)]
            if asn in seen:
                continue
            seen.add(asn)
            indicators.append({"name": r["ind"], "params": dict(r.get("ind_params") or {}),
                               "as": asn})

    def _group(rows: list, gname: str, logic: str) -> dict | None:
        conds = []
        for i, r in enumerate(rows):
            lhs = as_map[(gname, i)] if r.get("src") == "indicator" else r["src"]
            op = r["op"]
            if op in ("between", "not_between"):
                rhs = [float(r.get("lo", 0.0)), float(r.get("hi", 1.0))]
            elif op in ("crosses_above", "crosses_below"):
                rhs = r.get("rhs_ref", "close")
            elif r.get("rhs_type") == "ref":
                rhs = r.get("rhs_ref", "close")
            else:
                rhs = float(r.get("rhs_value", 0.0))
            conds.append({"lhs": lhs, "op": op, "rhs": rhs})
        if not conds:
            return None
        return {"logic": logic, "conditions": conds}

    logics = {k: st.session_state.get(f"rc_{k}_logic", "全部满足 (all)") for k in groups}
    return {
        "name": name_zh.strip() or name_en.strip() or "未命名策略",
        "description": desc,
        "indicators": indicators,
        "entry_long": _group(groups["buy"], "buy", "all" if "all" in logics["buy"] else "any"),
        "exit_long": _group(groups["sell"], "sell", "all" if "all" in logics["sell"] else "any"),
        "entry_short": _group(groups["short"], "short", "all" if "all" in logics.get("short", "all") else "any")
        if allow_short and groups["short"] else None,
        "exit_short": _group(groups["close"], "close", "all" if "all" in logics.get("close", "all") else "any")
        if allow_short and groups["close"] else None,
        "position": {"mode": pos_mode, "value": pos_value,
                     "risk_per_trade": risk, "stop_atr_mult": stop_mult,
                     "max_position": max_pos},
        "params": {"allow_short": allow_short, "only_long": only_long,
                   "max_hold_bars": max_hold or None},
    }


def _edit_rows(meta, edit_cfg: dict, group_name: str) -> list:
    """把已保存配置转成表单行（保留原指标输出名以便 rhs 引用匹配）。"""
    g = edit_cfg.get(group_name)
    if not g or not g.get("conditions"):
        return []
    inds = {i["key"]: i for i in meta["indicators"]}
    rows = []
    for cond in g["conditions"]:
        r = _row_defaults()
        r["op"] = cond["op"]
        lhs = cond["lhs"]
        matched = None
        for spec in edit_cfg.get("indicators", []):
            m = inds.get(spec["name"])
            if not m:
                continue
            as_name = spec.get("as") or spec.get("as_", "")  # 兼容 as / as_ 两种存储键
            outs = [t.format(**{"as": as_name}) for t in m["outputs"]]
            if lhs in outs:
                matched = spec
                break
        if matched is not None:
            r["src"] = "indicator"
            r["ind"] = matched["name"]
            r["ind_params"] = dict(matched.get("params") or {})
            r["_as_override"] = matched.get("as") or matched.get("as_", "")
        elif lhs in SRC_LABELS:
            r["src"] = lhs
        else:
            r["src"] = "close"  # 罕见：引用未匹配 → 兜底
        rhs = cond["rhs"]
        if cond["op"] in ("between", "not_between"):
            r["lo"], r["hi"] = float(rhs[0]), float(rhs[1])
        elif isinstance(rhs, str):
            r["rhs_type"] = "ref"
            r["rhs_ref"] = rhs
        else:
            r["rhs_type"] = "value"
            r["rhs_value"] = float(rhs)
        rows.append(r)
    return rows


def render(api) -> None:
    st.header("策略创建")
    st.caption("手动编写条件规则策略：选择指标 → 设置买卖条件 → 配置仓位 → 保存并回测")

    meta = api.indicators_meta()

    if "rule_edit" in st.session_state:
        st.session_state["rc_edit_cfg"] = st.session_state.pop("rule_edit")
    edit_cfg = st.session_state.get("rc_edit_cfg")
    edit_key = (edit_cfg or {}).get("key") if isinstance(edit_cfg, dict) else None
    edit_cfg = (edit_cfg or {}).get("config") if isinstance(edit_cfg, dict) else None

    groups = {
        "buy": _init_rows("buy", _edit_rows(meta, edit_cfg, "entry_long") if edit_cfg else None),
        "sell": _init_rows("sell", _edit_rows(meta, edit_cfg, "exit_long") if edit_cfg else None),
        "short": _init_rows("short", _edit_rows(meta, edit_cfg, "entry_short") if edit_cfg else None),
        "close": _init_rows("close", _edit_rows(meta, edit_cfg, "exit_short") if edit_cfg else None),
    }
    if edit_cfg and st.button("新建空白策略（清空编辑内容）", key="rc_new"):
        for k in ("rc_buy", "rc_sell", "rc_short", "rc_close", "rc_edit_cfg"):
            st.session_state.pop(k, None)
        st.rerun()

    as_map = _compute_as_map(groups)
    refs, ref_labels = _all_refs(meta, groups, as_map)

    c1, c2 = st.columns(2)
    name_zh = c1.text_input("策略名称（中文）", value=edit_cfg.get("name", "") if edit_cfg else "",
                            key="rc_name")
    name_en = c2.text_input("策略名称（英文，用于生成标识）", key="rc_name_en")
    desc = st.text_area("策略描述", value=edit_cfg.get("description", "") if edit_cfg else "",
                        key="rc_desc")

    _condition_editor(meta, groups["buy"], "买入条件", "buy", as_map, refs, ref_labels)
    if st.button("＋ 添加买入条件", key="rc_add_buy"):
        groups["buy"].append(_row_defaults())
        st.rerun()

    _condition_editor(meta, groups["sell"], "卖出条件", "sell", as_map, refs, ref_labels)
    if st.button("＋ 添加卖出条件", key="rc_add_sell"):
        groups["sell"].append(_row_defaults())
        st.rerun()

    st.divider()
    allow_short = st.checkbox("允许做空（显示做空/平空条件）", value=False, key="rc_short_allow")
    only_long = st.checkbox("仅做多（不开空）", value=True, key="rc_only_long")
    if allow_short:
        _condition_editor(meta, groups["short"], "做空条件", "short", as_map, refs, ref_labels)
        if st.button("＋ 添加做空条件", key="rc_add_short"):
            groups["short"].append(_row_defaults())
            st.rerun()
        _condition_editor(meta, groups["close"], "平空条件", "close", as_map, refs, ref_labels)
        if st.button("＋ 添加平空条件", key="rc_add_close"):
            groups["close"].append(_row_defaults())
            st.rerun()
    else:
        groups["short"].clear()
        groups["close"].clear()

    st.divider()
    st.markdown("##### 仓位设置")
    ep = (edit_cfg or {}).get("position", {}) or {}
    pc = st.columns(3)
    pos_mode = pc[0].selectbox("仓位模式", list(POS_LABELS.keys()),
                               format_func=lambda k: POS_LABELS[k],
                               index=0, key="rc_pos_mode")
    if pos_mode in ("fixed_pct", "strength_mapped"):
        dv = float(ep.get("value", 0.9 if pos_mode == "fixed_pct" else 0.5))
        pos_value = pc[1].slider("仓位比例", 0.05, 1.5, dv, 0.05, key="rc_pos_v")
    elif pos_mode == "fixed_amount":
        pos_value = pc[1].number_input("每笔金额（元）", value=float(ep.get("value", 10_000)),
                                       step=1000.0, key="rc_pos_v")
    else:  # atr_pct
        pos_value = 0.0
        pc[1].slider("单笔风险比例", 0.001, 0.05, float(ep.get("risk_per_trade", 0.01)),
                     0.001, key="rc_pos_risk")
        pc[2].slider("ATR 止损倍数", 0.5, 5.0, float(ep.get("stop_atr_mult", 2.0)),
                     0.1, key="rc_pos_stop")
    risk = st.session_state.get("rc_pos_risk", 0.01)
    stop = st.session_state.get("rc_pos_stop", 2.0)
    max_pos = pc[2].slider("最大仓位上限", 0.1, 2.0, float(ep.get("max_position", 1.0)),
                           0.05, key="rc_pos_max")

    st.divider()
    st.markdown("##### 策略参数")
    eparams = (edit_cfg or {}).get("params", {}) or {}
    max_hold = st.number_input("最大持仓时间（K线根数，0 = 不限）",
                               value=int(eparams.get("max_hold_bars") or 0), min_value=0,
                               max_value=10000, step=1, key="rc_max_hold")

    config = _build_config(meta, groups, as_map, name_zh, name_en, desc,
                           allow_short, only_long, max_hold, pos_mode, pos_value,
                           risk, stop, max_pos)

    st.divider()
    with st.expander("实时预览（声明式 JSON 配置）", expanded=True):
        st.json(config)

    st.divider()
    bt = st.columns(2)
    save_draft = bt[0].button("保存策略（草稿）", type="secondary", use_container_width=True,
                              key="rc_save")
    save_bt = bt[1].button("保存并回测", type="primary", use_container_width=True,
                           key="rc_save_bt")

    # 回测选择（保存并回测时使用）
    status = api.data_status()
    symbols = [m["symbol"] for m in status.get("symbols", []) if m.get("rows")]
    bc = st.columns([3, 2])
    bt_sym = bc[0].selectbox("回测标的", symbols if symbols else ["（暂无数据）"],
                             key="rc_bt_sym")
    bt_tf = bc[1].selectbox("回测周期", TIMEFRAMES, index=7, format_func=tf_label,
                            key="rc_bt_tf")

    if not (name_zh.strip() or name_en.strip()):
        st.warning("请填写策略名称（中文或英文至少一项）")
        return
    if not (save_draft or save_bt):
        return

    try:
        resp = api.add_rule_strategy(config, key=edit_key, enabled=False)
    except Exception as e:  # noqa: BLE001
        st.error(f"保存失败：{e}")
        return
    if not resp.get("strategy"):
        st.error("保存失败：" + str(resp))
        return
    sid = resp["strategy"]["id"]
    st.success(f"已保存为草稿：{config['name']}（#{sid}，key={resp['strategy']['key']}）"
               " — 校验通过，可在「策略管理」页启用")

    if save_bt and symbols:
        if bt_sym == "（暂无数据）":
            st.info("尚无本地行情，请先在「数据管理」页添加标的")
            return
        with st.spinner("回测运行中..."):
            r = api.backtest_combo({"symbols": [bt_sym], "timeframe": bt_tf,
                                    "strategy_ids": [resp["strategy"]["key"]],
                                    "engine": "event_driven"})
        st.markdown(f"#### 回测结果（{bt_sym} / {tf_label(bt_tf)}）")
        metrics_row(r.get("metrics", {}))
        st.markdown(
            f"交易 {r['metrics'].get('trade_count', 0)} 笔，"
            f"总收益率 {r['metrics'].get('total_return', 0):.2%}，"
            f"最大回撤 {r['metrics'].get('max_drawdown', 0):.2%}，"
            f"夏普 {r['metrics'].get('sharpe', 0):.2f}")
        if r.get("metrics", {}).get("trade_count", 0) > 0:
            if st.button("回测通过，激活启用该策略", type="primary", key="rc_activate"):
                api.strategy_update(sid, enabled=True)
                st.success("已启用！可在「策略管理 / 组合配置 / 回测」中使用")
                st.rerun()
        else:
            st.warning("回测无交易：请检查买卖条件是否合理（可能条件过严或方向反了）")
