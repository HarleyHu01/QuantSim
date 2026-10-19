"""策略创建页：手动编写声明式规则策略（指标白名单 + 条件组 + 仓位 + 实时 JSON 预览）。

- 支持 **AI 辅助**：用自然语言让大模型生成/改写配置，结果直接灌进表单，
  用户可以继续逐项手调（两种产物共用同一套 schema 与校验链）
- 保存为草稿（enabled=0）→ 可立即回测 → 回测通过后在策略管理页激活
- 从策略管理页可跳转本页编辑已保存的规则策略（预填）
"""
from __future__ import annotations

import json

import streamlit as st

from web.streamlit.components import (TIMEFRAMES, inject_theme, metrics_row,
                                      tf_label)

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

#: AI 辅助的示例需求（一条一个按钮，点一下即填入输入框）
AI_EXAMPLES = [
    "5 日均线上穿 20 日均线买入，下穿卖出，满仓",
    "RSI 低于 30 买入，高于 70 卖出，仓位 80%",
    "收盘价突破 20 日最高价买入，跌破 10 日最低价卖出",
]


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


def _extra_indicators(groups: dict[str, list], edit_cfg: dict | None) -> list[dict]:
    """配置里声明了、但没有任何条件行把它当左值的指标。

    表单是「行驱动」的——指标只能通过某条条件的左值引入。可 AI（或手写的
    JSON）常见 `crosses_above ma20` 这种只把指标当右值的写法，若不单独保留，
    它就会在 `_build_config` 里被丢掉、引用下拉框也回落到 close，导致
    「载入后条件悄悄变了」。这里把这类指标提取出来单独声明。
    """
    specs = list((edit_cfg or {}).get("indicators") or [])
    if not specs:
        return []
    used: set[str] = set()
    for rows in groups.values():
        for r in rows:
            if r.get("src") == "indicator":
                name = r.get("_as_override")
                if name:
                    used.add(name)
    out = []
    for spec in specs:
        asn = spec.get("as") or spec.get("as_", "")
        if asn and asn not in used:
            out.append(spec)
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
    for spec in st.session_state.get("rc_extra_inds") or []:
        m = inds.get(spec.get("name"))
        if not m:
            continue
        asn = spec.get("as") or spec.get("as_", "")
        for tpl in m["outputs"]:
            name = tpl.format(**{"as": asn})
            if name not in labels:
                refs.append(name)
                labels[name] = name
    return refs, labels


def _ver() -> int:
    """表单控件 key 的版本号。

    Streamlit 的控件一旦带 key，值就只认 session_state / 浏览器侧的控件状态，
    `value=` 与 `index=` 只在控件**首次出现**时生效。所以「把配置灌进表单」
    不能靠删 session_state 里的键 —— 前端控件仍记得旧值，下一轮又被写回来。
    唯一可靠的办法是换一批全新的 key，让控件被当成新控件重建。
    """
    return int(st.session_state.get("rc_ver", 0))


def _k(suffix: str) -> str:
    """带版本号的小部件 key。"""
    return f"rc_v{_ver()}_{suffix}"


def _condition_editor(meta, rows: list, title: str, gname: str,
                      as_map, refs: list, ref_labels: dict) -> None:
    """渲染一个条件组编辑器（行可增删）。"""
    st.markdown(f"##### {title}")
    inds = {i["key"]: i for i in meta["indicators"]}
    src_options = list(SRC_LABELS.keys())
    logic = st.radio("条件组合（内部）", ["全部满足 (all)", "任一满足 (any)"],
                     horizontal=True, key=_k(f"{gname}_logic"))
    rows_meta = st.session_state.get(f"rc_{gname}_logic_val", logic)

    deletes: list[int] = []
    for i, r in enumerate(rows):
        c = st.columns([2.1, 1.7, 1.3, 1.9, 0.5])
        src = c[0].selectbox(
            f"数据源 {i + 1}", src_options,
            format_func=lambda s: SRC_LABELS[s],
            index=src_options.index(r.get("src", "close"))
            if r.get("src") in src_options else 0,
            key=_k(f"{gname}_{i}_src"))
        r["src"] = src
        if src == "indicator":
            keys = list(inds.keys())
            r["ind"] = c[1].selectbox(
                "指标", keys,
                format_func=lambda k: inds[k]["label"],
                index=keys.index(r.get("ind", "sma")) if r.get("ind") in keys else 0,
                key=_k(f"{gname}_{i}_ind"))
            params = inds[r["ind"]]["params"]
            if params:
                c[2].caption("参数")
                cols = st.columns(len(params))
                for pi, p in enumerate(params):
                    cur = r.get("ind_params", {}).get(p["name"], p["default"])
                    if p["type"] == "integer":
                        cur = cols[pi].number_input(p["name"], int(p["min"]), int(p["max"]),
                                                    int(cur), key=_k(f"{gname}_{i}_p{p['name']}"))
                    else:
                        cur = cols[pi].number_input(p["name"], float(p["min"]), float(p["max"]),
                                                    float(cur), key=_k(f"{gname}_{i}_p{p['name']}"))
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
            key=_k(f"{gname}_{i}_op"))
        op = r["op"]

        c4 = st.columns(2)
        if op in ("between", "not_between"):
            r["lo"] = c4[0].number_input("下限", value=float(r.get("lo", 0.0)),
                                         key=_k(f"{gname}_{i}_lo"))
            r["hi"] = c4[1].number_input("上限", value=float(r.get("hi", 1.0)),
                                         key=_k(f"{gname}_{i}_hi"))
        elif op in ("crosses_above", "crosses_below"):
            r["rhs_type"] = "ref"
            r["rhs_ref"] = c4[0].selectbox(
                "参照（指标/字段）", refs, format_func=lambda k: ref_labels.get(k, k),
                index=refs.index(r.get("rhs_ref", "close")) if r.get("rhs_ref") in refs else 0,
                key=_k(f"{gname}_{i}_rhsref"))
        else:
            r["rhs_type"] = c4[0].radio("右值", ["数值 (value)", "引用 (ref)"],
                                        horizontal=True, key=_k(f"{gname}_{i}_rt"))
            if r["rhs_type"] == "ref":
                r["rhs_ref"] = c4[1].selectbox(
                    "引用", refs, format_func=lambda k: ref_labels.get(k, k),
                    index=refs.index(r.get("rhs_ref", "close")) if r.get("rhs_ref") in refs else 0,
                    key=_k(f"{gname}_{i}_rhsref2"))
            else:
                r["rhs_value"] = c4[1].number_input("数值", value=float(r.get("rhs_value", 0.0)),
                                                    key=_k(f"{gname}_{i}_rhsv"))

        if c[4].button("删除", key=_k(f"{gname}_{i}_del")):
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
    # 只作右值引用的指标（无左侧条件行）也要写进配置，否则会被静默丢掉
    for spec in st.session_state.get("rc_extra_inds") or []:
        asn = spec.get("as") or spec.get("as_", "")
        if not asn or asn in seen or spec.get("name") not in inds:
            continue
        seen.add(asn)
        indicators.append({"name": spec["name"], "params": dict(spec.get("params") or {}),
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

    logics = {k: st.session_state.get(_k(f"{k}_logic"), "全部满足 (all)") for k in groups}
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


def _apply_ai_config(cfg: dict) -> None:
    """把 AI 给的配置灌进表单并重跑。

    复用「编辑已有策略」那条链路（`rc_edit_cfg` → `_edit_rows`），
    而不是另写一套解析：AI 配置和库里存的配置本来就同构，
    走同一条路才能保证「AI 生成的」和「手填的」完全等价。

    注意：Streamlit 的控件一旦带了 key，`value=`/`index=` 就只在控件**首次
    出现**时生效，之后只认会话里的控件状态；前端控件本身也记住值，单纯删
    session_state 的键清不掉（删了本轮又会被写回）。所以这里只登记一个待办，
    真正的动作放到下一轮 `render()` 最开头：把控件 key 的版本号 +1，让所有
    控件以全新 key 重建，值才会真正换成 AI 给的配置。
    """
    st.session_state["rc_ai_apply_cfg"] = cfg
    st.rerun()


def _consume_ai_apply() -> None:
    """在 render() 最开头落地 AI 配置（此时本轮还没有任何控件被实例化）。"""
    cfg = st.session_state.pop("rc_ai_apply_cfg", None)
    if not isinstance(cfg, dict):
        return
    # 换一批全新的控件 key：控件会按新 key 重建，value=/index= 才会真正生效
    st.session_state["rc_ver"] = _ver() + 1
    for k in ("rc_buy", "rc_sell", "rc_short", "rc_close"):
        st.session_state.pop(k, None)
    st.session_state["rc_edit_cfg"] = {"config": cfg, "key": None}


def _ai_message(request: str, current: dict | None, include_current: bool) -> str:
    """拼给大模型的请求：需求 + （可选）当前表单配置。"""
    msg = request.strip()
    if include_current and current:
        msg += ("\n\n=== 当前配置（请在此基础上修改，保留没提到的部分）===\n"
                + json.dumps(current, ensure_ascii=False, indent=2))
    return msg


def _render_ai_panel(api, current_config: dict) -> None:
    """AI 辅助面板：自然语言 → 规则配置 → 灌进下面的表单继续手调。

    设计取舍：AI 只负责「生成一版起点」，落地后用户可以逐项改
    ——把 AI 输出直接写库会让人失去掌控感，而表单里手调的部分
    再交给 AI 改写一轮，才形成「生成 → 改 → 再生成」的闭环。
    """
    with st.expander("AI 辅助（用自然语言生成 / 改写这套条件）", expanded=False):
        try:
            stt = api.llm_status()
        except Exception as e:  # noqa: BLE001
            st.warning(f"无法读取大模型配置：{e}")
            return
        if not stt.get("configured"):
            st.info("尚未配置大模型。请到「系统设置 → 策略设置 → 大模型设置」填写 API Key 后回到此页。")
            if st.button("前往大模型设置", key="rc_ai_goto_settings"):
                st.session_state["nav_target"] = "策略设置"
                st.rerun()
            return

        st.caption(f"当前模型：`{stt.get('model')}` · {stt.get('base_url')}")

        if "rc_ai_req" not in st.session_state:
            st.session_state["rc_ai_req"] = ""
        ex = st.columns(len(AI_EXAMPLES))
        for i, text in enumerate(AI_EXAMPLES):
            if ex[i].button(text[:12] + "…" if len(text) > 12 else text,
                            key=f"rc_ai_ex{i}", help=text):
                st.session_state["rc_ai_req"] = text
                st.rerun()

        req = st.text_area(
            "需求描述",
            key="rc_ai_req",
            height=90,
            placeholder="例如：20 日均线上穿 60 日均线买入，下穿卖出，仓位 80%，持仓不超过 30 根 K 线",
        )
        has_current = bool((current_config or {}).get("entry_long")
                           or (current_config or {}).get("exit_long"))
        include_current = st.checkbox(
            "把当前表单的条件一起发给 AI（让它在此基础上改，而不是从零写）",
            value=has_current, key="rc_ai_use_current",
            help="勾选后会把下方实时预览里的 JSON 一起发给大模型")

        c = st.columns([1, 1, 2])
        gen = c[0].button("生成配置", type="primary", key="rc_ai_gen")
        if c[1].button("清空输入", key="rc_ai_clear"):
            st.session_state["rc_ai_req"] = ""
            st.session_state.pop("rc_ai_result", None)
            st.rerun()

        if gen:
            if not req.strip():
                st.warning("请先写下策略需求")
            else:
                message = _ai_message(req, current_config, include_current)
                with st.spinner("大模型正在生成配置…"):
                    try:
                        resp = api.llm_chat(message, mode="rule")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"生成失败：{_err_text(e)}")
                        return
                st.session_state["rc_ai_result"] = resp

        resp = st.session_state.get("rc_ai_result")
        if not resp:
            return
        result = resp.get("result") or {}
        st.divider()
        if not result.get("ok"):
            errs = result.get("errors") or ["模型未返回可用配置"]
            st.error("AI 生成的配置未通过校验：")
            for e in errs:
                st.markdown(f"- {e}")
            st.caption("可以让它再改一次（补充更明确的条件），或直接看下面的原始配置手动调整。")
            raw = result.get("config")
            if raw:
                with st.expander("原始配置（未通过校验）"):
                    st.json(raw)
            return

        st.success("AI 配置已通过校验。可一键载入下方表单，再逐项手调。")
        cfg = result.get("config") or {}
        st.json(cfg)
        if st.button("载入到下方表单（覆盖当前条件）", type="primary", key="rc_ai_apply"):
            _apply_ai_config(cfg)
    return


def _err_text(e: Exception) -> str:
    """把 API 错误拆成可读文本（后端 detail 可能是 {errors:[...]}）。"""
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            detail = resp.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = None
        if isinstance(detail, dict):
            errs = detail.get("errors")
            if isinstance(errs, list) and errs:
                return "；".join(str(x) for x in errs)
            if detail.get("message"):
                return str(detail["message"])
        if detail:
            return str(detail)
        return f"{resp.status_code} {resp.text[:200]}"
    return f"{type(e).__name__}: {e}"


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
    inject_theme()
    st.header("策略创建")
    st.caption("手动编写条件规则策略：选择指标 → 设置买卖条件 → 配置仓位 → 保存并回测")

    # 必须在任何控件实例化之前：AI 配置靠换一批新 key 才能真正灌进表单
    _consume_ai_apply()

    meta = api.indicators_meta()

    if "rule_edit" in st.session_state:
        st.session_state["rc_edit_cfg"] = st.session_state.pop("rule_edit")
        # 从其它页面跳过来时也换 key，确保预填值不被旧控件状态盖住
        st.session_state["rc_ver"] = _ver() + 1
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
        for k in ("rc_buy", "rc_sell", "rc_short", "rc_close", "rc_edit_cfg",
                  "rc_ai_result", "rc_ai_current", "rc_extra_inds"):
            st.session_state.pop(k, None)
        st.session_state["rc_ver"] = _ver() + 1
        st.rerun()

    # 只作右值引用的指标：保留下来供引用下拉框与配置复用（否则会被静默丢弃）
    st.session_state["rc_extra_inds"] = _extra_indicators(groups, edit_cfg)

    # AI 辅助面板放在表单顶部：它要读取「上一轮算出的配置」，
    # 而配置只有等控件都渲染完才算得出来，所以放进 session_state 跨轮传递。
    _render_ai_panel(api, st.session_state.get("rc_ai_current") or {})

    as_map = _compute_as_map(groups)
    refs, ref_labels = _all_refs(meta, groups, as_map)

    c1, c2 = st.columns(2)
    name_zh = c1.text_input("策略名称（中文）", value=edit_cfg.get("name", "") if edit_cfg else "",
                            key=_k("name"))
    name_en = c2.text_input("策略名称（英文，用于生成标识）", key=_k("name_en"))
    desc = st.text_area("策略描述", value=edit_cfg.get("description", "") if edit_cfg else "",
                        key=_k("desc"))

    _condition_editor(meta, groups["buy"], "买入条件", "buy", as_map, refs, ref_labels)
    if st.button("＋ 添加买入条件", key=_k("add_buy")):
        groups["buy"].append(_row_defaults())
        st.rerun()

    _condition_editor(meta, groups["sell"], "卖出条件", "sell", as_map, refs, ref_labels)
    if st.button("＋ 添加卖出条件", key=_k("add_sell")):
        groups["sell"].append(_row_defaults())
        st.rerun()

    st.divider()
    allow_short = st.checkbox("允许做空（显示做空/平空条件）", value=False, key=_k("short_allow"))
    only_long = st.checkbox("仅做多（不开空）", value=True, key=_k("only_long"))
    if allow_short:
        _condition_editor(meta, groups["short"], "做空条件", "short", as_map, refs, ref_labels)
        if st.button("＋ 添加做空条件", key=_k("add_short")):
            groups["short"].append(_row_defaults())
            st.rerun()
        _condition_editor(meta, groups["close"], "平空条件", "close", as_map, refs, ref_labels)
        if st.button("＋ 添加平空条件", key=_k("add_close")):
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
                               index=0, key=_k("pos_mode"))
    if pos_mode in ("fixed_pct", "strength_mapped"):
        dv = float(ep.get("value", 0.9 if pos_mode == "fixed_pct" else 0.5))
        pos_value = pc[1].slider("仓位比例", 0.05, 1.5, dv, 0.05, key=_k("pos_v"))
    elif pos_mode == "fixed_amount":
        pos_value = pc[1].number_input("每笔金额（元）", value=float(ep.get("value", 10_000)),
                                       step=1000.0, key=_k("pos_v"))
    else:  # atr_pct
        pos_value = 0.0
        pc[1].slider("单笔风险比例", 0.001, 0.05, float(ep.get("risk_per_trade", 0.01)),
                     0.001, key=_k("pos_risk"))
        pc[2].slider("ATR 止损倍数", 0.5, 5.0, float(ep.get("stop_atr_mult", 2.0)),
                     0.1, key=_k("pos_stop"))
    risk = st.session_state.get(_k("pos_risk"), 0.01)
    stop = st.session_state.get(_k("pos_stop"), 2.0)
    max_pos = pc[2].slider("最大仓位上限", 0.1, 2.0, float(ep.get("max_position", 1.0)),
                           0.05, key=_k("pos_max"))

    st.divider()
    st.markdown("##### 策略参数")
    eparams = (edit_cfg or {}).get("params", {}) or {}
    max_hold = st.number_input("最大持仓时间（K线根数，0 = 不限）",
                               value=int(eparams.get("max_hold_bars") or 0), min_value=0,
                               max_value=10000, step=1, key=_k("max_hold"))

    config = _build_config(meta, groups, as_map, name_zh, name_en, desc,
                           allow_short, only_long, max_hold, pos_mode, pos_value,
                           risk, stop, max_pos)

    st.divider()
    with st.expander("实时预览（声明式 JSON 配置）", expanded=True):
        st.json(config)
    # 缓存本轮配置，供下一轮顶部的 AI 面板作为「当前配置」发给大模型
    st.session_state["rc_ai_current"] = config

    st.divider()
    bt = st.columns(2)
    save_draft = bt[0].button("保存策略（草稿）", type="secondary", use_container_width=True,
                              key=_k("save"))
    save_bt = bt[1].button("保存并回测", type="primary", use_container_width=True,
                           key=_k("save_bt"))

    # 回测选择（保存并回测时使用）
    status = api.data_status()
    symbols = [m["symbol"] for m in status.get("symbols", []) if m.get("rows")]
    bc = st.columns([3, 2])
    bt_sym = bc[0].selectbox("回测标的", symbols if symbols else ["（暂无数据）"],
                             key=_k("bt_sym"))
    bt_tf = bc[1].selectbox("回测周期", TIMEFRAMES, index=TIMEFRAMES.index("1d"),
                            format_func=tf_label, key=_k("bt_tf"))

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
