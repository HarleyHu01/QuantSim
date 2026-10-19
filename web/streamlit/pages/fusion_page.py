"""策略融合配置页：大模型集成 × 因子策略。

这一页回答三个问题：

1. **启用哪些大模型**（Kronos / 统计基线 / 通用时序模型）以及各自权重；
2. **启用哪些因子策略**（双均线 / MACD / RSI 突破 / 动量 …）以及各自权重；
3. **两侧怎么缝起来**（加权混合 / 模型闸门 / 因子闸门 / 投票制）与两侧份额。

保存后写入 kv_store「ml.fusion」，策略「模型因子融合」下次回测/模拟盘即用这套配置。
"""
from __future__ import annotations

import streamlit as st

from .. import api
from ..components import TIMEFRAMES, inject_theme, tf_label


def _err_detail(e: Exception) -> str:
    try:
        d = e.response.json()  # type: ignore[attr-defined]
        return str(d.get("detail") or d)
    except Exception:  # noqa: BLE001
        return str(e)


#: 融合方式 → 中文说明（后端 fusion_modes 已带 label，这里只补一句「什么时候用」）
FUSION_HINTS = {
    "blend": "同向加强、反向抵消，最常用",
    "model_gate": "模型只当开关，方向也听模型的",
    "factor_gate": "因子只当开关，方向也听因子的",
    "vote": "谁的话语权重合计占优听谁的",
}

KIND_LABELS = {
    "dl": "深度学习（金融专用）",
    "stat": "统计基线（零依赖）",
    "external": "通用时序基础模型（需额外安装）",
}

# 试算面板默认上下文/预测长度，与后端默认配置保持一致
_DEFAULT_CONTEXT = 400
_DEFAULT_PRED_LEN = 12


def _weights_from_json(raw) -> dict:
    """服务端返回的权重可能是 JSON 字符串，也可能是 dict。"""
    if isinstance(raw, dict):
        return raw
    try:
        import json

        return json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001
        return {}


def _factor_rows(cfg: dict) -> list[dict]:
    """把因子权重 dict 转成 dataframe 行。"""
    out = []
    for k, v in (cfg.get("factor_weights") or {}).items():
        out.append({"标识": k, "权重": float(v or 0.0)})
    return out


def render(api_client: api.ApiClient) -> None:
    inject_theme()
    st.markdown("## 策略融合")
    st.caption("把「大模型集成」和「因子策略」两条信号缝成一条：自由选择模型、因子、"
               "融合方式与两侧份额，保存后策略『模型因子融合』在回测/模拟盘中直接生效。")

    # ------------------------------------------------------------ 载入
    try:
        cfg = api_client.fusion_config()
    except Exception as e:  # noqa: BLE001
        st.error(f"读取融合配置失败：{_err_detail(e)}")
        return

    conf = cfg.get("config") or {}
    model_weights = _weights_from_json(conf.get("model_weights"))
    factor_weights = _weights_from_json(conf.get("factor_weights"))
    saved_errors = cfg.get("errors") or []
    if saved_errors:
        st.warning("当前保存的配置有问题：" + "；".join(saved_errors))

    # 可选清单：ml_models 带 available/name/kind，融合页据此禁用未装依赖的模型
    try:
        models = api_client.ml_models()
    except Exception as e:  # noqa: BLE001
        models = []
        st.warning(f"读取模型清单失败（{_err_detail(e)}），仍可配置因子侧。")
    # ml_models 返回 {models:[...]} 或裸 list，两种都兼容
    if isinstance(models, dict):
        models = models.get("models") or []

    model_avail = {m.get("key"): bool(m.get("available")) for m in models}
    model_label = {m.get("key"): (m.get("name") or m.get("key")) for m in models}
    model_kind = {m.get("key"): (m.get("kind") or "stat") for m in models}

    # ------------------------------------------------------------ 一、大模型
    st.subheader("1 · 启用哪些大模型")
    st.caption("权重 ≥ 0，0 = 不启用。Kronos 与通用时序模型需要额外安装依赖，未安装时不可选。")

    new_model_weights: dict[str, float] = {}
    for kind in ("dl", "stat", "external"):
        keys = [k for k in model_weights
                if model_kind.get(k, "stat") == kind and k in model_weights]
        if not keys:
            continue
        st.markdown(f"**{KIND_LABELS.get(kind, kind)}**")
        cols = st.columns(min(len(keys), 4) or 1)
        for i, k in enumerate(keys):
            with cols[i % len(cols)]:
                avail = model_avail.get(k, False)
                new_model_weights[k] = st.number_input(
                    f"{model_label.get(k, k)}",
                    min_value=0.0, max_value=10.0, step=0.1,
                    value=float(model_weights.get(k) or 0.0),
                    key=f"fw_m_{k}",
                    disabled=not avail,
                    help="未安装依赖，暂不可用" if not avail else None,
                )
        if len(keys) > 4:
            st.caption("")

    if not any(v > 0 for v in new_model_weights.values()):
        st.info("当前没有启用任何大模型 —— 融合策略将只使用因子侧信号。")

    # 门控（与大模型集成共用同一套阈值，语义见模型预测页）
    st.markdown("**模型侧门控**")
    gc = st.columns(4)
    min_abs_return = gc[0].number_input(
        "最小预期收益", min_value=0.0, max_value=0.5, step=0.001,
        value=float(cfg.get("min_abs_return") or 0.0), format="%.3f",
        key="fw_min_abs", help="|预期收益| 低于该值视为无方向")
    min_return_sigma = gc[1].number_input(
        "最小收益/波动", min_value=0.0, max_value=10.0, step=0.1,
        value=float(cfg.get("min_return_sigma") or 0.0), key="fw_min_sig",
        help="信噪比，越高越严格")
    min_agreement = gc[2].number_input(
        "最小一致度", min_value=0.0, max_value=1.0, step=0.05,
        value=float(cfg.get("min_agreement") or 0.0), key="fw_min_agr",
        help="各模型方向一致的比例")
    max_dispersion = gc[3].number_input(
        "最大分歧度", min_value=0.0, max_value=20.0, step=0.1,
        value=float(cfg.get("max_dispersion") or 0.0), key="fw_max_dis",
        help="模型间预期收益的离散程度")
    gc2 = st.columns(4)
    min_models = gc2[0].number_input(
        "最少模型数", min_value=1, max_value=9, step=1,
        value=int(cfg.get("min_models") or 1), key="fw_min_models",
        help="至少这么多个模型给出有效预测才开仓")
    pred_len = gc2[1].number_input(
        "预测长度(根)", min_value=1, max_value=120, step=1,
        value=int(cfg.get("pred_len") or _DEFAULT_PRED_LEN), key="fw_pred_len")
    context = gc2[2].number_input(
        "上下文(根)", min_value=32, max_value=4000, step=32,
        value=int(cfg.get("context") or _DEFAULT_CONTEXT), key="fw_context")
    position = gc2[3].number_input(
        "模型侧仓位", min_value=0.0, max_value=1.0, step=0.05,
        value=float(cfg.get("position") or 1.0), key="fw_position")
    allow_short = st.checkbox("允许做空（模型侧）", value=bool(cfg.get("allow_short")),
                              key="fw_short")

    # ------------------------------------------------------------ 二、因子策略
    st.divider()
    st.subheader("2 · 启用哪些因子策略")
    st.caption("权重 ≥ 0，0 = 不启用。因子策略与回测页用的是同一套代码，可单独回测验证。")

    new_factor_weights: dict[str, float] = {}
    fkeys = list(factor_weights.keys())
    fcols = st.columns(min(len(fkeys), 4) or 1)
    for i, k in enumerate(fkeys):
        with fcols[i % len(fcols)]:
            new_factor_weights[k] = st.number_input(
                str(k), min_value=0.0, max_value=10.0, step=0.1,
                value=float(factor_weights.get(k) or 0.0), key=f"fw_f_{k}",
            )
    if len(fkeys) > 4:
        st.caption("")

    if not any(v > 0 for v in new_factor_weights.values()):
        st.info("当前没有启用任何因子策略 —— 融合策略将只使用模型侧信号。")

    # ------------------------------------------------------------ 三、融合方式
    st.divider()
    st.subheader("3 · 两侧怎么结合")
    mode_opts = [m.get("key") for m in (cfg.get("fusion_modes") or []) if m.get("key")]
    if not mode_opts:
        mode_opts = ["blend", "model_gate", "factor_gate", "vote"]
    mode_labels = {m.get("key"): m.get("label") for m in (cfg.get("fusion_modes") or [])}
    current_mode = conf.get("fusion_mode") or "blend"
    mode = st.radio(
        "融合方式",
        options=mode_opts,
        format_func=lambda k: f"{mode_labels.get(k, k)}（{FUSION_HINTS.get(k, '')}）",
        index=max(0, mode_opts.index(current_mode)),
        key="fw_mode",
    )
    mc, fc = st.columns(2)
    model_share = mc.slider("模型侧份额", 0.0, 1.0,
                            float(conf.get("model_share") or 0.5), 0.05, key="fw_m_share")
    fc.slider("因子侧份额", 0.0, 1.0, float(1.0 - model_share), 0.05,
              key="fw_f_share", disabled=True)
    st.caption(f"两侧份额按 **{model_share:.0%} : {1 - model_share:.0%}** 折算权重比"
               f"（{model_share:.2f} : {1 - model_share:.2f}）。份额只影响相对强弱，"
               "单侧为 0 时另一侧自动接管。")
    confidence_weighted = st.checkbox(
        "按预测置信度加权模型侧", value=bool(conf.get("confidence_weighted")),
        key="fw_conf_w", help="模型自己没把握时少说话，让因子主导")

    # ------------------------------------------------------------ 保存
    payload = {
        "model_weights": {k: float(v) for k, v in new_model_weights.items()},
        "factor_weights": {k: float(v) for k, v in new_factor_weights.items()},
        "fusion_mode": mode,
        "model_share": float(model_share),
        "confidence_weighted": bool(confidence_weighted),
        "min_abs_return": float(min_abs_return),
        "min_return_sigma": float(min_return_sigma),
        "min_agreement": float(min_agreement),
        "max_dispersion": float(max_dispersion),
        "min_models": int(min_models),
        "pred_len": int(pred_len),
        "context": int(context),
        "position": float(position),
        "allow_short": bool(allow_short),
    }

    st.divider()
    c_save, c_try = st.columns([1, 2])
    with c_save:
        if st.button("保存融合配置", type="primary", use_container_width=True):
            try:
                api_client.fusion_config_put(payload)
                st.success("已保存。策略『模型因子融合』下次回测/模拟盘会使用这套配置。")
            except Exception as e:  # noqa: BLE001
                st.error(f"保存失败：{_err_detail(e)}")
    with c_try:
        st.caption("保存后可在下方试算，或直接去「策略回测」对『模型因子融合』跑历史验证。")
    # ------------------------------------------------------------ 试算
    st.divider()
    st.subheader("试算：用最近一段行情看看融合信号")
    st.caption("试算不会保存任何结果，纯粹用于确认「这套配置现在会下什么单」。")

    try:
        rows = (api_client.data_status() or {}).get("symbols") or []
    except Exception as e:  # noqa: BLE001
        rows = []
        st.warning(f"读取数据状态失败：{_err_detail(e)}")
    symbols = [r["symbol"] for r in rows if r.get("rows")]
    if not symbols:
        st.info("还没有行情数据，请先到「数据管理」下载。")
        return

    sc = st.columns(3)
    sym = sc[0].selectbox("标的", symbols, key="fw_sym")
    tf = sc[1].selectbox("周期", TIMEFRAMES, index=TIMEFRAMES.index("5m"),
                         format_func=tf_label, key="fw_tf")
    bars = sc[2].number_input("K 线根数", 64, 4000, int(context), 32, key="fw_bars")

    if st.button("开始试算", key="fw_run"):
        try:
            with st.spinner("融合试算中…"):
                r = api_client.fusion_preview(symbol=sym, timeframe=tf, bars=int(bars),
                                              config=payload)
        except Exception as e:  # noqa: BLE001
            st.error(f"试算失败：{_err_detail(e)}")
            return

        if not r.get("ok"):
            st.warning(r.get("reason") or "数据不足，无法试算")
            return

        st.success("试算完成（未保存）")
        m = st.columns(4)
        m[0].metric("融合仓位", f"{float(r.get('target_weight') or 0):.2f}")
        m[1].metric("模型侧", f"{float(r.get('model_signal') or 0):.2f}")
        m[2].metric("因子侧", f"{float(r.get('factor_signal') or 0):.2f}")
        m[3].metric("预测置信度", f"{float(r.get('confidence') or 0):.0%}")
        st.info(r.get("reason") or "")

        st.caption(f"模型侧：{r.get('model_reason') or '—'}")
        st.caption(f"因子侧：{r.get('factor_reason') or '—'}")
        st.caption(f"最后一根：{r.get('last_bar')}，收盘 {r.get('last_close')}，"
                   f"上下文 {r.get('context_bars')} 根，耗时 {r.get('elapsed_seconds')}s")

        votes = r.get("votes") or []
        if votes:
            st.markdown("**模型投票**")
            st.dataframe(votes, use_container_width=True, hide_index=True)

        gates = r.get("gates") or []
        if gates:
            st.markdown("**门控**")
            st.dataframe(gates, use_container_width=True, hide_index=True)
