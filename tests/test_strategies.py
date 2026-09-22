"""内置经典策略库测试：逐策略信号验证、注册表、引擎集成。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.core.models import PortfolioConfig, TradingSettings
from quant.engine.backtest import EventDrivenBacktestEngine
from quant.portfolio.portfolio import StrategyPortfolio
from quant.strategies.registry import create, ensure_loaded, list_all
from tests.strategies import make_bars

ALL_KEYS = ["bollinger", "dual_ma", "grid", "macd", "mean_reversion", "momentum", "rsi", "turtle"]


def rising_bars(n: int = 120, start: float = 100.0) -> pd.DataFrame:
    return make_bars([start + i for i in range(n)])


def falling_bars(n: int = 120, start: float = 200.0) -> pd.DataFrame:
    return make_bars([start - i for i in range(n)])


def dip_recover_bars() -> pd.DataFrame:
    """平盘 → 急跌 → 回升（触发超卖/均值回归/布林下轨类策略）。"""
    prices = [100.0] * 30 + [80.0] * 10 + [105.0] * 30
    return make_bars(prices)


def breakout_bars() -> pd.DataFrame:
    """长期窄幅 → 突破（触发动量/海龟类策略）。"""
    prices = [100.0] * 40 + [110.0] * 20
    return make_bars(prices)


def run_signals(key: str, bars: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return create(key, params).generate_signals(bars, None)


# ---------- 注册表 ----------

def test_registry_has_eight_builtin_strategies():
    ensure_loaded()
    metas = {m["key"]: m for m in list_all()}
    assert set(ALL_KEYS) <= set(metas)
    for k in ALL_KEYS:
        assert "params_schema" in metas[k]
        assert "params_defaults" in metas[k]  # 每个参数必须有默认值
        assert metas[k]["name"]


def test_params_schema_has_ui_metadata():
    """UI 参数表单契约：数值类型带 min/max，bool 带默认值（仅校验 8 个内置策略）。"""
    ensure_loaded()
    for m in list_all():
        if m["key"] not in ALL_KEYS:
            continue
        props = m["params_schema"].get("properties", {})
        assert props, f"{m['key']} 参数 schema 为空"
        for name, spec in props.items():
            if spec.get("type") in ("integer", "number"):
                assert "minimum" in spec and "maximum" in spec, \
                    f"{m['key']}.{name} 缺少 minimum/maximum"
            elif spec.get("type") == "boolean":
                assert "default" in spec, f"{m['key']}.{name} 缺少默认值"


# ---------- 信号契约（通用） ----------

@pytest.mark.parametrize("key", ALL_KEYS)
def test_signal_contract(key):
    """所有策略输出：列=[target_weight, reason, confidence]，index 对齐，取值范围合法。"""
    bars = rising_bars()
    df = run_signals(key, bars)
    assert list(df.columns) == ["target_weight", "reason", "confidence"]
    assert df.index.equals(bars.index)
    assert df["target_weight"].between(-1, 1).all()
    assert df["confidence"].between(0, 1).all()
    assert df["reason"].apply(lambda s: isinstance(s, str)).all()


# ---------- 逐策略行为 ----------

def test_dual_ma_trend_following():
    # 平台 → 下跌 → 拉升：金叉做多
    up = run_signals("dual_ma", make_bars([120.0] * 30 + [100.0] * 20 + [130.0] * 50))
    assert (up["target_weight"] == 1).any()
    # 平台 → 下跌：死叉做空（only_long=False）
    down = run_signals("dual_ma", make_bars([100.0] * 80 + [80.0] * 40), {"only_long": False})
    assert (down["target_weight"] == -1).any()


def test_macd_golden_cross():
    df = run_signals("macd", dip_recover_bars(), {"fast": 3, "slow": 8, "signal": 2})
    assert (df["target_weight"] == 1).any()  # 急跌后回升触发金叉
    assert (df["target_weight"] == -1).sum() == 0  # only_long 不做空


def test_rsi_oversold_entry():
    df = run_signals("rsi", dip_recover_bars())
    assert (df["target_weight"] == 1).any()  # RSI 脱离超卖区买入
    assert df["reason"].str.contains("超卖").any()


def test_bollinger_lower_band_entry():
    df = run_signals("bollinger", dip_recover_bars())
    assert (df["target_weight"] == 1).any()  # 跌破下轨买入
    assert df["reason"].str.contains("跌破下轨").any()


def test_momentum_sign():
    up = run_signals("momentum", rising_bars(), {"lookback": 10})
    assert (up["target_weight"] == 1).any()
    down = run_signals("momentum", falling_bars(), {"lookback": 10})
    assert (down["target_weight"] == 0).all()  # only_long：下跌不入场


def test_mean_reversion_oversold():
    df = run_signals("mean_reversion", dip_recover_bars(), {"entry_z": 1.5})
    assert (df["target_weight"] == 1).any()  # Z 低于 -entry_z 买入


def test_grid_position_by_price():
    """价格低于锚点 → 加仓（target > base_position）；高于锚点 → 减仓。"""
    below = run_signals("grid", make_bars([100.0] * 5 + [90.0] * 5))
    assert (below["target_weight"] > 0.5).any()  # base_position=0.5
    above = run_signals("grid", make_bars([100.0] * 5 + [110.0] * 5))
    assert (above["target_weight"] < 0.5).any()
    assert below["reason"].str.contains("网格").any()


def test_turtle_breakout_entry():
    """突破入场：目标仓位为正（ATR 缩放：单单位 ≈ risk/(2×N)）。"""
    df = run_signals("turtle", breakout_bars(), {"entry_n": 10, "exit_n": 5})
    assert (df["target_weight"] > 0.1).any()  # 突破 10 日高点入场
    assert df["reason"].str.contains("突破").any()
    down = run_signals("turtle", falling_bars(), {"entry_n": 10, "exit_n": 5})
    assert (down["target_weight"] < -0.1).any()  # 下跌破位做空


def test_turtle_pyramiding():
    """突破后持续上涨 → 0.5N 加仓至多单位（目标仓位 > 单单位）。"""
    prices = [100.0] * 40 + [110.0, 112, 115, 119, 124, 130, 137, 145]
    df = run_signals("turtle", make_bars(prices),
                     {"entry_n": 10, "exit_n": 5, "atr_period": 5})
    tw = df["target_weight"]
    assert tw.max() > 0.4  # 加仓后超过单单位
    assert tw.max() <= 1.0
    assert "加仓" in " ".join(df["reason"])


def test_turtle_stop_loss():
    """突破后回落 → 2N 止损离场；only_long 下止损后保持空仓（不再反向开仓）。"""
    prices = [100.0] * 40 + [110.0, 105, 100, 96, 92]
    df = run_signals("turtle", make_bars(prices),
                     {"entry_n": 10, "exit_n": 5, "atr_period": 5, "only_long": True})
    assert "止损" in " ".join(df["reason"])
    tail = df["target_weight"].iloc[-3:]
    assert (tail == 0).all()  # 止损后空仓


# ---------- 引擎集成 ----------

@pytest.mark.parametrize("key", ALL_KEYS)
def test_all_strategies_engine_integration(key):
    """每个内置策略都能跑通事件驱动引擎并产出指标。"""
    bars = dip_recover_bars()
    settings = TradingSettings(commission_rate=0.0, commission_min=0.0, commission_fixed=0.0,
                               stamp_tax_rate=0.0, transfer_fee_rate=0.0, slippage_bps=0.0,
                               signal_threshold=0.0, max_position_pct=1.0, t_plus_enabled=False)
    engine = EventDrivenBacktestEngine(
        settings, StrategyPortfolio(PortfolioConfig(mode="signal_merge", weights={key: 100.0})))
    result = engine.run({"X": bars}, [create(key)])
    assert result.metrics.get("total_return") is not None
    assert key in result.per_strategy_metrics
    assert isinstance(result.equity, pd.DataFrame) and len(result.equity) == len(bars)


# ---------- 声明式规则策略（手动创建 / LLM 共用） ----------

def _rule_cfg(**over):
    cfg = {
        "name": "金叉策略",
        "description": "测试",
        "indicators": [
            {"name": "sma", "params": {"window": 20}, "as": "sma20"},
            {"name": "sma", "params": {"window": 60}, "as": "sma60"},
        ],
        "entry_long": {"logic": "all", "conditions": [
            {"lhs": "sma20", "op": "crosses_above", "rhs": "sma60"}]},
        "exit_long": {"logic": "all", "conditions": [
            {"lhs": "sma20", "op": "crosses_below", "rhs": "sma60"}]},
        "position": {"mode": "fixed_pct", "value": 0.9, "max_position": 1.0},
        "params": {"allow_short": False, "only_long": True, "max_hold_bars": None},
    }
    cfg.update(over)
    return cfg


def _rule_bars(n=300, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 10 * np.sin(t / 25) + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1_000, 10_000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


def test_rule_strategy_validates_and_trades():
    from quant.strategies.llm.schema import RuleStrategyConfig
    from quant.strategies.llm.validator import validate_config
    from quant.strategies.rule_strategy import RuleStrategy

    cfg = RuleStrategyConfig(**_rule_cfg())
    assert validate_config(cfg) == []

    s = RuleStrategy(config=cfg.model_dump())
    sig = s.generate_signals(_rule_bars(), None)
    assert sig["target_weight"].between(-1, 1).all()
    assert (sig["target_weight"] > 0).sum() > 0  # 有持仓段
    assert (sig["target_weight"] == 0).sum() > 0  # 有空仓段

    # 引擎集成
    settings = TradingSettings(commission_rate=0.0, commission_min=0.0, commission_fixed=0.0,
                               stamp_tax_rate=0.0, transfer_fee_rate=0.0, slippage_bps=0.0,
                               signal_threshold=0.0, max_position_pct=1.0, t_plus_enabled=False)
    engine = EventDrivenBacktestEngine(
        settings, StrategyPortfolio(PortfolioConfig(mode="signal_merge", weights={"rule": 100.0})))
    r = engine.run({"X": _rule_bars()}, [s])
    assert r.metrics["trade_count"] >= 2


def test_rule_validator_rejects_bad_config():
    from quant.strategies.llm.schema import RuleStrategyConfig
    from quant.strategies.llm.validator import validate_config

    bad = RuleStrategyConfig(
        name="坏", indicators=[{"name": "nope", "params": {}, "as": "x"}],
        entry_long={"logic": "all", "conditions": [{"lhs": "x", "op": "???", "rhs": 1}]},
    )
    errs = validate_config(bad)
    assert any("未知指标" in e for e in errs)
    assert any("未知操作符" in e for e in errs)
    assert any("未定义" in e for e in errs)

    # between 缺区间 / 做空互斥
    bad2 = RuleStrategyConfig(
        name="b2",
        entry_long={"logic": "all", "conditions": [
            {"lhs": "close", "op": "between", "rhs": 1}]},
        params={"allow_short": True, "only_long": True},
    )
    errs2 = validate_config(bad2)
    assert any("区间" in e for e in errs2)
    assert any("互斥" in e for e in errs2)


def test_rule_strategy_max_hold_force_exit():
    from quant.strategies.llm.schema import RuleStrategyConfig
    from quant.strategies.rule_strategy import RuleStrategy

    # 买入后无卖出条件，仅靠最大持仓到期平仓
    cfg = RuleStrategyConfig(**{
        "name": "限时持有", "indicators": [{"name": "sma", "params": {"window": 5}, "as": "s5"}],
        "entry_long": {"logic": "all", "conditions": [
            {"lhs": "s5", "op": ">", "rhs": "s5"}]},  # 恒假 → 用 close>0 恒真
    })
    cfg = RuleStrategyConfig(**{
        "name": "限时持有",
        "entry_long": {"logic": "all", "conditions": [{"lhs": "close", "op": ">", "rhs": 0}]},
        "params": {"max_hold_bars": 5},
    })
    s = RuleStrategy(config=cfg.model_dump())
    sig = s.generate_signals(_rule_bars(), None)
    # 第一根买入，5 根后平仓，反复
    hold_lens = []
    cur = 0
    for v in sig["target_weight"]:
        if v > 0:
            cur += 1
        else:
            if cur:
                hold_lens.append(cur)
            cur = 0
    if cur:
        hold_lens.append(cur)
    assert hold_lens and max(hold_lens) <= 5


def test_rule_strategy_registry_loader(tmp_path, monkeypatch):
    """保存到策略表后，registry.create(key) 能按 key 加载声明式策略。"""
    import json

    from quant.storage.db import init_db
    from quant.storage.repositories import StrategiesRepo

    from quant.strategies.rule_strategy import RuleStrategy

    db = tmp_path / "t.db"
    init_db(str(db))
    repo = StrategiesRepo(str(db))
    cfg = _rule_cfg()
    repo.upsert(key="golden_cross", name=cfg["name"], description=cfg["description"],
                strategy_type="llm_generated", class_name="RuleStrategy",
                config_json=cfg)

    from quant.strategies.registry import set_rule_loader

    def loader(key, params=None):
        import json

        row = repo.get(key)
        if row and row.get("strategy_type") == "llm_generated":
            return RuleStrategy(config=json.loads(row["config_json"]))
        return None

    set_rule_loader(loader)
    try:
        s = create("golden_cross")
        assert isinstance(s, RuleStrategy)
        sig = s.generate_signals(_rule_bars(), None)
        assert (sig["target_weight"] > 0).any()
    finally:
        set_rule_loader(None)


def test_api_strategy_delete_rules(tmp_path):
    """策略删除：内置 403；规则策略可删；被组合引用时级联清理组合权重。"""
    import json

    from fastapi import HTTPException

    from quant.api.routers import strategies as router
    from quant.api.state import app_state
    from quant.storage.db import init_db
    from quant.storage.repositories import PortfolioRepo, StrategiesRepo

    db = tmp_path / "t3.db"
    init_db(str(db))
    repo = StrategiesRepo(str(db))
    port_repo = PortfolioRepo(str(db))
    app_state["strategies_repo"] = repo
    app_state["portfolio_repo"] = port_repo
    try:
        # 内置策略 → 403
        repo.upsert(key="dual_ma", name="双均线", strategy_type="builtin",
                    class_name="DualMA", config_json={})
        builtin = repo.get("dual_ma")
        try:
            router.delete_strategy(builtin["id"])
            assert False, "内置策略删除应抛 403"
        except HTTPException as e:
            assert e.status_code == 403
            assert "内置" in e.detail
        assert repo.get("dual_ma") is not None  # 未被删除

        # 规则策略未引用 → 正常删除
        repo.upsert(key="my_rule", name="我的规则", strategy_type="llm_generated",
                    class_name="RuleStrategy", config_json=_rule_cfg())
        rid = repo.get("my_rule")
        r = router.delete_strategy(rid["id"])
        assert r["ok"] and r["removed_from_portfolio"] is False
        assert repo.get("my_rule") is None

        # 规则策略被组合引用 → 删除后级联清理组合权重
        repo.upsert(key="my_rule2", name="引用测试", strategy_type="llm_generated",
                    class_name="RuleStrategy", config_json=_rule_cfg())
        port_repo.put(mode="signal_merge", weights={"dual_ma": 60.0, "my_rule2": 40.0},
                      rebalance_freq="none", signal_threshold=0.0)
        rid2 = repo.get("my_rule2")
        r2 = router.delete_strategy(rid2["id"])
        assert r2["ok"] and r2["removed_from_portfolio"] is True
        saved = port_repo.get()
        weights = json.loads(saved["weights_json"])
        assert "my_rule2" not in weights
        assert weights.get("dual_ma") == 60.0  # 其他权重保留
    finally:
        app_state.clear()


def test_api_rule_strategy_endpoint(tmp_path, monkeypatch):
    """POST /api/strategies/rule：校验通过入库；非法配置 400。"""
    import json

    from fastapi import HTTPException

    from quant.api.routers import strategies as router
    from quant.api.state import app_state
    from quant.storage.db import init_db
    from quant.storage.repositories import StrategiesRepo

    db = tmp_path / "t2.db"
    init_db(str(db))
    repo = StrategiesRepo(str(db))
    app_state["strategies_repo"] = repo
    try:
        r = router.add_rule_strategy(router.RuleStrategyIn(**_rule_cfg()))
        assert r["strategy"]["strategy_type"] == "llm_generated"
        assert r["strategy"]["enabled"] == 0  # 草稿
        assert "golden_cross" not in r["strategy"]["key"]  # key 由名称生成
        # 非法 → 400
        try:
            router.add_rule_strategy(router.RuleStrategyIn(
                name="坏", indicators=[{"name": "nope", "params": {}, "as": "x"}],
                entry_long={"logic": "all", "conditions": [{"lhs": "x", "op": "???", "rhs": 1}]}))
            assert False, "应抛 400"
        except HTTPException as e:
            assert e.status_code == 400
            assert "errors" in e.detail
        # 回测可直接用 key（registry 加载器走默认 loader，需配置真实 db → 此处验证路由本身）
        meta = router.list_indicators()
        assert any(i["key"] == "sma" for i in meta["indicators"])
        assert "crosses_above" in [o["key"] for o in meta["operators"]]
    finally:
        app_state.clear()
