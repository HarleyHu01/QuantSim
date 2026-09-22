"""命令行回测。

用法:
  python scripts/run_backtest.py -s AAPL -tf 1d -st dual_ma
  python scripts/run_backtest.py -s AAPL -tf 1d -st dual_ma --engine vectorized
  python scripts/run_backtest.py -s AAPL -tf 1d -st dual_ma --params '{"fast": 10, "slow": 30}'
  python scripts/run_backtest.py --combo -s AAPL,SPY -st dual_ma,macd --weights '{"dual_ma": 60, "macd": 40}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.core.config import AppConfig  # noqa: E402
from quant.core.models import PortfolioConfig, TradingSettings  # noqa: E402
from quant.data.manager import DataManager  # noqa: E402
from quant.engine.backtest import EventDrivenBacktestEngine, VectorizedBacktestEngine  # noqa: E402
from quant.portfolio.portfolio import StrategyPortfolio  # noqa: E402
from quant.strategies.registry import create, ensure_loaded  # noqa: E402


def _symbol_meta(dm, symbols: list[str]) -> dict[str, dict]:
    out = {}
    for s in symbols:
        m = dm.symbols_repo.get(s)
        out[s] = {k: m[k] for k in ("price_limit_pct", "t_plus", "lot_size") if m and m.get(k) is not None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="QuantSim 命令行回测")
    ap.add_argument("-s", "--symbols", required=True, help="逗号分隔标的")
    ap.add_argument("-tf", "--timeframe", default="1d")
    ap.add_argument("-st", "--strategies", required=True, help="逗号分隔策略 key")
    ap.add_argument("--params", default=None, help='策略参数 JSON，如 {"dual_ma": {"fast": 10}}')
    ap.add_argument("-w", "--weights", default=None, help='组合权重 JSON，如 {"dual_ma": 60, "macd": 40}')
    ap.add_argument("--engine", choices=["event_driven", "vectorized"], default="event_driven")
    ap.add_argument("--settings", default=None, help='交易设置 JSON（覆盖默认）')
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    ensure_loaded()
    config = AppConfig.load()
    dm = DataManager.create(config)
    symbols = args.symbols.split(",")
    strategy_keys = args.strategies.split(",")
    params = json.loads(args.params) if args.params else None
    weights = json.loads(args.weights) if args.weights else None
    settings = TradingSettings(**(json.loads(args.settings) if args.settings else {}))

    symbol_bars = {}
    for s in symbols:
        bars = dm.load_bars(s, args.timeframe)
        if bars.empty:
            print(f"错误: {s} 无 {args.timeframe} 数据，请先 python scripts/update_data.py -s {s} -tf {args.timeframe}")
            return 1
        symbol_bars[s] = bars

    settings.timeframe = args.timeframe
    strategies = [create(k, (params or {}).get(k)) for k in strategy_keys]
    port_cfg = PortfolioConfig(
        mode="signal_merge" if weights else "equal",
        weights=weights or {},
    )
    st = StrategyPortfolio(port_cfg)
    engine = VectorizedBacktestEngine(settings, st) if args.engine == "vectorized" \
        else EventDrivenBacktestEngine(settings, st)
    result = engine.run(symbol_bars, strategies, _symbol_meta(dm, symbols))

    if args.json:
        print(json.dumps({
            "metrics": result.metrics,
            "per_strategy_metrics": result.per_strategy_metrics,
            "trades": [t.to_dict() for t in result.trades],
            "warnings": result.warnings,
        }, ensure_ascii=False, indent=2))
        return 0

    print(f"\n===== {args.engine} 回测: {','.join(symbols)} {args.timeframe} 策略={','.join(strategy_keys)} =====")
    m = result.metrics
    print(f"总收益率   : {m.get('total_return') or 0:.2%}")
    print(f"年化收益率 : {m.get('annual_return') or 0:.2%}")
    print(f"最大回撤   : {m.get('max_drawdown') or 0:.2%}")
    print(f"夏普比率   : {m.get('sharpe')}")
    print(f"索提诺     : {m.get('sortino')}")
    print(f"胜率       : {m.get('win_rate')}")
    print(f"盈亏比     : {m.get('profit_factor')}")
    print(f"交易次数   : {m.get('trade_count')}   耗时: {m.get('elapsed_seconds')}s")
    print(f"基准收益   : {m.get('benchmark_return') or 0:.2%}")
    if result.per_strategy_metrics:
        print("\n--- 单策略独立回测对比 ---")
        for k, pm in result.per_strategy_metrics.items():
            print(f"  {k:<12} 收益 {pm.get('total_return') or 0:.2%}  回撤 {pm.get('max_drawdown') or 0:.2%}"
                  f"  夏普 {pm.get('sharpe')}  胜率 {pm.get('win_rate')}")
    print("\n--- 交易明细（前 5 笔）---")
    for t in result.trades[:5]:
        print(f"  {t.timestamp:%Y-%m-%d} {t.symbol:<8} {t.side:<5} qty={t.qty:>10.0f} "
              f"price={t.price:>10.4f} fee={t.fee:>8.2f} pnl={t.pnl:>10.2f} {t.reason[:36]}")
    if result.warnings:
        print(f"\n警告 ({len(result.warnings)}): {result.warnings[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
