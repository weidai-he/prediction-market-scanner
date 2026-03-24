"""Demonstrate the backtest runner on synthetic market data.

Run with:
    python scripts/demo_backtest_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest.runner import BacktestConfig, get_equity_curve, run_backtest


def build_synthetic_data() -> pd.DataFrame:
    """Create a small synthetic historical dataset for demonstration."""

    rows: list[dict[str, object]] = []
    base_date = pd.Timestamp("2026-01-01", tz="UTC")

    for day_offset in range(10):
        decision_date = base_date + pd.Timedelta(days=day_offset)
        for market_number in range(5):
            market_prob = 0.02 + (0.015 * market_number)
            model_prob = market_prob + (0.01 * ((market_number % 3) + 1))
            resolved_yes = 1 if (day_offset + market_number) % 4 == 0 else 0
            liquidity = 1000 + 250 * market_number

            rows.append(
                {
                    "market_id": f"synthetic-{day_offset}-{market_number}",
                    "source_platform": "synthetic",
                    "title": f"Synthetic market {day_offset}-{market_number}",
                    "category": "weather",
                    "market_prob": round(market_prob, 4),
                    "model_prob": round(model_prob, 4),
                    "bid": round(max(market_prob - 0.01, 0.01), 4),
                    "ask": round(min(market_prob + 0.01, 0.99), 4),
                    "last": round(market_prob, 4),
                    "close_time": (decision_date + pd.Timedelta(days=7)).isoformat(),
                    "date": decision_date.isoformat(),
                    "resolved_yes": resolved_yes,
                    "liquidity": liquidity,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Run the synthetic demo and print decisions plus summary metrics."""

    markets = build_synthetic_data()
    output_dir = PROJECT_ROOT / "artifacts" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions, metrics = run_backtest(
        markets,
        BacktestConfig(
            max_market_prob=0.10,
            min_edge=0.02,
            fixed_stake=100.0,
            decisions_csv_path=str(output_dir / "synthetic_daily_decisions.csv"),
        ),
    )
    equity_curve = get_equity_curve(decisions)

    print("Metrics:")
    for key, value in metrics.items():
        if key == "compact_summary":
            continue
        print(f"  {key}: {value}")

    print("\nCompact summary:")
    for key, value in metrics["compact_summary"].items():
        print(f"  {key}: {value}")

    print("\nHead of decisions:")
    print(decisions.head(10).to_string(index=False))

    print("\nEquity curve head:")
    print(equity_curve.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
