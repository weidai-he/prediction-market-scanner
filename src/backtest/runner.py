"""Simple backtest runner for low-probability market opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.metrics import compact_summary, summarize_backtest


MARKET_PROB_ALIASES = ("market_prob", "implied_prob")
CLOSE_TIME_ALIASES = ("close_time", "market_close_time", "end_date")


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the scanner backtest."""

    min_market_prob: float = 0.0
    max_market_prob: float = 0.10
    min_edge: float = 0.03
    fixed_stake: float = 100.0
    decisions_csv_path: str = "artifacts/backtest/daily_decisions.csv"


def _get_first_present_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    """Return the first matching column name from a list of aliases."""

    for alias in aliases:
        if alias in frame.columns:
            return alias
    return None


def _resolve_market_prob_column(markets: pd.DataFrame) -> str:
    """Resolve the market probability column or raise a helpful error."""

    column = _get_first_present_column(markets, MARKET_PROB_ALIASES)
    if column is None:
        raise ValueError("Expected one of these market probability columns: market_prob, implied_prob")
    return column


def _resolve_close_time_column(markets: pd.DataFrame) -> str | None:
    """Resolve an optional market close time column."""

    return _get_first_present_column(markets, CLOSE_TIME_ALIASES)


def _coerce_probability_series(series: pd.Series) -> pd.Series:
    """Convert a series into numeric probabilities clipped to [0, 1]."""

    return pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0)


def _prepare_backtest_frame(markets: pd.DataFrame, *, config: BacktestConfig) -> pd.DataFrame:
    """Normalize columns used by the backtest strategy."""

    prepared = markets.copy()
    market_prob_column = _resolve_market_prob_column(prepared)
    prepared["market_prob"] = _coerce_probability_series(prepared[market_prob_column])

    if "model_prob" not in prepared.columns:
        raise ValueError("Backtest input must include a model_prob column")
    prepared["model_prob"] = _coerce_probability_series(prepared["model_prob"])

    if "resolved_yes" not in prepared.columns:
        raise ValueError("Backtest input must include a resolved_yes column")
    prepared["resolved_yes"] = pd.to_numeric(prepared["resolved_yes"], errors="coerce").clip(lower=0.0, upper=1.0)

    close_time_column = _resolve_close_time_column(prepared)
    prepared["close_time"] = (
        pd.to_datetime(prepared[close_time_column], errors="coerce", utc=True)
        if close_time_column is not None
        else pd.NaT
    )

    if "date" in prepared.columns:
        prepared["decision_date"] = pd.to_datetime(prepared["date"], errors="coerce", utc=True).dt.normalize()
    elif prepared["close_time"].notna().any():
        prepared["decision_date"] = prepared["close_time"].dt.normalize()
    else:
        prepared["decision_date"] = pd.Timestamp("1970-01-01", tz="UTC")

    if "source_platform" not in prepared.columns:
        prepared["source_platform"] = "unknown"

    prepared["edge"] = prepared["model_prob"] - prepared["market_prob"]
    prepared["stake"] = float(config.fixed_stake)
    prepared["entered"] = False
    prepared["pnl"] = 0.0
    prepared["entry_price"] = prepared["market_prob"]
    return prepared


def _apply_strategy(prepared: pd.DataFrame, *, config: BacktestConfig) -> pd.DataFrame:
    """Apply the simple threshold strategy and compute trade-level PnL.

    Strategy:
    - only consider low-probability markets
    - enter when model_prob exceeds market_prob by at least the configured edge
    - use a fixed stake per trade

    PnL assumes buying the YES side at the observed market probability.
    If the event resolves YES, payout is stake * ((1 / price) - 1) in profit.
    If the event resolves NO, the stake is lost.
    """

    decisions = prepared.copy()
    eligible = decisions["market_prob"].between(config.min_market_prob, config.max_market_prob, inclusive="both")
    eligible &= decisions["edge"] >= config.min_edge
    eligible &= decisions["model_prob"].notna() & decisions["market_prob"].notna() & decisions["resolved_yes"].notna()

    decisions.loc[eligible, "entered"] = True
    decisions["outcome"] = decisions["resolved_yes"].map({1.0: "yes", 0.0: "no"}).fillna("unknown")

    entered = decisions["entered"].astype(bool)
    price = decisions.loc[entered, "entry_price"].clip(lower=0.01, upper=0.99)
    stake = decisions.loc[entered, "stake"]
    resolved_yes = decisions.loc[entered, "resolved_yes"]

    shares = stake / price
    payouts = shares * resolved_yes
    decisions.loc[entered, "pnl"] = payouts - stake

    decisions = decisions.sort_values(by=["decision_date", "market_id"], na_position="last").reset_index(drop=True)
    decisions["cumulative_pnl"] = pd.to_numeric(decisions["pnl"], errors="coerce").fillna(0.0).cumsum()
    return decisions


def get_equity_curve(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return a simple daily equity curve DataFrame for dashboard plotting."""

    if decisions.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "cumulative_pnl", "trades"])

    curve = decisions.copy()
    curve["date"] = pd.to_datetime(curve["decision_date"], errors="coerce", utc=True).dt.normalize()
    curve["pnl"] = pd.to_numeric(curve["pnl"], errors="coerce").fillna(0.0)
    curve["entered"] = curve["entered"].fillna(False).astype(bool)

    equity_curve = (
        curve.groupby("date", dropna=False)
        .agg(
            daily_pnl=("pnl", "sum"),
            trades=("entered", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )
    equity_curve["cumulative_pnl"] = equity_curve["daily_pnl"].cumsum()
    return equity_curve


def save_daily_decisions_csv(decisions: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist daily backtest decisions to CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    export = decisions.copy()
    export["date"] = pd.to_datetime(export["decision_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    if "close_time" in export.columns:
        export["close_time"] = export["close_time"].astype(str)
    export["market_prob"] = pd.to_numeric(export["market_prob"], errors="coerce").fillna(0.0)
    export["model_prob"] = pd.to_numeric(export["model_prob"], errors="coerce").fillna(0.0)
    export["edge"] = pd.to_numeric(export["edge"], errors="coerce").fillna(0.0)
    export["stake"] = pd.to_numeric(export["stake"], errors="coerce").fillna(0.0)
    export["pnl"] = pd.to_numeric(export["pnl"], errors="coerce").fillna(0.0)
    export["source_platform"] = export["source_platform"].fillna("unknown")
    export["outcome"] = export["outcome"].fillna("unknown")

    csv_columns = [
        "date",
        "market_id",
        "source_platform",
        "market_prob",
        "model_prob",
        "edge",
        "stake",
        "outcome",
        "pnl",
    ]
    for column in csv_columns:
        if column not in export.columns:
            export[column] = pd.NA

    export.loc[:, csv_columns].to_csv(output, index=False)
    return output


def run_backtest(markets: pd.DataFrame, config: BacktestConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the threshold strategy and return decisions plus summary metrics."""

    config = config or BacktestConfig()
    prepared = _prepare_backtest_frame(markets, config=config)
    decisions = _apply_strategy(prepared, config=config)
    save_daily_decisions_csv(decisions, config.decisions_csv_path)
    metrics = summarize_backtest(decisions)
    metrics["compact_summary"] = compact_summary(decisions)
    metrics["decisions_csv_path"] = str(Path(config.decisions_csv_path))
    metrics["edge_threshold"] = config.min_edge
    metrics["max_market_prob"] = config.max_market_prob
    metrics["fixed_stake"] = config.fixed_stake
    return decisions, metrics


__all__ = [
    "BacktestConfig",
    "get_equity_curve",
    "run_backtest",
    "save_daily_decisions_csv",
]
