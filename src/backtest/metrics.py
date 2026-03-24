"""Performance metrics for simple market-scanner backtests."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def compute_roi(total_pnl: float, total_staked: float) -> float:
    """Return return-on-investment as profit divided by capital staked."""

    if total_staked <= 0:
        return 0.0
    return total_pnl / total_staked


def compute_hit_rate(decisions: pd.DataFrame) -> float:
    """Return the fraction of executed trades that settled profitably."""

    executed = decisions.loc[decisions["entered"].astype(bool)].copy()
    if executed.empty:
        return 0.0
    wins = (executed["pnl"] > 0).sum()
    return float(wins) / float(len(executed))


def compute_average_edge(decisions: pd.DataFrame) -> float:
    """Return the average edge across executed trades."""

    executed = decisions.loc[decisions["entered"].astype(bool)].copy()
    if executed.empty:
        return 0.0
    return float(pd.to_numeric(executed["edge"], errors="coerce").mean())


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Return max drawdown from a cumulative PnL series."""

    if equity_curve.empty:
        return 0.0

    running_peak = equity_curve.cummax()
    drawdown = equity_curve - running_peak
    return float(drawdown.min())


def compute_brier_score(decisions: pd.DataFrame) -> float:
    """Return the Brier score using model probability vs. binary outcome."""

    executed = decisions.loc[decisions["entered"].astype(bool)].copy()
    if executed.empty:
        return 0.0

    model_prob = pd.to_numeric(executed["model_prob"], errors="coerce")
    outcome = pd.to_numeric(executed["resolved_yes"], errors="coerce")
    valid = model_prob.notna() & outcome.notna()
    if not valid.any():
        return 0.0

    squared_error = (model_prob[valid] - outcome[valid]) ** 2
    return float(squared_error.mean())


def summarize_backtest(decisions: pd.DataFrame) -> dict[str, Any]:
    """Compute a compact metric summary for a completed backtest."""

    executed = decisions.loc[decisions["entered"].astype(bool)].copy()
    total_pnl = float(pd.to_numeric(executed.get("pnl"), errors="coerce").fillna(0.0).sum())
    total_staked = float(pd.to_numeric(executed.get("stake"), errors="coerce").fillna(0.0).sum())

    equity_curve = pd.to_numeric(executed.get("cumulative_pnl"), errors="coerce").dropna()
    max_drawdown = compute_max_drawdown(equity_curve) if not equity_curve.empty else 0.0

    return {
        "trades": int(len(executed)),
        "total_pnl": total_pnl,
        "total_staked": total_staked,
        "roi": compute_roi(total_pnl, total_staked),
        "hit_rate": compute_hit_rate(decisions),
        "average_edge": compute_average_edge(decisions),
        "max_drawdown": max_drawdown,
        "brier_score": compute_brier_score(decisions),
    }


def compact_summary(decisions: pd.DataFrame) -> dict[str, float | int]:
    """Return a dashboard-friendly subset of key backtest metrics."""

    summary = summarize_backtest(decisions)
    return {
        "trades": int(summary["trades"]),
        "roi": float(summary["roi"]),
        "hit_rate": float(summary["hit_rate"]),
        "average_edge": float(summary["average_edge"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "brier_score": float(summary["brier_score"]),
        "total_pnl": float(summary["total_pnl"]),
    }


__all__ = [
    "compact_summary",
    "compute_average_edge",
    "compute_brier_score",
    "compute_hit_rate",
    "compute_max_drawdown",
    "compute_roi",
    "summarize_backtest",
]
