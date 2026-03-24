"""Transparent opportunity scoring for low-probability markets.

The goal of this module is to rank scanner candidates with a deliberately simple
and inspectable scoring function. The score favors:

- larger positive edge between model probability and market probability
- enough time remaining before expiration to act on the thesis
- available liquidity signals when present

It also penalizes missing data and stale records so incomplete rows drift lower
in the ranking instead of silently looking attractive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd


MARKET_PROB_ALIASES = ("market_prob", "implied_prob")
CLOSE_TIME_ALIASES = ("close_time", "market_close_time", "end_date")
UPDATED_AT_ALIASES = ("updated_at", "last_updated", "snapshot_time", "timestamp")
LIQUIDITY_ALIASES = ("liquidity", "volume", "open_interest")


@dataclass(frozen=True)
class ScoringWeights:
    """Simple weights for the opportunity score components."""

    edge_weight: float = 100.0
    time_weight: float = 20.0
    liquidity_weight: float = 15.0
    stale_penalty_weight: float = 10.0
    missing_penalty_weight: float = 15.0
    max_time_horizon_days: float = 30.0
    stale_after_hours: float = 24.0


def _get_first_present_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    """Return the first matching column name from a list of aliases."""

    for alias in aliases:
        if alias in frame.columns:
            return alias
    return None


def _coerce_probability_series(series: pd.Series) -> pd.Series:
    """Convert a series into numeric probabilities clipped to [0, 1]."""

    return pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0)


def _coerce_datetime_series(series: pd.Series) -> pd.Series:
    """Convert a series to timezone-aware UTC datetimes when possible."""

    return pd.to_datetime(series, errors="coerce", utc=True)


def _resolve_market_prob_column(markets: pd.DataFrame) -> str:
    """Find the market probability column or raise a helpful error."""

    column = _get_first_present_column(markets, MARKET_PROB_ALIASES)
    if column is None:
        raise ValueError("Expected one of these market probability columns: market_prob, implied_prob")
    return column


def filter_by_market_probability(
    markets: pd.DataFrame,
    *,
    min_prob: float = 0.0,
    max_prob: float = 0.10,
    market_prob_column: str | None = None,
) -> pd.DataFrame:
    """Filter markets to a target market-probability band."""

    if min_prob < 0.0 or max_prob > 1.0 or min_prob > max_prob:
        raise ValueError("Probability filter bounds must satisfy 0 <= min_prob <= max_prob <= 1")

    market_prob_column = market_prob_column or _resolve_market_prob_column(markets)
    filtered = markets.copy()
    filtered[market_prob_column] = _coerce_probability_series(filtered[market_prob_column])
    mask = filtered[market_prob_column].notna()
    mask &= filtered[market_prob_column].between(min_prob, max_prob, inclusive="both")
    return filtered.loc[mask].copy()


def compute_edge(
    markets: pd.DataFrame,
    *,
    model_prob_column: str = "model_prob",
    market_prob_column: str | None = None,
) -> pd.DataFrame:
    """Compute edge as model probability minus market probability."""

    market_prob_column = market_prob_column or _resolve_market_prob_column(markets)
    scored = markets.copy()
    scored[model_prob_column] = _coerce_probability_series(scored[model_prob_column])
    scored[market_prob_column] = _coerce_probability_series(scored[market_prob_column])
    scored["edge"] = scored[model_prob_column] - scored[market_prob_column]
    return scored


def _compute_time_remaining_days(markets: pd.DataFrame, *, now: pd.Timestamp | None = None) -> pd.Series:
    """Return non-negative days remaining until market close when available."""

    close_column = _get_first_present_column(markets, CLOSE_TIME_ALIASES)
    if close_column is None:
        return pd.Series(pd.NA, index=markets.index, dtype="float64")

    now_utc = now if now is not None else pd.Timestamp(datetime.now(UTC))
    close_times = _coerce_datetime_series(markets[close_column])
    days_remaining = (close_times - now_utc).dt.total_seconds() / 86400.0
    return days_remaining.clip(lower=0.0)


def _compute_time_score(
    markets: pd.DataFrame,
    *,
    weights: ScoringWeights,
    now: pd.Timestamp | None = None,
) -> pd.Series:
    """Score remaining time on a 0-1 scale with diminishing returns.

    We cap the benefit at a configurable horizon so extremely distant expiries
    do not dominate the ranking. Markets with a few days or weeks left get
    a moderate boost; expired or missing-close-time rows get little or no boost.
    """

    days_remaining = _compute_time_remaining_days(markets, now=now)
    return (days_remaining / weights.max_time_horizon_days).clip(lower=0.0, upper=1.0).fillna(0.0)


def _compute_liquidity_score(markets: pd.DataFrame) -> pd.Series:
    """Estimate liquidity from whichever signal is available.

    Priority:
    - explicit liquidity / volume / open interest columns when present
    - otherwise tighter bid/ask information acts as a weak liquidity proxy
    """

    liquidity_column = _get_first_present_column(markets, LIQUIDITY_ALIASES)
    if liquidity_column is not None:
        liquidity = pd.to_numeric(markets[liquidity_column], errors="coerce").clip(lower=0.0)
        max_value = liquidity.max(skipna=True)
        if pd.notna(max_value) and max_value and max_value > 0:
            return (liquidity / max_value).fillna(0.0)
        return liquidity.fillna(0.0)

    bid = _coerce_probability_series(markets["bid"]) if "bid" in markets.columns else pd.Series(0.0, index=markets.index)
    ask = _coerce_probability_series(markets["ask"]) if "ask" in markets.columns else pd.Series(0.0, index=markets.index)
    spread = (ask - bid).where(ask.notna() & bid.notna())

    # A smaller spread is usually more tradable, so convert tight spreads into a
    # higher liquidity score. Missing quotes get zero.
    liquidity_score = (1.0 - spread.clip(lower=0.0, upper=1.0)).fillna(0.0)
    liquidity_score[(bid == 0.0) & (ask == 0.0)] = 0.0
    return liquidity_score


def _compute_staleness_penalty(
    markets: pd.DataFrame,
    *,
    weights: ScoringWeights,
    now: pd.Timestamp | None = None,
) -> pd.Series:
    """Compute a 0-1 penalty based on how old the latest update appears."""

    updated_column = _get_first_present_column(markets, UPDATED_AT_ALIASES)
    if updated_column is None:
        return pd.Series(1.0, index=markets.index, dtype="float64")

    now_utc = now if now is not None else pd.Timestamp(datetime.now(UTC))
    updated_at = _coerce_datetime_series(markets[updated_column])
    age_hours = (now_utc - updated_at).dt.total_seconds() / 3600.0
    return (age_hours / weights.stale_after_hours).clip(lower=0.0, upper=1.0).fillna(1.0)


def _compute_missing_data_penalty(markets: pd.DataFrame, *, model_prob_column: str, market_prob_column: str) -> pd.Series:
    """Compute a simple missing-data penalty from important scoring inputs."""

    key_columns = [model_prob_column, market_prob_column]
    for optional_column in ("bid", "ask", "last"):
        if optional_column in markets.columns:
            key_columns.append(optional_column)

    missing_fraction = markets[key_columns].isna().mean(axis=1)
    return missing_fraction.clip(lower=0.0, upper=1.0)


def score_opportunities(
    markets: pd.DataFrame,
    *,
    model_prob_column: str = "model_prob",
    market_prob_column: str | None = None,
    weights: ScoringWeights | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score and rank market opportunities with a transparent additive formula."""

    weights = weights or ScoringWeights()
    market_prob_column = market_prob_column or _resolve_market_prob_column(markets)

    scored = compute_edge(
        markets,
        model_prob_column=model_prob_column,
        market_prob_column=market_prob_column,
    )

    if "last" in scored.columns:
        scored["last"] = _coerce_probability_series(scored["last"])
    elif "last_price" in scored.columns:
        scored["last"] = _coerce_probability_series(scored["last_price"])
    else:
        scored["last"] = pd.NA

    if "bid" in scored.columns:
        scored["bid"] = _coerce_probability_series(scored["bid"])
    else:
        scored["bid"] = pd.NA

    if "ask" in scored.columns:
        scored["ask"] = _coerce_probability_series(scored["ask"])
    else:
        scored["ask"] = pd.NA

    positive_edge = scored["edge"].clip(lower=0.0).fillna(0.0)
    time_score = _compute_time_score(scored, weights=weights, now=now)
    liquidity_score = _compute_liquidity_score(scored)
    stale_penalty = _compute_staleness_penalty(scored, weights=weights, now=now)
    missing_penalty = _compute_missing_data_penalty(
        scored,
        model_prob_column=model_prob_column,
        market_prob_column=market_prob_column,
    )

    # The scoring formula is intentionally linear and inspectable:
    # - positive edge is the main driver
    # - time remaining and liquidity add moderate support
    # - stale or incomplete rows are discounted rather than removed
    scored["time_score"] = time_score
    scored["liquidity_score"] = liquidity_score
    scored["stale_penalty"] = stale_penalty
    scored["missing_penalty"] = missing_penalty
    scored["opportunity_score"] = (
        positive_edge * weights.edge_weight
        + time_score * weights.time_weight
        + liquidity_score * weights.liquidity_weight
        - stale_penalty * weights.stale_penalty_weight
        - missing_penalty * weights.missing_penalty_weight
    )

    return scored.sort_values(
        by=["opportunity_score", "edge", market_prob_column],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def rank_low_probability_opportunities(
    markets: pd.DataFrame,
    *,
    model_prob_column: str = "model_prob",
    market_prob_column: str | None = None,
    min_market_prob: float = 0.0,
    max_market_prob: float = 0.10,
    weights: ScoringWeights | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter to a low-probability band and return ranked opportunities."""

    market_prob_column = market_prob_column or _resolve_market_prob_column(markets)
    filtered = filter_by_market_probability(
        markets,
        min_prob=min_market_prob,
        max_prob=max_market_prob,
        market_prob_column=market_prob_column,
    )
    return score_opportunities(
        filtered,
        model_prob_column=model_prob_column,
        market_prob_column=market_prob_column,
        weights=weights,
        now=now,
    )


__all__ = [
    "ScoringWeights",
    "compute_edge",
    "filter_by_market_probability",
    "rank_low_probability_opportunities",
    "score_opportunities",
]
