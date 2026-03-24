"""Streamlit dashboard for the prediction market scanner MVP.

Run locally from the project root with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest.metrics import compact_summary
from backtest.runner import BacktestConfig, get_equity_curve, run_backtest
from ingest.kalshi import fetch_markets_dataframe as fetch_kalshi_markets
from ingest.polymarket import fetch_active_markets_dataframe as fetch_polymarket_markets
from models.edge_score import rank_low_probability_opportunities
from models.market_universe import (
    UNIVERSE_OPTIONS,
    build_market_explanation,
    classify_market_subtype,
    filter_market_universe,
)


st.set_page_config(
    page_title="Prediction Market Scanner",
    page_icon=":bar_chart:",
    layout="wide",
)


def build_synthetic_markets() -> pd.DataFrame:
    """Create a synthetic dataset so the dashboard always has demo content."""

    rows: list[dict[str, object]] = []
    base_date = pd.Timestamp("2026-01-01", tz="UTC")
    categories = ["weather", "climate", "temperature"]
    platforms = ["synthetic-polymarket", "synthetic-kalshi"]

    for day_offset in range(14):
        decision_date = base_date + pd.Timedelta(days=day_offset)
        for market_number in range(8):
            market_prob = 0.01 + (0.0125 * market_number)
            model_prob = min(market_prob + 0.015 + 0.005 * (market_number % 4), 0.95)
            resolved_yes = 1 if (day_offset + market_number) % 5 in {0, 1} else 0
            category = categories[market_number % len(categories)]
            platform = platforms[market_number % len(platforms)]

            rows.append(
                {
                    "date": decision_date.isoformat(),
                    "market_id": f"synthetic-{day_offset}-{market_number}",
                    "source_platform": platform,
                    "title": f"Synthetic {category} market {day_offset}-{market_number}",
                    "category": category,
                    "market_prob": round(market_prob, 4),
                    "model_prob": round(model_prob, 4),
                    "bid": round(max(market_prob - 0.01, 0.01), 4),
                    "ask": round(min(market_prob + 0.01, 0.99), 4),
                    "last": round(market_prob, 4),
                    "close_time": (decision_date + pd.Timedelta(days=5 + market_number)).isoformat(),
                    "resolved_yes": resolved_yes,
                    "liquidity": 500 + 250 * market_number,
                    "event_title": f"{category.title()} event cluster",
                }
            )

    return pd.DataFrame(rows)


def ensure_opportunity_columns(markets: pd.DataFrame) -> pd.DataFrame:
    """Normalize fields needed by the opportunity table and filters."""

    normalized = markets.copy()

    if "source_platform" not in normalized.columns:
        if "source" in normalized.columns:
            normalized["source_platform"] = normalized["source"]
        else:
            normalized["source_platform"] = "unknown"

    if "title" not in normalized.columns:
        if "question" in normalized.columns:
            normalized["title"] = normalized["question"]
        elif "event_title" in normalized.columns:
            normalized["title"] = normalized["event_title"]
        else:
            normalized["title"] = (
                normalized["market_id"].astype(str)
                if "market_id" in normalized.columns
                else pd.Series("Untitled market", index=normalized.index)
            )
    else:
        question_fallback = normalized["question"] if "question" in normalized.columns else pd.Series(pd.NA, index=normalized.index)
        normalized["title"] = normalized["title"].fillna(question_fallback)
        if "event_title" in normalized.columns:
            normalized["title"] = normalized["title"].fillna(normalized["event_title"])
        market_id_fallback = (
            normalized["market_id"].astype(str)
            if "market_id" in normalized.columns
            else pd.Series("Untitled market", index=normalized.index)
        )
        normalized["title"] = normalized["title"].fillna(market_id_fallback)

    if "category" not in normalized.columns:
        normalized["category"] = "uncategorized"
    else:
        normalized["category"] = normalized["category"].fillna("uncategorized")

    if "market_prob" not in normalized.columns:
        if "implied_prob" in normalized.columns:
            normalized["market_prob"] = pd.to_numeric(normalized["implied_prob"], errors="coerce")
        else:
            normalized["market_prob"] = pd.NA

    if "close_time" not in normalized.columns:
        if "market_close_time" in normalized.columns:
            normalized["close_time"] = normalized["market_close_time"]
        elif "end_date" in normalized.columns:
            normalized["close_time"] = normalized["end_date"]
        else:
            normalized["close_time"] = pd.NaT
    else:
        if "market_close_time" in normalized.columns:
            normalized["close_time"] = normalized["close_time"].fillna(normalized["market_close_time"])
        if "end_date" in normalized.columns:
            normalized["close_time"] = normalized["close_time"].fillna(normalized["end_date"])

    for column in ("market_prob", "bid", "ask", "last", "liquidity"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        else:
            normalized[column] = pd.NA

    normalized["market_prob"] = normalized["market_prob"].where(normalized["market_prob"].between(0.0, 1.0), pd.NA)
    normalized = normalized.loc[normalized["market_prob"].notna() & normalized["market_prob"].gt(0.0)].copy()

    normalized["close_time"] = pd.to_datetime(normalized["close_time"], errors="coerce", utc=True)
    normalized["time_to_expiry_days"] = (
        (normalized["close_time"] - pd.Timestamp.now(tz="UTC")).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)
    normalized = normalized.loc[
        normalized["close_time"].notna() & normalized["title"].notna() & normalized["category"].notna()
    ].copy()

    if "model_prob" not in normalized.columns:
        # Demo-only model estimate so the dashboard can rank live markets before
        # the dedicated forecasting model is fully wired in.
        demo_uplift = (0.12 - normalized["market_prob"].fillna(0.0)).clip(lower=0.0) * 0.25
        liquidity_bonus = normalized["liquidity"].fillna(0.0)
        if liquidity_bonus.max(skipna=True) and liquidity_bonus.max(skipna=True) > 0:
            liquidity_bonus = liquidity_bonus / liquidity_bonus.max(skipna=True) * 0.02
        else:
            liquidity_bonus = 0.0
        normalized["model_prob"] = (normalized["market_prob"].fillna(0.0) + demo_uplift + liquidity_bonus).clip(
            lower=0.0,
            upper=1.0,
        )

    return normalized


@st.cache_data(ttl=900, show_spinner=False)
def load_live_markets() -> tuple[pd.DataFrame, list[str]]:
    """Load live markets from available ingestors with graceful fallbacks."""

    frames: list[pd.DataFrame] = []
    warnings: list[str] = []

    try:
        polymarket = fetch_polymarket_markets(max_pages=1)
        if not polymarket.empty:
            polymarket = polymarket.copy()
            polymarket["source_platform"] = "polymarket"
            frames.append(polymarket)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Polymarket unavailable: {exc}")

    try:
        kalshi = fetch_kalshi_markets(max_pages=1)
        if not kalshi.empty:
            kalshi = kalshi.copy()
            kalshi["source_platform"] = "kalshi"
            if "title" not in kalshi.columns and "question" in kalshi.columns:
                kalshi["title"] = kalshi["question"]
            frames.append(kalshi)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Kalshi unavailable: {exc}")

    if not frames:
        return pd.DataFrame(), warnings

    live_markets = pd.concat(frames, ignore_index=True, sort=False)
    return ensure_opportunity_columns(live_markets), warnings


@st.cache_data(show_spinner=False)
def load_synthetic_backtest() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Build synthetic data and run the backtest for demo-safe dashboards."""

    markets = build_synthetic_markets()
    output_path = PROJECT_ROOT / "artifacts" / "backtest" / "synthetic_dashboard_decisions.csv"
    decisions, metrics = run_backtest(
        markets,
        BacktestConfig(
            max_market_prob=0.10,
            min_edge=0.02,
            fixed_stake=100.0,
            decisions_csv_path=str(output_path),
        ),
    )
    equity_curve = get_equity_curve(decisions)
    return decisions, equity_curve, metrics["compact_summary"]


def apply_market_filters(markets: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to the opportunity universe."""

    filtered = markets.copy()

    st.sidebar.header("Filters")

    platforms = sorted(platform for platform in filtered["source_platform"].dropna().unique())
    selected_platforms = st.sidebar.multiselect(
        "Source platform",
        options=platforms,
        default=platforms,
    )
    if selected_platforms:
        filtered = filtered.loc[filtered["source_platform"].isin(selected_platforms)]

    min_prob = float(filtered["market_prob"].fillna(0.0).min()) if not filtered.empty else 0.0
    max_prob = float(filtered["market_prob"].fillna(0.0).max()) if not filtered.empty else 1.0
    prob_range = st.sidebar.slider(
        "Market probability range",
        min_value=0.0,
        max_value=1.0,
        value=(max(0.0, min_prob), min(1.0, max_prob if max_prob > 0 else 0.25)),
        step=0.01,
    )
    filtered = filtered.loc[filtered["market_prob"].between(prob_range[0], prob_range[1], inclusive="both")]

    max_expiry_days = int(min(365, max(1, round(filtered["time_to_expiry_days"].fillna(0).max())))) if not filtered.empty else 30
    expiry_days = st.sidebar.slider(
        "Max time to expiry (days)",
        min_value=0,
        max_value=max(30, max_expiry_days),
        value=max(7, min(30, max_expiry_days)),
        step=1,
    )
    filtered = filtered.loc[filtered["time_to_expiry_days"].fillna(0).le(expiry_days)]

    return filtered


def render_summary_cards(summary: dict[str, float | int]) -> None:
    """Render compact backtest metric cards."""

    card_columns = st.columns(5)
    card_columns[0].metric("Trades", int(summary.get("trades", 0)))
    card_columns[1].metric("ROI", f"{summary.get('roi', 0.0):.1%}")
    card_columns[2].metric("Hit Rate", f"{summary.get('hit_rate', 0.0):.1%}")
    card_columns[3].metric("Avg Edge", f"{summary.get('average_edge', 0.0):.3f}")
    card_columns[4].metric("Max Drawdown", f"{summary.get('max_drawdown', 0.0):.2f}")
    st.caption(f"Brier score: {summary.get('brier_score', 0.0):.4f} | Total PnL: {summary.get('total_pnl', 0.0):.2f}")


def _format_probability(value: object) -> str:
    """Format a probability-like value as a percentage string."""

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "N/A"
    return f"{float(numeric):.1%}"


def build_recommendation(row: pd.Series) -> str:
    """Return a simple recommendation based on model and market probabilities."""

    market_prob = pd.to_numeric(pd.Series([row.get("market_prob")]), errors="coerce").iloc[0]
    model_prob = pd.to_numeric(pd.Series([row.get("model_prob")]), errors="coerce").iloc[0]

    if pd.isna(market_prob) or pd.isna(model_prob):
        return "Avoid until pricing confidence improves"
    if model_prob > market_prob:
        return f"Buy YES under {market_prob:.1%}"
    return "Avoid"


def render_top_opportunity_card(opportunities: pd.DataFrame) -> None:
    """Render a visually distinct top-opportunity card."""

    st.subheader("Top Opportunity Today")

    if opportunities.empty:
        st.info("No ranked opportunity is available right now. Adjust filters or fall back to synthetic demo data.")
        return

    top_row = opportunities.iloc[0]
    title = top_row.get("title") or top_row.get("market_id") or "Untitled market"

    with st.container(border=True):
        st.markdown(f"### {title}")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Market Prob", _format_probability(top_row.get("market_prob")))
        metric_columns[1].metric("Model Prob", _format_probability(top_row.get("model_prob")))
        metric_columns[2].metric("Edge", _format_probability(top_row.get("edge")))
        metric_columns[3].metric("Subtype", str(top_row.get("market_subtype") or "other"))

        st.caption(f"Platform: {str(top_row.get('source_platform') or 'unknown')}")

        st.markdown("**Why mispriced**")
        st.write(build_market_explanation(top_row))

        st.markdown("**Recommendation**")
        st.write(build_recommendation(top_row))


def render_opportunity_chart(opportunities: pd.DataFrame) -> None:
    """Render a scatter chart comparing market and model probabilities."""

    chart_data = opportunities.copy()
    if chart_data.empty:
        st.info("No opportunity points available for the current filters.")
        return

    chart_frame = chart_data.loc[:, ["title", "source_platform", "market_prob", "model_prob", "opportunity_score"]].copy()
    st.scatter_chart(
        chart_frame,
        x="market_prob",
        y="model_prob",
        size="opportunity_score",
        color="source_platform",
    )


def render_equity_curve(equity_curve: pd.DataFrame) -> None:
    """Render the daily cumulative PnL chart."""

    if equity_curve.empty:
        st.info("No equity curve available yet. Run a backtest or use the synthetic demo data.")
        return

    curve = equity_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    curve = curve.set_index("date")
    st.line_chart(curve[["cumulative_pnl"]])


def main() -> None:
    """Render the Streamlit dashboard."""

    st.title("Prediction Market Scanner")
    st.caption("Low-probability opportunity scanner with live-ingestion fallback and backtest demo support.")

    live_markets, live_warnings = load_live_markets()
    synthetic_decisions, synthetic_equity_curve, synthetic_summary = load_synthetic_backtest()

    if live_warnings:
        st.warning("Live feeds are partially unavailable. The dashboard is falling back to synthetic demo data where needed.")
        with st.expander("Feed status details"):
            for message in live_warnings:
                st.write(f"- {message}")

    if live_markets.empty:
        st.info(
            "No live market data is currently available. Showing a dashboard-ready synthetic dataset so the product demo remains usable."
        )
        opportunity_universe = ensure_opportunity_columns(build_synthetic_markets())
    else:
        opportunity_universe = live_markets

    st.sidebar.header("Universe")
    selected_universe = st.sidebar.selectbox("Market Universe", options=UNIVERSE_OPTIONS, index=1)
    show_filter_debug = st.sidebar.checkbox("Show filter debug", value=False)

    before_universe_count = len(opportunity_universe)
    universe_constrained = filter_market_universe(opportunity_universe, universe=selected_universe)
    after_universe_count = len(universe_constrained)
    filtered_universe = apply_market_filters(universe_constrained)

    if filtered_universe.empty:
        ranked_opportunities = filtered_universe.copy()
    else:
        ranked_opportunities = rank_low_probability_opportunities(
            filtered_universe,
            model_prob_column="model_prob",
            market_prob_column="market_prob",
            min_market_prob=0.0,
            max_market_prob=1.0,
        )
        ranked_opportunities["market_subtype"] = ranked_opportunities.apply(classify_market_subtype, axis=1)
        ranked_opportunities["explanation"] = ranked_opportunities.apply(build_market_explanation, axis=1)

    top_opportunities = ranked_opportunities.head(25).copy()
    if not top_opportunities.empty:
        top_opportunities["close_time"] = pd.to_datetime(top_opportunities["close_time"], errors="coerce", utc=True)
        top_opportunities["close_time"] = top_opportunities["close_time"].dt.strftime("%Y-%m-%d")

    render_top_opportunity_card(ranked_opportunities)

    st.subheader("Top Opportunities")
    if top_opportunities.empty:
        st.info(f"No {selected_universe.lower()} markets match the current filters.")
    else:
        st.dataframe(
            top_opportunities[
                [
                    "source_platform",
                    "market_subtype",
                    "market_id",
                    "title",
                    "market_prob",
                    "model_prob",
                    "edge",
                    "opportunity_score",
                    "explanation",
                    "close_time",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if show_filter_debug:
        st.subheader("Filter Debug")
        debug_columns = st.columns(2)
        debug_columns[0].metric("Rows Before Universe Filter", before_universe_count)
        debug_columns[1].metric("Rows After Universe Filter", after_universe_count)
        sample_titles = universe_constrained.get("title", pd.Series(dtype=str)).dropna().astype(str).head(10)
        if sample_titles.empty:
            st.caption("No matched titles available for the current universe.")
        else:
            st.caption("Sample matched titles")
            st.write(sample_titles.tolist())

    chart_col, summary_col = st.columns([1.4, 1.0])
    with chart_col:
        st.subheader("Market vs Model Probability")
        render_opportunity_chart(top_opportunities)
    with summary_col:
        st.subheader("Backtest Summary")
        render_summary_cards(synthetic_summary)

    st.subheader("Equity Curve")
    render_equity_curve(synthetic_equity_curve)

    st.subheader("Daily Prediction / Backtest Log")
    if synthetic_decisions.empty:
        st.info("No backtest decision log is available yet.")
    else:
        log_columns = [
            "decision_date",
            "market_id",
            "source_platform",
            "market_prob",
            "model_prob",
            "edge",
            "stake",
            "outcome",
            "pnl",
        ]
        log_frame = synthetic_decisions.copy()
        log_frame["decision_date"] = pd.to_datetime(log_frame["decision_date"], errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%d"
        )
        st.dataframe(log_frame[log_columns], use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Run locally**

        ```bash
        streamlit run app/app.py
        ```
        """
    )


if __name__ == "__main__":
    main()
