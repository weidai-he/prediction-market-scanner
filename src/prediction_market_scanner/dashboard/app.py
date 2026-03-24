"""Streamlit dashboard for the scanner scaffold."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from prediction_market_scanner.ingestion.kalshi import KalshiIngestionClient
from prediction_market_scanner.ingestion.polymarket import PolymarketIngestionClient
from prediction_market_scanner.scanners.low_probability import LowProbabilityScanner


def load_candidate_table() -> pd.DataFrame:
    """Build a small dataframe of current scan candidates."""

    scanner = LowProbabilityScanner(max_probability=0.10)
    clients = [
        PolymarketIngestionClient(),
        KalshiIngestionClient(),
    ]

    candidates = []
    for client in clients:
        candidates.extend(scanner.scan(client.fetch_markets()))

    return pd.DataFrame(
        [
            {
                "source": market.source,
                "title": market.title,
                "probability": market.implied_probability,
                "price": market.price,
                "metadata": str(market.metadata),
            }
            for market in candidates
        ]
    )


def main() -> None:
    """Render the Streamlit application."""

    st.set_page_config(page_title="Prediction Market Scanner", layout="wide")
    st.title("Prediction Market Scanner")
    st.caption("Initial scaffold for low-probability opportunity discovery.")

    table = load_candidate_table()
    st.metric("Candidates Found", len(table))

    if table.empty:
        st.info("No low-probability candidates found in the current mock dataset.")
    else:
        st.dataframe(table, use_container_width=True)


if __name__ == "__main__":
    main()
