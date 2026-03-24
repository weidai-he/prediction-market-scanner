"""Small demonstration script for the Polymarket ingestion module.

Run with:
    set PYTHONPATH=src
    python tests/test_polymarket_ingest_demo.py
"""

from __future__ import annotations

from ingest.polymarket import fetch_active_markets_dataframe, filter_low_probability_markets


def main() -> None:
    """Fetch active markets and print a small low-probability sample."""

    markets = fetch_active_markets_dataframe(max_pages=1)
    low_prob = filter_low_probability_markets(markets, max_implied_prob=0.10)

    print("Fetched rows:", len(markets))
    print("Low-probability rows:", len(low_prob))
    print(low_prob.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
