"""Small demonstration script for the Polymarket ingestion module.

Run with:
    python tests/test_polymarket_ingest_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingest.polymarket import fetch_active_markets_dataframe, filter_low_probability_markets, format_summary


def main() -> None:
    """Fetch active markets and print a small low-probability sample."""

    markets = fetch_active_markets_dataframe(max_pages=1)
    low_prob = filter_low_probability_markets(markets, max_implied_prob=0.10)
    print(format_summary(markets, low_prob))


if __name__ == "__main__":
    main()
