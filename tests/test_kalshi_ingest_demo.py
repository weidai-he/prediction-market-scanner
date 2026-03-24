"""Small demonstration script for the Kalshi ingestion module.

Run with:
    python tests/test_kalshi_ingest_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingest.kalshi import fetch_markets_dataframe, format_summary


def main() -> None:
    """Fetch markets and print a small summary."""

    markets = fetch_markets_dataframe(max_pages=1)
    print(format_summary(markets))


if __name__ == "__main__":
    main()
