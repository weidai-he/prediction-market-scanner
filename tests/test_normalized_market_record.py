"""Unit tests for the shared normalized market schema."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingest.schema import NormalizedMarketRecord, dataframe_to_market_records, row_to_market_record


class NormalizedMarketRecordTests(unittest.TestCase):
    """Validation and conversion tests for normalized market records."""

    def test_from_polymarket_style_mapping(self) -> None:
        row = {
            "market_id": "poly-123",
            "question": "Will it snow in Austin?",
            "category": "weather",
            "implied_prob": 0.08,
            "end_date": "2026-12-01T00:00:00Z",
            "raw_json": {"id": "poly-123"},
        }

        record = row_to_market_record(row, default_source_platform="polymarket")

        self.assertEqual(record.source_platform, "polymarket")
        self.assertEqual(record.title, "Will it snow in Austin?")
        self.assertEqual(record.market_prob, 0.08)
        self.assertEqual(record.close_time, "2026-12-01T00:00:00Z")

    def test_from_kalshi_style_mapping(self) -> None:
        row = pd.Series(
            {
                "market_id": "KXHIGHTEMP-001",
                "title": "Will Dallas hit 110F?",
                "category": "weather",
                "market_prob": 0.17,
                "bid": 0.15,
                "ask": 0.18,
                "last_price": 0.16,
                "market_close_time": "2026-07-01T00:00:00Z",
                "raw_json": {"ticker": "KXHIGHTEMP-001"},
            }
        )

        record = row_to_market_record(row, default_source_platform="kalshi")

        self.assertEqual(record.source_platform, "kalshi")
        self.assertEqual(record.last, 0.16)
        self.assertEqual(record.ask, 0.18)
        self.assertEqual(record.close_time, "2026-07-01T00:00:00Z")

    def test_dataframe_conversion(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "question": "Test market 1",
                    "implied_prob": 0.1,
                    "raw_json": {"id": "m1"},
                },
                {
                    "market_id": "m2",
                    "question": "Test market 2",
                    "implied_prob": 0.2,
                    "raw_json": {"id": "m2"},
                },
            ]
        )

        records = dataframe_to_market_records(frame, default_source_platform="polymarket")

        self.assertEqual(len(records), 2)
        self.assertTrue(all(isinstance(record, NormalizedMarketRecord) for record in records))

    def test_probability_above_one_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "market_prob must be between 0 and 1"):
            NormalizedMarketRecord(
                source_platform="polymarket",
                market_id="bad-1",
                title="Bad market",
                category="weather",
                market_prob=1.5,
                bid=None,
                ask=None,
                last=None,
                close_time=None,
                raw_json={},
            )

    def test_negative_bid_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "bid must be between 0 and 1"):
            NormalizedMarketRecord(
                source_platform="kalshi",
                market_id="bad-2",
                title="Bad bid market",
                category=None,
                market_prob=0.3,
                bid=-0.1,
                ask=0.2,
                last=0.15,
                close_time=None,
                raw_json={},
            )

    def test_non_numeric_probability_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "market_prob must be numeric"):
            row_to_market_record(
                {
                    "market_id": "bad-3",
                    "question": "Malformed probability",
                    "implied_prob": "not-a-number",
                    "raw_json": {"id": "bad-3"},
                },
                default_source_platform="polymarket",
            )

    def test_missing_required_fields_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "market_id is required"):
            row_to_market_record(
                {
                    "question": "Missing market id",
                    "implied_prob": 0.2,
                    "raw_json": {},
                },
                default_source_platform="polymarket",
            )

    def test_non_dict_raw_json_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw_json must be a dictionary"):
            row_to_market_record(
                {
                    "market_id": "bad-4",
                    "question": "Bad raw_json",
                    "implied_prob": 0.2,
                    "raw_json": "oops",
                },
                default_source_platform="polymarket",
            )


if __name__ == "__main__":
    unittest.main()
