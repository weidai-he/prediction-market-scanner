"""Tests for market universe filtering and subtype classification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.market_universe import (
    build_market_explanation,
    classify_market_subtype,
    filter_market_universe,
    is_weather_market,
)


class MarketUniverseTests(unittest.TestCase):
    """Simple rule-based checks for universe selection."""

    def test_classify_weather_snow_market(self) -> None:
        row = pd.Series({"title": "Will Chicago snowfall exceed 2 inches?", "category": "weather"})
        self.assertEqual(classify_market_subtype(row), "snow")

    def test_filter_weather_universe(self) -> None:
        frame = pd.DataFrame(
            [
                {"title": "Will Miami rainfall exceed 1 inch?", "category": "weather"},
                {"title": "Will Team A win the championship?", "category": "sports"},
            ]
        )
        filtered = filter_market_universe(frame, universe="Weather")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["market_subtype"], "rain")

    def test_weather_filter_excludes_geopolitical_false_positive(self) -> None:
        row = pd.Series(
            {
                "title": "Will Russia stop attacks after winter storm in Ukraine?",
                "question": "Will a ceasefire follow?",
                "category": "weather",
            }
        )
        self.assertFalse(is_weather_market(row))

    def test_weather_filter_requires_title_or_question(self) -> None:
        row = pd.Series({"title": None, "question": None, "category": "weather"})
        self.assertFalse(is_weather_market(row))

    def test_weather_explanation_mentions_weather_context(self) -> None:
        row = pd.Series(
            {
                "title": "Will Houston hit 100F?",
                "category": "weather",
                "market_prob": 0.08,
                "model_prob": 0.16,
                "edge": 0.08,
                "market_subtype": "temperature",
            }
        )
        explanation = build_market_explanation(row)
        self.assertIn("temperature-threshold event", explanation)
        self.assertIn("8.0%", explanation)


if __name__ == "__main__":
    unittest.main()
