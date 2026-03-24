"""Tests for generic weather-event probability mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.weather_mapping import (
    ForecastFeatures,
    LinearShrinkCalibrator,
    WeatherEventSpec,
    example_forecasts,
    map_event_probability,
    map_precipitation_above_threshold,
    map_snowfall_occurrence,
    map_temperature_above_threshold,
)


class WeatherMappingTests(unittest.TestCase):
    """Interface and probability-mapping tests."""

    def test_temperature_above_threshold_increases_with_hotter_forecast(self) -> None:
        cool = ForecastFeatures(
            location="Austin, TX",
            forecast_time="2026-07-01T00:00:00Z",
            valid_start="2026-07-01T00:00:00Z",
            valid_end="2026-07-01T23:59:59Z",
            max_temperature_c=34.0,
            confidence=0.8,
        )
        hot = ForecastFeatures(
            location="Austin, TX",
            forecast_time="2026-07-01T00:00:00Z",
            valid_start="2026-07-01T00:00:00Z",
            valid_end="2026-07-01T23:59:59Z",
            max_temperature_c=41.0,
            confidence=0.8,
        )

        cool_prob = map_temperature_above_threshold(cool, threshold_c=38.0)
        hot_prob = map_temperature_above_threshold(hot, threshold_c=38.0)

        self.assertGreater(hot_prob, cool_prob)
        self.assertTrue(0.0 <= hot_prob <= 1.0)

    def test_precipitation_threshold_uses_amount_and_precip_probability(self) -> None:
        forecast = ForecastFeatures(
            location="Seattle, WA",
            forecast_time="2026-10-01T00:00:00Z",
            valid_start="2026-10-01T00:00:00Z",
            valid_end="2026-10-01T23:59:59Z",
            expected_precip_mm=12.0,
            precip_probability=0.8,
            confidence=0.7,
        )

        probability = map_precipitation_above_threshold(forecast, threshold_mm=5.0)

        self.assertTrue(0.0 <= probability <= 1.0)
        self.assertGreater(probability, 0.5)

    def test_snowfall_occurrence_rises_with_snow_signals(self) -> None:
        low_snow = ForecastFeatures(
            location="Denver, CO",
            forecast_time="2026-11-01T00:00:00Z",
            valid_start="2026-11-01T00:00:00Z",
            valid_end="2026-11-01T23:59:59Z",
            expected_snow_mm=0.0,
            snow_probability=0.1,
        )
        high_snow = ForecastFeatures(
            location="Denver, CO",
            forecast_time="2026-11-01T00:00:00Z",
            valid_start="2026-11-01T00:00:00Z",
            valid_end="2026-11-01T23:59:59Z",
            expected_snow_mm=6.0,
            snow_probability=0.8,
        )

        self.assertGreater(map_snowfall_occurrence(high_snow), map_snowfall_occurrence(low_snow))

    def test_calibrator_can_shrink_probability(self) -> None:
        forecast, event_spec = example_forecasts()[0]

        raw_probability = map_event_probability(forecast, event_spec)
        calibrated_probability = map_event_probability(
            forecast,
            event_spec,
            calibrator=LinearShrinkCalibrator(shrinkage=0.5),
        )

        self.assertTrue(0.0 <= calibrated_probability <= 1.0)
        self.assertLess(abs(calibrated_probability - 0.5), abs(raw_probability - 0.5))

    def test_missing_threshold_raises_for_threshold_events(self) -> None:
        forecast = ForecastFeatures(
            location="Miami, FL",
            forecast_time="2026-08-01T00:00:00Z",
            valid_start="2026-08-01T00:00:00Z",
            valid_end="2026-08-01T23:59:59Z",
            expected_precip_mm=20.0,
            precip_probability=0.9,
        )

        with self.assertRaisesRegex(ValueError, "threshold is required"):
            map_event_probability(
                forecast,
                WeatherEventSpec(event_type="precipitation_above_threshold"),
            )

    def test_missing_required_forecast_field_raises(self) -> None:
        forecast = ForecastFeatures(
            location="Phoenix, AZ",
            forecast_time="2026-06-01T00:00:00Z",
            valid_start="2026-06-01T00:00:00Z",
            valid_end="2026-06-01T23:59:59Z",
            confidence=0.8,
        )

        with self.assertRaisesRegex(ValueError, "max_temperature_c is required"):
            map_temperature_above_threshold(forecast, threshold_c=35.0)

    def test_invalid_confidence_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence must be between 0 and 1"):
            ForecastFeatures(
                location="Boston, MA",
                forecast_time="2026-01-01T00:00:00Z",
                valid_start="2026-01-01T00:00:00Z",
                valid_end="2026-01-01T23:59:59Z",
                confidence=1.5,
            )


if __name__ == "__main__":
    unittest.main()
