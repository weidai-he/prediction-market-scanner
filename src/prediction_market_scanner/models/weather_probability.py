"""Placeholder weather-based probability model."""

from __future__ import annotations

from dataclasses import dataclass

from prediction_market_scanner.schemas.market import Market


@dataclass
class WeatherProbabilityModel:
    """Future interface for estimating weather-event probabilities."""

    model_name: str = "baseline-weather-placeholder"

    def estimate(self, market: Market) -> float:
        """Return a placeholder probability estimate for a weather market."""

        return market.implied_probability
