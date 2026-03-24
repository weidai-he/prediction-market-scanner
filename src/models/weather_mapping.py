"""Weather-event probability mapping utilities.

This module converts generic weather forecast features into probabilities for
binary event markets. It is intentionally provider-agnostic: upstream code can
map data from NOAA, Open-Meteo, ECMWF, or any other source into the shared
forecast schema defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Any, Literal


EventType = Literal[
    "temperature_above_threshold",
    "precipitation_above_threshold",
    "snowfall_occurrence",
]


def _clip_probability(value: float) -> float:
    """Clamp a probability into the [0, 1] interval."""

    return min(max(value, 0.0), 1.0)


def _sigmoid(value: float) -> float:
    """Stable logistic transform for smooth threshold-based mappings."""

    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class ForecastFeatures:
    """Generic forecast input schema.

    The fields are intentionally simple so this schema can be produced by any
    forecast provider adapter later.
    """

    location: str
    forecast_time: str
    valid_start: str
    valid_end: str
    max_temperature_c: float | None = None
    expected_precip_mm: float | None = None
    expected_snow_mm: float | None = None
    precip_probability: float | None = None
    snow_probability: float | None = None
    confidence: float | None = None
    provider: str | None = None
    raw_features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("precip_probability", "snow_probability", "confidence"):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")


@dataclass(frozen=True)
class WeatherEventSpec:
    """Binary weather-event market definition."""

    event_type: EventType
    threshold: float | None = None
    unit: str | None = None
    label: str | None = None


class BaseCalibrator:
    """Minimal calibration interface for future post-processing."""

    def calibrate(self, probability: float, *, event_spec: WeatherEventSpec, forecast: ForecastFeatures) -> float:
        """Return a calibrated probability."""

        return _clip_probability(probability)


@dataclass(frozen=True)
class LinearShrinkCalibrator(BaseCalibrator):
    """Simple placeholder calibrator that shrinks toward 50%.

    This keeps the interface explicit without pretending we already have a
    production calibration model trained on historical forecast outcomes.
    """

    shrinkage: float = 0.0

    def calibrate(self, probability: float, *, event_spec: WeatherEventSpec, forecast: ForecastFeatures) -> float:
        base = _clip_probability(probability)
        shrinkage = min(max(self.shrinkage, 0.0), 1.0)
        calibrated = (1.0 - shrinkage) * base + shrinkage * 0.5
        return _clip_probability(calibrated)


def _require_numeric(value: float | None, field_name: str) -> float:
    """Require a numeric forecast field."""

    if value is None:
        raise ValueError(f"{field_name} is required for this event type")
    return float(value)


def map_temperature_above_threshold(forecast: ForecastFeatures, threshold_c: float) -> float:
    """Map a forecast maximum temperature to a threshold-exceedance probability.

    The mapping uses a logistic curve centered at the threshold. Values above
    the threshold quickly move toward high probability, while values below it
    decay smoothly instead of flipping abruptly.
    """

    max_temp = _require_numeric(forecast.max_temperature_c, "max_temperature_c")
    confidence = forecast.confidence if forecast.confidence is not None else 0.7
    slope = 0.9 + confidence
    return _clip_probability(_sigmoid((max_temp - threshold_c) * slope))


def map_precipitation_above_threshold(forecast: ForecastFeatures, threshold_mm: float) -> float:
    """Map forecast precipitation totals to a threshold-exceedance probability."""

    precip_amount = _require_numeric(forecast.expected_precip_mm, "expected_precip_mm")
    precip_chance = forecast.precip_probability if forecast.precip_probability is not None else 0.5

    amount_component = _sigmoid((precip_amount - threshold_mm) / max(threshold_mm * 0.25, 1.0))
    combined = 0.65 * amount_component + 0.35 * precip_chance
    return _clip_probability(combined)


def map_snowfall_occurrence(forecast: ForecastFeatures) -> float:
    """Map snowfall-related features to a binary snow-occurrence probability."""

    snow_chance = forecast.snow_probability if forecast.snow_probability is not None else 0.0
    snow_amount = forecast.expected_snow_mm if forecast.expected_snow_mm is not None else 0.0

    # Any expected snow accumulation should boost probability materially even
    # if the explicit snow probability is unavailable or conservative.
    amount_component = _sigmoid((snow_amount - 0.2) * 4.0)
    combined = 0.7 * snow_chance + 0.3 * amount_component
    return _clip_probability(combined)


def map_event_probability(
    forecast: ForecastFeatures,
    event_spec: WeatherEventSpec,
    *,
    calibrator: BaseCalibrator | None = None,
) -> float:
    """Convert generic forecast features into a binary event probability."""

    calibrator = calibrator or BaseCalibrator()

    if event_spec.event_type == "temperature_above_threshold":
        if event_spec.threshold is None:
            raise ValueError("threshold is required for temperature_above_threshold")
        probability = map_temperature_above_threshold(forecast, threshold_c=event_spec.threshold)
    elif event_spec.event_type == "precipitation_above_threshold":
        if event_spec.threshold is None:
            raise ValueError("threshold is required for precipitation_above_threshold")
        probability = map_precipitation_above_threshold(forecast, threshold_mm=event_spec.threshold)
    elif event_spec.event_type == "snowfall_occurrence":
        probability = map_snowfall_occurrence(forecast)
    else:
        raise ValueError(f"Unsupported event type: {event_spec.event_type}")

    return calibrator.calibrate(probability, event_spec=event_spec, forecast=forecast)


def example_forecasts() -> list[tuple[ForecastFeatures, WeatherEventSpec]]:
    """Return small example inputs for manual experimentation or docs."""

    return [
        (
            ForecastFeatures(
                location="Dallas, TX",
                forecast_time="2026-07-01T12:00:00Z",
                valid_start="2026-07-02T00:00:00Z",
                valid_end="2026-07-02T23:59:59Z",
                max_temperature_c=43.0,
                confidence=0.8,
                provider="generic",
            ),
            WeatherEventSpec(
                event_type="temperature_above_threshold",
                threshold=40.0,
                unit="C",
                label="Temperature above 40C",
            ),
        ),
        (
            ForecastFeatures(
                location="Miami, FL",
                forecast_time="2026-08-10T00:00:00Z",
                valid_start="2026-08-10T00:00:00Z",
                valid_end="2026-08-10T23:59:59Z",
                expected_precip_mm=18.0,
                precip_probability=0.75,
                confidence=0.7,
                provider="generic",
            ),
            WeatherEventSpec(
                event_type="precipitation_above_threshold",
                threshold=10.0,
                unit="mm",
                label="Rainfall above 10mm",
            ),
        ),
        (
            ForecastFeatures(
                location="Chicago, IL",
                forecast_time="2026-12-20T00:00:00Z",
                valid_start="2026-12-20T00:00:00Z",
                valid_end="2026-12-20T23:59:59Z",
                expected_snow_mm=3.0,
                snow_probability=0.65,
                confidence=0.6,
                provider="generic",
            ),
            WeatherEventSpec(
                event_type="snowfall_occurrence",
                label="Snow occurs today",
            ),
        ),
    ]


__all__ = [
    "BaseCalibrator",
    "EventType",
    "ForecastFeatures",
    "LinearShrinkCalibrator",
    "WeatherEventSpec",
    "example_forecasts",
    "map_event_probability",
    "map_precipitation_above_threshold",
    "map_snowfall_occurrence",
    "map_temperature_above_threshold",
]
