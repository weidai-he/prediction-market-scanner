"""Shared normalized schema and conversion helpers for market data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pandas as pd


def _is_missing(value: Any) -> bool:
    """Return True when a value should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _coerce_str(value: Any, field_name: str, required: bool = False) -> str | None:
    """Coerce a value to a stripped string."""

    if _is_missing(value):
        if required:
            raise ValueError(f"{field_name} is required")
        return None

    text = str(value).strip()
    if not text:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    return text


def _coerce_probability(value: Any, field_name: str) -> float | None:
    """Coerce and validate a probability-like value."""

    if _is_missing(value) or value == "":
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc

    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return numeric


def _coerce_raw_json(value: Any, fallback: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize raw_json into a dictionary."""

    if _is_missing(value):
        return dict(fallback)
    if isinstance(value, dict):
        return value
    raise ValueError("raw_json must be a dictionary when provided")


def _pick_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    """Return the first present non-missing mapping value for the given keys."""

    for key in keys:
        if key in mapping and not _is_missing(mapping[key]):
            return mapping[key]
    return None


@dataclass(frozen=True)
class NormalizedMarketRecord:
    """Shared normalized market record used across ingestion layers."""

    source_platform: str
    market_id: str
    title: str
    category: str | None
    market_prob: float | None
    bid: float | None
    ask: float | None
    last: float | None
    close_time: str | None
    raw_json: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_platform", _coerce_str(self.source_platform, "source_platform", True))
        object.__setattr__(self, "market_id", _coerce_str(self.market_id, "market_id", True))
        object.__setattr__(self, "title", _coerce_str(self.title, "title", True))
        object.__setattr__(self, "category", _coerce_str(self.category, "category"))
        object.__setattr__(self, "market_prob", _coerce_probability(self.market_prob, "market_prob"))
        object.__setattr__(self, "bid", _coerce_probability(self.bid, "bid"))
        object.__setattr__(self, "ask", _coerce_probability(self.ask, "ask"))
        object.__setattr__(self, "last", _coerce_probability(self.last, "last"))
        object.__setattr__(self, "close_time", _coerce_str(self.close_time, "close_time"))

        if not isinstance(self.raw_json, dict):
            raise ValueError("raw_json must be a dictionary")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        default_source_platform: str | None = None,
    ) -> "NormalizedMarketRecord":
        """Build a normalized record from a dict-like row."""

        fallback_source = default_source_platform or _pick_value(mapping, "source", "platform")
        raw_json = _coerce_raw_json(mapping.get("raw_json"), fallback=dict(mapping))

        return cls(
            source_platform=_pick_value(mapping, "source_platform") or fallback_source,
            market_id=_pick_value(mapping, "market_id"),
            title=_pick_value(mapping, "title", "question"),
            category=_pick_value(mapping, "category"),
            market_prob=_pick_value(mapping, "market_prob", "implied_prob"),
            bid=_pick_value(mapping, "bid"),
            ask=_pick_value(mapping, "ask"),
            last=_pick_value(mapping, "last", "last_price"),
            close_time=_pick_value(mapping, "close_time", "market_close_time", "end_date"),
            raw_json=raw_json,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""

        return asdict(self)


def row_to_market_record(
    row: pd.Series | Mapping[str, Any],
    *,
    default_source_platform: str | None = None,
) -> NormalizedMarketRecord:
    """Convert a pandas row or mapping into a normalized market record."""

    mapping = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return NormalizedMarketRecord.from_mapping(
        mapping,
        default_source_platform=default_source_platform,
    )


def dataframe_to_market_records(
    frame: pd.DataFrame,
    *,
    default_source_platform: str | None = None,
) -> list[NormalizedMarketRecord]:
    """Convert a DataFrame into normalized market record objects."""

    return [
        row_to_market_record(row, default_source_platform=default_source_platform)
        for _, row in frame.iterrows()
    ]


__all__ = [
    "NormalizedMarketRecord",
    "dataframe_to_market_records",
    "row_to_market_record",
]
