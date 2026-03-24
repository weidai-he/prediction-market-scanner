"""Normalized market schema used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Market:
    """Normalized representation of a prediction market contract."""

    source: str
    market_id: str
    title: str
    price: float
    implied_probability: float
    metadata: dict[str, Any] = field(default_factory=dict)
