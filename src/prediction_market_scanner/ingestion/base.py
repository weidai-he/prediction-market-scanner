"""Base interfaces for ingestion clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

from prediction_market_scanner.schemas.market import Market


class BaseIngestionClient(ABC):
    """Abstract interface for market data ingestion."""

    def __init__(self, base_url: str = "") -> None:
        self.base_url = base_url

    @abstractmethod
    def fetch_markets(self) -> list[Market]:
        """Return normalized market records from a venue."""
