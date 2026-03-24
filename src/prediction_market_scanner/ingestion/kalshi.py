"""Kalshi ingestion scaffold."""

from __future__ import annotations

from prediction_market_scanner.ingestion.base import BaseIngestionClient
from prediction_market_scanner.schemas.market import Market


class KalshiIngestionClient(BaseIngestionClient):
    """Minimal Kalshi client returning normalized mock markets."""

    def __init__(self, base_url: str = "https://trading-api.kalshi.com/trade-api/v2") -> None:
        super().__init__(base_url=base_url)

    def fetch_markets(self) -> list[Market]:
        """Fetch market data.

        This scaffold currently returns static records so the project works
        before live API integration is implemented.
        """

        return [
            Market(
                source="kalshi",
                market_id="kalshi-1",
                title="Chicago snowfall exceeds 8 inches this week",
                price=0.08,
                implied_probability=0.08,
                metadata={"category": "weather"},
            ),
            Market(
                source="kalshi",
                market_id="kalshi-2",
                title="Dallas hits 110F this summer",
                price=0.15,
                implied_probability=0.15,
                metadata={"category": "weather"},
            ),
        ]
