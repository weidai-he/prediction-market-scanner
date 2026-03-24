"""Polymarket ingestion scaffold."""

from __future__ import annotations

from prediction_market_scanner.ingestion.base import BaseIngestionClient
from prediction_market_scanner.schemas.market import Market


class PolymarketIngestionClient(BaseIngestionClient):
    """Minimal Polymarket client returning normalized mock markets."""

    def __init__(self, base_url: str = "https://gamma-api.polymarket.com") -> None:
        super().__init__(base_url=base_url)

    def fetch_markets(self) -> list[Market]:
        """Fetch market data.

        This scaffold currently returns static records so the project works
        before live API integration is implemented.
        """

        return [
            Market(
                source="polymarket",
                market_id="poly-1",
                title="Category 5 hurricane makes U.S. landfall this month",
                price=0.06,
                implied_probability=0.06,
                metadata={"category": "weather"},
            ),
            Market(
                source="polymarket",
                market_id="poly-2",
                title="Snow in Houston this week",
                price=0.12,
                implied_probability=0.12,
                metadata={"category": "weather"},
            ),
        ]
