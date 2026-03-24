"""Scanner for low-probability market opportunities."""

from __future__ import annotations

from dataclasses import dataclass

from prediction_market_scanner.schemas.market import Market


@dataclass
class LowProbabilityScanner:
    """Select markets below a target implied probability threshold."""

    max_probability: float = 0.10

    def scan(self, markets: list[Market]) -> list[Market]:
        """Return markets whose implied probability is at or below the threshold."""

        return [
            market
            for market in markets
            if market.implied_probability <= self.max_probability
        ]
