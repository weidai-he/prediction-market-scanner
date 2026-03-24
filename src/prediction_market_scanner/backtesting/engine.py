"""Backtesting scaffold for future strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from prediction_market_scanner.schemas.market import Market


@dataclass
class BacktestResult:
    """Container for future backtest outputs."""

    total_markets_evaluated: int
    notes: list[str] = field(default_factory=list)


class BacktestEngine:
    """Minimal backtesting entry point."""

    def run(self, historical_markets: list[Market]) -> BacktestResult:
        """Run a placeholder backtest over normalized historical markets."""

        return BacktestResult(
            total_markets_evaluated=len(historical_markets),
            notes=["Backtesting logic not implemented yet."],
        )
