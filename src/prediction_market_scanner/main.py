"""CLI entry point for the initial scanner scaffold."""

from __future__ import annotations

from prediction_market_scanner.config import get_settings
from prediction_market_scanner.ingestion.kalshi import KalshiIngestionClient
from prediction_market_scanner.ingestion.polymarket import PolymarketIngestionClient
from prediction_market_scanner.logging_utils import configure_logging, get_logger
from prediction_market_scanner.scanners.low_probability import LowProbabilityScanner


def main() -> None:
    """Run the scaffolded market scan."""

    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    clients = [
        PolymarketIngestionClient(base_url=settings.polymarket_api_url),
        KalshiIngestionClient(base_url=settings.kalshi_api_url),
    ]
    scanner = LowProbabilityScanner(max_probability=0.10)

    results = []
    for client in clients:
        markets = client.fetch_markets()
        results.extend(scanner.scan(markets))

    logger.info("Found %s candidate low-probability opportunities.", len(results))
    for market in results:
        logger.info(
            "[%s] %s | probability=%.3f | price=%.3f",
            market.source,
            market.title,
            market.implied_probability,
            market.price,
        )


if __name__ == "__main__":
    main()
