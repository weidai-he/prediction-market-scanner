"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    polymarket_api_url: str = os.getenv(
        "POLYMARKET_API_URL",
        "https://gamma-api.polymarket.com",
    )
    kalshi_api_url: str = os.getenv(
        "KALSHI_API_URL",
        "https://trading-api.kalshi.com/trade-api/v2",
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_settings() -> Settings:
    """Return immutable application settings."""

    return Settings()
