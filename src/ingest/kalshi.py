"""Kalshi market ingestion utilities.

This module fetches public Kalshi market data suitable for scanner workflows,
normalizes it into a DataFrame aligned with the Polymarket schema where possible,
and preserves a few Kalshi-specific pricing fields for downstream analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
MARKETS_PATH = "/markets"
DEFAULT_PAGE_SIZE = 200
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_RETRY_STATUSES = (429, 500, 502, 503, 504)
OUTPUT_COLUMNS = [
    "market_id",
    "question",
    "event_title",
    "end_date",
    "category",
    "outcome_prices",
    "implied_prob",
    "active",
    "closed",
    "bid",
    "ask",
    "last_price",
    "market_close_time",
]


class KalshiAPIError(RuntimeError):
    """Raised when Kalshi data cannot be fetched or normalized."""


@dataclass(frozen=True)
class KalshiClientConfig:
    """Configuration for Kalshi ingestion."""

    base_url: str = BASE_URL
    page_size: int = DEFAULT_PAGE_SIZE
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = 3
    backoff_factor: float = 0.5
    status: str = "open"


def build_retry_session(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple[int, ...] = DEFAULT_RETRY_STATUSES,
) -> Session:
    """Create a requests session configured with retry behavior."""

    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Accept": "application/json", "User-Agent": "prediction-market-scanner/0.1"})
    return session


def _coerce_float(value: Any) -> float | None:
    """Safely coerce a value to float."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    """Convert common boolean-like API fields into bool."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "open", "active"}:
            return True
        if normalized in {"false", "0", "no", "closed", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _clip_probability(value: float | None) -> float | None:
    """Clip probability-like values into the [0, 1] interval."""

    if value is None:
        return None
    return min(max(value, 0.0), 1.0)


def _normalize_price(value: Any) -> float | None:
    """Normalize Kalshi prices to a 0-1 probability scale.

    Kalshi commonly returns price fields in cents (0-100). If the value is greater
    than 1, this helper treats it as cents and converts it to probability space.
    """

    numeric = _coerce_float(value)
    if numeric is None:
        return None
    if numeric > 1.0:
        numeric = numeric / 100.0
    return _clip_probability(numeric)


def _extract_category(market: dict[str, Any]) -> str | None:
    """Resolve the best available category for a Kalshi market record."""

    for candidate in (
        market.get("category"),
        market.get("event", {}).get("category") if isinstance(market.get("event"), dict) else None,
        market.get("series_ticker"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _extract_event_title(market: dict[str, Any]) -> str | None:
    """Resolve the best available event title for a market record."""

    event = market.get("event")
    if isinstance(event, dict):
        for candidate in (event.get("title"), event.get("sub_title")):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    for candidate in (market.get("event_title"), market.get("subtitle"), market.get("title")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def derive_market_state(market: dict[str, Any]) -> tuple[bool, bool]:
    """Derive active and closed flags from Kalshi market status fields.

    This helper is intentionally standalone so future WebSocket update handlers can
    reuse the exact same state logic.
    """

    status = str(market.get("status") or "").strip().lower()
    closed = status in {"closed", "settled", "finalized", "expired", "inactive"}
    active = not closed

    if "can_close_early" in market and closed:
        active = False
    if "active" in market:
        active = _coerce_bool(market.get("active"))
        closed = not active if status == "" else closed

    return active, closed


def normalize_market_record(market: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single Kalshi market payload into the scanner schema."""

    bid = _normalize_price(market.get("yes_bid"))
    ask = _normalize_price(market.get("yes_ask"))
    last_price = _normalize_price(market.get("last_price") or market.get("yes_price"))
    implied_prob = last_price if last_price is not None else ask if ask is not None else bid
    active, closed = derive_market_state(market)

    outcome_prices = [price for price in (bid, ask, last_price) if price is not None]

    return {
        "market_id": str(market.get("ticker") or market.get("market_ticker") or ""),
        "question": market.get("title"),
        "event_title": _extract_event_title(market),
        "end_date": market.get("close_time") or market.get("expiration_time"),
        "category": _extract_category(market),
        "outcome_prices": outcome_prices,
        "implied_prob": _clip_probability(implied_prob),
        "active": active,
        "closed": closed,
        "bid": bid,
        "ask": ask,
        "last_price": last_price,
        "market_close_time": market.get("close_time") or market.get("expiration_time"),
    }


def select_normalized_columns(markets: pd.DataFrame) -> pd.DataFrame:
    """Return only the normalized downstream columns in a stable order."""

    normalized = markets.copy()
    for column in OUTPUT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    for column in ("bid", "ask", "last_price", "implied_prob"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").clip(lower=0.0, upper=1.0)

    normalized["active"] = normalized["active"].fillna(False).astype(bool)
    normalized["closed"] = normalized["closed"].fillna(False).astype(bool)

    return normalized.loc[:, OUTPUT_COLUMNS].copy()


def normalize_markets_payload(markets: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize a list of Kalshi markets into a scanner-friendly DataFrame."""

    records: list[dict[str, Any]] = []
    for market in markets:
        if not isinstance(market, dict):
            LOGGER.warning("Skipping market with unexpected payload type: %s", type(market).__name__)
            continue

        record = normalize_market_record(market)
        if record["market_id"] and record["question"]:
            records.append(record)

    frame = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame = frame.drop_duplicates(subset=["market_id"]).reset_index(drop=True)
    return select_normalized_columns(frame)


def _raise_for_bad_response(response: Response) -> None:
    """Raise a domain-specific error for non-success responses."""

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = (
            f"Kalshi request failed with status {response.status_code} "
            f"for {response.request.method} {response.url}"
        )
        raise KalshiAPIError(message) from exc


def fetch_markets_page(
    session: Session,
    *,
    base_url: str,
    limit: int,
    cursor: str | None,
    status: str,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch one cursor-paginated page of Kalshi markets."""

    url = f"{base_url.rstrip('/')}{MARKETS_PATH}"
    params: dict[str, Any] = {
        "limit": limit,
        "status": status,
    }
    if cursor:
        params["cursor"] = cursor

    try:
        response = session.get(url, params=params, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise KalshiAPIError(f"Request to {url} failed") from exc

    _raise_for_bad_response(response)

    try:
        payload = response.json()
    except ValueError as exc:
        raise KalshiAPIError(f"Invalid JSON returned by {response.url}") from exc

    if not isinstance(payload, dict):
        raise KalshiAPIError(
            f"Unexpected response shape from {response.url}: expected dict, got {type(payload).__name__}"
        )

    markets = payload.get("markets")
    if not isinstance(markets, list):
        raise KalshiAPIError(f"Unexpected markets payload from {response.url}")

    next_cursor = payload.get("cursor")
    return [item for item in markets if isinstance(item, dict)], next_cursor if isinstance(next_cursor, str) else None


def fetch_markets(
    config: KalshiClientConfig | None = None,
    session: Session | None = None,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch Kalshi markets with graceful pagination handling."""

    config = config or KalshiClientConfig()
    session = session or build_retry_session(
        max_retries=config.max_retries,
        backoff_factor=config.backoff_factor,
    )

    markets: list[dict[str, Any]] = []
    cursor: str | None = None
    page_count = 0

    while True:
        page, cursor = fetch_markets_page(
            session=session,
            base_url=config.base_url,
            limit=config.page_size,
            cursor=cursor,
            status=config.status,
            timeout_seconds=config.timeout_seconds,
        )
        page_count += 1
        markets.extend(page)
        LOGGER.info("Fetched %s Kalshi markets on page %s", len(page), page_count)

        if not cursor:
            break
        if max_pages is not None and page_count >= max_pages:
            break

    return markets


def fetch_markets_dataframe(
    config: KalshiClientConfig | None = None,
    session: Session | None = None,
    max_pages: int | None = None,
) -> pd.DataFrame:
    """Fetch Kalshi markets and return a normalized DataFrame.

    The function fails gracefully by logging and returning an empty normalized
    DataFrame when the upstream request cannot be completed.
    """

    try:
        markets = fetch_markets(config=config, session=session, max_pages=max_pages)
    except KalshiAPIError:
        LOGGER.exception("Unable to fetch Kalshi markets.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return normalize_markets_payload(markets)


def format_summary(markets: pd.DataFrame) -> str:
    """Build a concise printable summary for manual validation."""

    head_text = markets.head(5).to_string(index=False)
    return "\n".join(
        [
            f"total markets fetched: {len(markets)}",
            f"columns: {list(markets.columns)}",
            "head(5):",
            head_text,
        ]
    )


def main() -> None:
    """Run a simple Kalshi fetch and print a concise summary."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    markets = fetch_markets_dataframe(max_pages=1)
    print(format_summary(markets))


if __name__ == "__main__":
    main()


__all__ = [
    "BASE_URL",
    "OUTPUT_COLUMNS",
    "KalshiAPIError",
    "KalshiClientConfig",
    "build_retry_session",
    "derive_market_state",
    "fetch_markets",
    "fetch_markets_dataframe",
    "fetch_markets_page",
    "format_summary",
    "normalize_market_record",
    "normalize_markets_payload",
    "select_normalized_columns",
]
