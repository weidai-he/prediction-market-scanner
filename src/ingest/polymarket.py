"""Polymarket market ingestion utilities.

This module fetches active Polymarket events from the public Gamma API,
flattens nested markets, and normalizes the results into a pandas DataFrame.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)

BASE_URL = "https://gamma-api.polymarket.com"
EVENTS_PATH = "/events"
DEFAULT_PAGE_SIZE = 100
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
]


class PolymarketAPIError(RuntimeError):
    """Raised when Polymarket data cannot be fetched or normalized."""


@dataclass(frozen=True)
class PolymarketClientConfig:
    """Configuration for Polymarket ingestion."""

    base_url: str = BASE_URL
    page_size: int = DEFAULT_PAGE_SIZE
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = 3
    backoff_factor: float = 0.5


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


def _parse_json_like_list(value: Any) -> list[Any]:
    """Parse list-like API fields that may arrive as JSON strings or Python lists."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else []
    return []


def _coerce_float(value: Any) -> float | None:
    """Safely coerce a value to float."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    """Convert common API boolean representations into Python booleans."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _normalize_outcome_prices(value: Any) -> list[float]:
    """Return numeric outcome prices from a market payload."""

    parsed = _parse_json_like_list(value)
    return [price for price in (_coerce_float(item) for item in parsed) if price is not None]


def _extract_implied_probability(outcomes: Any, outcome_prices: Any) -> float | None:
    """Extract the implied probability for the market's primary event outcome.

    If a binary market includes an outcome labeled ``Yes``, this function uses the
    corresponding price as the event probability. Otherwise it falls back to the
    first parsed outcome price.
    """

    labels = [str(item).strip() for item in _parse_json_like_list(outcomes)]
    prices = _normalize_outcome_prices(outcome_prices)

    if not prices:
        return None

    for index, label in enumerate(labels):
        if label.lower() == "yes" and index < len(prices):
            return prices[index]

    return prices[0]


def _extract_category(event: dict[str, Any], market: dict[str, Any]) -> str | None:
    """Resolve the best available category for a market record."""

    for candidate in (market.get("category"), event.get("category")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def normalize_market_record(event: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single Polymarket market nested under an event."""

    outcome_prices = _normalize_outcome_prices(market.get("outcomePrices"))
    implied_prob = _extract_implied_probability(
        outcomes=market.get("outcomes"),
        outcome_prices=market.get("outcomePrices"),
    )

    return {
        "market_id": str(market.get("id") or market.get("conditionId") or ""),
        "question": market.get("question"),
        "event_title": event.get("title"),
        "end_date": market.get("endDate") or event.get("endDate"),
        "category": _extract_category(event=event, market=market),
        "outcome_prices": outcome_prices,
        "implied_prob": implied_prob,
        "active": _coerce_bool(market.get("active", event.get("active"))),
        "closed": _coerce_bool(market.get("closed", event.get("closed"))),
    }


def normalize_events_payload(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten event payloads into a normalized market DataFrame."""

    records: list[dict[str, Any]] = []
    for event in events:
        markets = event.get("markets") or []
        if not isinstance(markets, list):
            LOGGER.warning("Skipping event with unexpected markets payload type: %s", type(markets).__name__)
            continue

        for market in markets:
            if not isinstance(market, dict):
                LOGGER.warning("Skipping market with unexpected payload type: %s", type(market).__name__)
                continue
            record = normalize_market_record(event=event, market=market)
            if record["market_id"] and record["question"]:
                records.append(record)

    frame = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame = frame.drop_duplicates(subset=["market_id"]).reset_index(drop=True)
    frame["implied_prob"] = pd.to_numeric(frame["implied_prob"], errors="coerce")
    frame["active"] = frame["active"].astype(bool)
    frame["closed"] = frame["closed"].astype(bool)
    return frame[OUTPUT_COLUMNS]


def _raise_for_bad_response(response: Response) -> None:
    """Raise a domain-specific error for non-success responses."""

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = (
            f"Polymarket request failed with status {response.status_code} "
            f"for {response.request.method} {response.url}"
        )
        raise PolymarketAPIError(message) from exc


def fetch_active_events_page(
    session: Session,
    *,
    base_url: str,
    limit: int,
    offset: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Fetch a single page of active, open events from Polymarket."""

    url = f"{base_url.rstrip('/')}{EVENTS_PATH}"
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "offset": offset,
    }

    try:
        response = session.get(url, params=params, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise PolymarketAPIError(f"Request to {url} failed") from exc

    _raise_for_bad_response(response)

    try:
        payload = response.json()
    except ValueError as exc:
        raise PolymarketAPIError(f"Invalid JSON returned by {response.url}") from exc

    if not isinstance(payload, list):
        raise PolymarketAPIError(
            f"Unexpected response shape from {response.url}: expected list, got {type(payload).__name__}"
        )

    return [item for item in payload if isinstance(item, dict)]


def fetch_active_events(
    config: PolymarketClientConfig | None = None,
    session: Session | None = None,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch all paginated active events from Polymarket."""

    config = config or PolymarketClientConfig()
    session = session or build_retry_session(
        max_retries=config.max_retries,
        backoff_factor=config.backoff_factor,
    )

    events: list[dict[str, Any]] = []
    offset = 0
    page_count = 0

    while True:
        page = fetch_active_events_page(
            session=session,
            base_url=config.base_url,
            limit=config.page_size,
            offset=offset,
            timeout_seconds=config.timeout_seconds,
        )
        events.extend(page)
        page_count += 1

        LOGGER.info("Fetched %s events from offset=%s", len(page), offset)

        if len(page) < config.page_size:
            break
        if max_pages is not None and page_count >= max_pages:
            break

        offset += config.page_size

    return events


def fetch_active_markets_dataframe(
    config: PolymarketClientConfig | None = None,
    session: Session | None = None,
    max_pages: int | None = None,
) -> pd.DataFrame:
    """Fetch active Polymarket markets and return a normalized DataFrame."""

    events = fetch_active_events(config=config, session=session, max_pages=max_pages)
    return normalize_events_payload(events)


def filter_low_probability_markets(
    markets: pd.DataFrame,
    max_implied_prob: float = 0.10,
    include_closed: bool = False,
) -> pd.DataFrame:
    """Filter a normalized Polymarket DataFrame for low-probability markets."""

    required_columns = {"implied_prob", "closed"}
    missing = required_columns.difference(markets.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame is missing required columns: {missing_csv}")

    mask = markets["implied_prob"].notna() & (markets["implied_prob"] <= max_implied_prob)
    if not include_closed:
        mask &= ~markets["closed"].astype(bool)

    filtered = markets.loc[mask].copy()
    return filtered.sort_values(by="implied_prob", ascending=True).reset_index(drop=True)


__all__ = [
    "BASE_URL",
    "OUTPUT_COLUMNS",
    "PolymarketAPIError",
    "PolymarketClientConfig",
    "build_retry_session",
    "fetch_active_events",
    "fetch_active_events_page",
    "fetch_active_markets_dataframe",
    "filter_low_probability_markets",
    "normalize_events_payload",
    "normalize_market_record",
]
