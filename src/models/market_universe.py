"""Market-universe filtering, subtype classification, and rule-based explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd


WEATHER_POSITIVE_KEYWORDS = [
    "weather",
    "forecast",
    "temperature",
    "high temperature",
    "low temperature",
    "heat",
    "cold",
    "snow",
    "snowfall",
    "rain",
    "rainfall",
    "precipitation",
    "hurricane",
    "tornado",
    "blizzard",
    "thunderstorm",
    "wind speed",
    "freeze",
    "frost",
    "storm",
    "storm surge",
]

WEATHER_NEGATIVE_KEYWORDS = [
    "ukraine",
    "russia",
    "war",
    "ceasefire",
    "nato",
    "missile",
    "border",
    "putin",
    "zelensky",
    "troops",
    "invasion",
    "military",
    "sanctions",
]

WEATHER_SUBTYPE_KEYWORDS = {
    "temperature": ["temperature", "high temperature", "low temperature", "heat", "cold", "freeze", "frost"],
    "snow": ["snow", "snowfall", "blizzard"],
    "rain": ["rain", "rainfall", "precipitation"],
    "storm": ["storm", "storm surge", "hurricane", "tornado", "thunderstorm", "wind speed"],
}

POLITICS_KEYWORDS = [
    "election",
    "senate",
    "president",
    "congress",
    "vote",
    "party",
    "governor",
    "parliament",
    "campaign",
]

ECONOMICS_KEYWORDS = [
    "inflation",
    "cpi",
    "gdp",
    "recession",
    "unemployment",
    "rate",
    "fed",
    "ecb",
    "payrolls",
    "yield",
]

SPORTS_KEYWORDS = [
    "tournament",
    "championship",
    "playoffs",
    "cup",
    "masters",
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "golf",
    "tennis",
    "match",
    "win",
]

UNIVERSE_OPTIONS = ["All", "Weather", "Politics", "Economics", "Sports"]


def _normalize_text(value: Any) -> str:
    """Return lowercase text for keyword matching."""

    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _title_question_text(row: pd.Series) -> str:
    """Combine title/question text for stricter semantic filtering."""

    parts = [
        _normalize_text(row.get("title")),
        _normalize_text(row.get("question")),
    ]
    return " | ".join(part for part in parts if part)


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Return True if any keyword appears in the provided text."""

    return any(keyword.lower() in text for keyword in keywords)


def matches_positive_keywords(text: str, keywords: list[str]) -> bool:
    """Return True when text matches at least one positive keyword."""

    return bool(text) and _contains_any(text, keywords)


def matches_exclusion_keywords(text: str, keywords: list[str]) -> bool:
    """Return True when text matches any exclusion keyword."""

    return bool(text) and _contains_any(text, keywords)


def is_weather_market(row: pd.Series) -> bool:
    """Return True only for confidently weather-related rows.

    Weather mode uses title/question semantics only. Rows missing both title and
    question are excluded. A row must match at least one positive weather phrase
    and none of the negative geopolitics exclusion phrases.
    """

    text = _title_question_text(row)
    if not text:
        return False
    if matches_exclusion_keywords(text, WEATHER_NEGATIVE_KEYWORDS):
        return False
    return matches_positive_keywords(text, WEATHER_POSITIVE_KEYWORDS)


def _matches_universe_text(row: pd.Series, keywords: list[str]) -> bool:
    """Return True if title/question text matches the provided universe rules."""

    text = _title_question_text(row)
    return matches_positive_keywords(text, keywords)


def classify_market_subtype(row: pd.Series) -> str:
    """Return a simple rule-based subtype for a market.

    Weather subtypes depend only on title/question semantics.
    """

    text = _title_question_text(row)
    if not text:
        return "other"

    if is_weather_market(row):
        for subtype, keywords in WEATHER_SUBTYPE_KEYWORDS.items():
            if _contains_any(text, keywords):
                return subtype
        return "general_weather"

    if _matches_universe_text(row, POLITICS_KEYWORDS):
        return "politics"
    if _matches_universe_text(row, ECONOMICS_KEYWORDS):
        return "economics"
    if _matches_universe_text(row, SPORTS_KEYWORDS):
        return "sports"
    return "other"


def filter_market_universe(markets: pd.DataFrame, universe: str = "All") -> pd.DataFrame:
    """Filter markets to the selected universe before scoring."""

    filtered = markets.copy()
    filtered["market_subtype"] = filtered.apply(classify_market_subtype, axis=1)

    if filtered.empty or universe == "All":
        return filtered.reset_index(drop=True)

    if universe == "Weather":
        mask = filtered.apply(is_weather_market, axis=1)
    elif universe == "Politics":
        mask = filtered.apply(lambda row: _matches_universe_text(row, POLITICS_KEYWORDS), axis=1)
    elif universe == "Economics":
        mask = filtered.apply(lambda row: _matches_universe_text(row, ECONOMICS_KEYWORDS), axis=1)
    elif universe == "Sports":
        mask = filtered.apply(lambda row: _matches_universe_text(row, SPORTS_KEYWORDS), axis=1)
    else:
        mask = pd.Series(True, index=filtered.index)

    filtered = filtered.loc[mask].copy()
    filtered["market_subtype"] = filtered.apply(classify_market_subtype, axis=1)
    return filtered.reset_index(drop=True)


def build_market_explanation(row: pd.Series) -> str:
    """Return a rule-based explanation tailored to the market subtype."""

    market_prob = pd.to_numeric(pd.Series([row.get("market_prob")]), errors="coerce").iloc[0]
    model_prob = pd.to_numeric(pd.Series([row.get("model_prob")]), errors="coerce").iloc[0]
    edge = pd.to_numeric(pd.Series([row.get("edge")]), errors="coerce").iloc[0]
    subtype = str(row.get("market_subtype") or classify_market_subtype(row))

    if pd.isna(market_prob) or pd.isna(model_prob) or pd.isna(edge):
        return "Insufficient pricing inputs to explain the opportunity cleanly."

    lead = f"Model probability is {model_prob:.1%} versus market {market_prob:.1%}, a {edge:+.1%} gap."

    templates = {
        "temperature": "Model probability is meaningfully above market probability for a temperature-threshold event. This looks like a heat/cold threshold market where tail risk may be underpriced.",
        "snow": "Model probability is above market pricing for a snowfall event. These snow-threshold markets can be mispriced when forecast uncertainty shifts late.",
        "rain": "Model probability is above market pricing for a rainfall event. Precipitation threshold events often show nonlinear tail behavior that simple pricing can underweight.",
        "storm": "Model probability is above market pricing for an extreme-weather event. Tail events such as storms or hurricane-related outcomes may remain underpriced when conditions evolve quickly.",
        "general_weather": "Model probability is above market pricing for a weather-linked event. Low-probability forecast tails can move faster than simple market pricing updates.",
        "politics": "Model probability differs from market pricing on a politics-linked event. This may reflect a lag between narrative flow and market repricing.",
        "economics": "Model probability differs from market pricing on an economics-linked event. Macro releases can create nonlinear repricing around threshold outcomes.",
        "sports": "Model probability differs from market pricing on a sports-linked event. These markets can move quickly when matchup assumptions or form shifts are not fully reflected.",
        "other": "Model probability is above market pricing for this market, suggesting the current quote may understate the modeled outcome likelihood.",
    }

    return f"{lead} {templates.get(subtype, templates['other'])}"


__all__ = [
    "UNIVERSE_OPTIONS",
    "build_market_explanation",
    "classify_market_subtype",
    "filter_market_universe",
    "is_weather_market",
    "matches_exclusion_keywords",
    "matches_positive_keywords",
]
