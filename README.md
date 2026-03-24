# prediction-market-scanner

Prediction markets are often efficient at the center and sloppy at the tails. This project is an AI-assisted research system for scanning Polymarket and Kalshi for low-probability contracts where market-implied odds may diverge from model-driven probabilities.

It is built for fast iteration: ingest public market data, normalize it into a shared schema, score candidate opportunities, and inspect the results through a backtest-friendly Streamlit dashboard. The current focus is research, ranking, and evaluation, not automated execution.

## Why This Exists

Low-probability event markets are noisy, thinly traded, and easy to overlook. That makes them a good target for a system that can:

- standardize market data across platforms
- estimate event probabilities from domain-specific signals
- rank opportunities with transparent logic
- replay threshold-based strategies before touching anything live

The first domain focus here is weather-linked binary events

## Architecture

```text
prediction-market-scanner/
|-- app/
|   `-- app.py                  # Streamlit dashboard
|-- scripts/
|   `-- demo_backtest_synthetic.py
|-- src/
|   |-- backtest/
|   |   |-- metrics.py          # ROI, hit rate, drawdown, Brier score
|   |   `-- runner.py           # Threshold strategy backtest runner
|   |-- ingest/
|   |   |-- kalshi.py           # Kalshi public market ingestion
|   |   |-- polymarket.py       # Polymarket public market ingestion
|   |   `-- schema.py           # Shared normalized market model
|   `-- models/
|       |-- edge_score.py       # Opportunity ranking logic
|       `-- weather_mapping.py  # Forecast-to-probability mapping
|-- tests/
|   |-- test_normalized_market_record.py
|   `-- test_weather_mapping.py
|-- .env.example
|-- requirements.txt
`-- README.md
```

## Data Flow

```text
Polymarket / Kalshi public APIs
        ->
platform-specific ingestors
        ->
shared normalized market schema
        ->
model probability estimates
        ->
edge scoring and ranking
        ->
backtest runner + metrics
        ->
Streamlit dashboard
```

More concretely:

1. `src/ingest/polymarket.py` and `src/ingest/kalshi.py` fetch and normalize venue-specific market data.
2. `src/ingest/schema.py` defines the shared market record boundary for downstream modules.
3. `src/models/weather_mapping.py` turns forecast-style inputs into binary event probabilities.
4. `src/models/edge_score.py` ranks opportunities using market probability, model probability, time to expiry, liquidity, and data quality penalties.
5. `src/backtest/runner.py` simulates a simple threshold strategy and writes daily decisions to CSV.
6. `app/app.py` presents opportunities, charts, and backtest results in a recruiter-friendly MVP dashboard.

## Screenshots

Add screenshots once the dashboard is running locally:

- `docs/images/dashboard-overview.png`  
  Placeholder: top opportunities table + summary cards
- `docs/images/opportunity-scatter.png`  
  Placeholder: market probability vs. model probability chart
- `docs/images/equity-curve.png`  
  Placeholder: backtest equity curve and daily log

Example markdown when ready:

```md
![Dashboard Overview](docs/images/dashboard-overview.png)
![Opportunity Scatter](docs/images/opportunity-scatter.png)
![Equity Curve](docs/images/equity-curve.png)
```

## Local Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Copy environment variables

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

### 4. Run the synthetic backtest demo

```bash
python scripts/demo_backtest_synthetic.py
```

This writes a decisions CSV under `artifacts/backtest/` and prints summary metrics plus the first rows of the decision log.

### 5. Launch the Streamlit app

```bash
streamlit run app/app.py
```

The dashboard attempts live market ingestion first. If live feeds are unavailable, it falls back to synthetic data so the app remains demoable.

## How Scoring Works

The opportunity score is intentionally simple and inspectable. The system is not trying to hide the ranking logic behind a black box.

Core idea:

```text
edge = model_prob - market_prob
```

Markets rank higher when they have:

- larger positive edge
- enough time remaining before expiry
- better liquidity or tighter pricing

Markets rank lower when they have:

- stale data
- missing quote fields
- weak or negative edge

The current scoring implementation lives in `src/models/edge_score.py` and uses a linear additive formula:

- positive edge is the main driver
- time-to-expiry adds a moderate boost, capped at a fixed horizon
- liquidity helps when volume-like fields exist, otherwise bid/ask spread acts as a weak proxy
- stale or incomplete data is penalized rather than silently ignored

This is deliberate. For an MVP, transparent heuristics are easier to debug, explain, and improve than a heavier ranking system with unclear failure modes.

## Backtesting

The backtest runner in `src/backtest/runner.py` simulates a simple strategy:

- restrict to low-probability markets
- enter when `model_prob - market_prob` exceeds a threshold
- use a fixed stake per trade
- compute `ROI`, `hit_rate`, `average_edge`, `max_drawdown`, and `Brier score`
- export daily decisions to CSV for inspection and dashboard use

### Backtesting caveats

This backtest is intentionally lightweight and should be treated as a research tool, not a production trading simulator.

Known limitations:

- no slippage model
- no fees
- no partial fills
- no queue position modeling
- no latency modeling
- no market impact modeling
- simplified payout assumptions
- synthetic demo data is not representative of real market microstructure

A good backtest here is a filter for ideas, not proof that a live strategy will survive deployment.

## Shared Schema

`src/ingest/schema.py` defines a normalized market record that both Polymarket and Kalshi data can map into.

Normalized fields include:

- `source_platform`
- `market_id`
- `title`
- `category`
- `market_prob`
- `bid`
- `ask`
- `last`
- `close_time`
- `raw_json`

This keeps the rest of the system insulated from venue-specific payload differences.

## Weather Probability Mapping

`src/models/weather_mapping.py` provides a provider-agnostic interface for converting forecast features into binary market probabilities.

Current supported event types:

- temperature above threshold
- precipitation above threshold
- snowfall occurrence

The current calibration layer is intentionally lightweight. It exists as a clean interface for future empirical calibration rather than pretending the system already has institutional-grade weather calibration.

## Current MVP Capabilities

- public market ingestion for Polymarket and Kalshi
- shared market normalization layer
- transparent opportunity scoring
- simple synthetic-data backtesting workflow
- Streamlit dashboard with live-data fallback behavior

## Roadmap

- add richer historical data storage for repeatable backtests
- connect real weather forecast providers into the mapping layer
- improve liquidity-aware ranking and market quality filters
- build venue-specific parsers for more contract types
- add calibration based on historical forecast vs. realized outcome data
- support event clustering and duplicate-market detection across venues
- expand dashboard views for market drill-down and postmortem analysis
- add optional execution components only when explicitly enabled

## Disclaimer

This repository is for research and education unless trading or execution components are explicitly enabled. Nothing here is investment advice, a solicitation to trade, or a claim that the current models are production-ready.

## License / Usage Notes

If you publish or adapt this system, be explicit about:

- which data sources are public vs. authenticated
- which components are synthetic vs. live
- which assumptions are heuristic vs. empirically validated

That clarity matters more than pretending an MVP is already a finished trading platform.
