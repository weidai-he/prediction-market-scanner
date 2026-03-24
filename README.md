# prediction-market-scanner

Python scaffold for scanning prediction markets for potentially mispriced low-probability events. The project is designed to stay modular so we can add richer pricing logic, weather-driven models, and research workflows without reshaping the package later.

## What is included

- `src/` package layout
- Ingestion module scaffolds for Polymarket and Kalshi
- Low-probability scanner scaffold
- Weather model placeholder
- Backtesting module scaffold
- Streamlit dashboard
- Logging and configuration utilities

## What is not included yet

- Trading or order execution
- Live strategy logic
- Production data storage
- Full exchange authentication flows

## Project structure

```text
prediction-market-scanner/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── prediction_market_scanner/
│       ├── config.py
│       ├── logging_utils.py
│       ├── main.py
│       ├── dashboard/
│       ├── ingestion/
│       ├── models/
│       ├── scanners/
│       ├── backtesting/
│       └── schemas/
└── tests/
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment variables:

```bash
copy .env.example .env
```

4. Run the CLI scaffold:

```bash
set PYTHONPATH=src
python -m prediction_market_scanner.main
```

5. Run the Streamlit dashboard:

```bash
set PYTHONPATH=src
streamlit run src/prediction_market_scanner/dashboard/app.py
```

## Development notes

- Ingestion clients currently return mock data so the scaffold runs without credentials.
- The scanner only flags candidate opportunities. It does not place trades.
- The weather model is a placeholder interface for future probability estimation work.

## Next suggested steps

1. Add authenticated data clients for Polymarket and Kalshi.
2. Define a normalized market schema for more market types.
3. Build pricing and mispricing heuristics on top of historical data.
4. Add tests around scanner logic and backtest assumptions.
