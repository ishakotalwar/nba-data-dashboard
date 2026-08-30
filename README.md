# NBA Data Dashboard

An interactive dashboard for exploring NBA player and team statistics. Combines a local Parquet-backed historical dataset with live fetches from stats.nba.com for shot charts, game logs, and player bios.

## Features

- **Compare** — radar + bar charts for up to 5 players on any subset of metrics.
- **Trends** — season-by-season line charts with a league-average overlay.
- **Percentiles** — color-coded league-percentile rankings for any player-season.
- **Similar Players** — weighted cosine similarity with per-feature sliders + radar overlay.
- **Game Log** — per-game line + rolling average, live from stats.nba.com.
- **Age Curves** — metric vs. age for multiple players, with birthdates pulled on demand.
- **Teams** — historical wins/losses, shooting %, and Dean Oliver's Four Factors vs. league average.
- **Shot Chart** — raw scatter or hexbin density view (FG% coloring, shot-volume sizing).

## Stack

- **Backend**: FastAPI (Python) — pandas + scikit-learn + scipy + nba_api.
- **Frontend**: Vite + React + TypeScript + Tailwind + Radix UI + Plotly.js.
- **Data**: local Parquet files populated by `etl/nba_etl.py`; live data via `nba_api`.
- **Legacy**: the original Streamlit prototype (`app.py`) is still present as a fallback.

## Setup

Clone and install once:

```bash
git clone https://github.com/your-username/nba-data-dashboard.git
cd nba-data-dashboard

# Python backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Populate local Parquet data (once)
python etl/nba_etl.py

# Frontend
cd frontend
npm install
cd ..
```

## Run (dev)

Two terminals. Backend:

```bash
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to the FastAPI backend on :8000.

## Run the legacy Streamlit app

```bash
source venv/bin/activate
streamlit run app.py
```

## Project layout

```
backend/         FastAPI app (data.py, live.py, main.py)
frontend/        Vite + React SPA
etl/             nba_etl.py — populates data/*.parquet
data/            Parquet files (players, teams, shots)
app.py           Legacy Streamlit prototype
```

## Notes

- Live endpoints (shots, game log, age curves) hit stats.nba.com which rate-limits aggressively from some IPs. Results are cached in-process; a VPN to a US residential IP helps if fetches time out.
- Four Factors' ORB% uses OREB / (OREB + DREB) as a proxy — the full formula requires opponent DREB.
