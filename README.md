# Hoops Data Dashboard

An interactive dashboard for exploring **NBA and WNBA** player and team statistics, covering 2003 to the present. Every panel is served from local Parquet files, so the app makes no network calls at request time.

Switch leagues with the NBA/WNBA toggle in the header. Each league has its own Parquet files, and every stat — percentile ranks, similarity scores, league averages, Four Factors — is computed within the selected league only.

## Features

- **Compare** — radar + bar charts for up to 5 players on any subset of metrics.
- **Trends** — season-by-season line charts with a league-average overlay.
- **Percentiles** — color-coded league-percentile rankings for any player-season.
- **Similar Players** — weighted cosine similarity with per-feature sliders + radar overlay.
- **Game Log** — per-game line + rolling average for a player-season.
- **Age Curves** — metric vs. age for multiple players, using birthdates from `player_bio_*.parquet`.
- **Teams** — historical wins/losses, shooting %, and Dean Oliver's Four Factors vs. league average.
- **Shot Chart** — raw scatter or hexbin density view (FG% coloring, shot-volume sizing). Covers the most recent seasons only; see [Shot data coverage](#shot-data-coverage).

## Stack

- **Backend**: FastAPI (Python) — pandas + scikit-learn + scipy. `nba_api` is a fallback only, used when a league has no local shot/game-log file.
- **Frontend**: Vite + React + TypeScript + Tailwind + Radix UI + Plotly.js.
- **Data**: per-league Parquet files from [sportsdataverse](https://github.com/sportsdataverse) via `etl/sdv_etl.py` (hoopR for the NBA, wehoop for the WNBA). `etl/nba_etl.py` remains for stats.nba.com, and `nba_api` is the fallback when a league has no local shot/game-log data.
- **Legacy**: the original Streamlit prototype (`app.py`) is still present as a fallback.

## Setup

Clone and install once:

```bash
git clone https://github.com/ishakotalwar/nba-data-dashboard.git
cd nba-data-dashboard

# Python backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Populate local Parquet data (once per league)
python etl/sdv_etl.py --league nba     # 2003-present
python etl/sdv_etl.py --league wnba

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
backend/         FastAPI app (data.py, leagues.py, live.py, main.py)
frontend/        Vite + React SPA
etl/             sdv_etl.py — sportsdataverse -> data/*.parquet (either league)
                 nba_etl.py — stats.nba.com (blocked on many networks)
data/            Parquet files, suffixed per league
                 (players_nba.parquet, players_wnba.parquet, ...)
app.py           Legacy Streamlit prototype (NBA only)
```

## Leagues

`backend/leagues.py` is the single source of truth for what differs between
leagues — stats.nba.com `LeagueID`, Parquet suffix, the month a season starts
(used to age players), and three-point geometry:

| | NBA | WNBA |
|---|---|---|
| LeagueID | `00` | `10` |
| Season starts | October | May |
| 3PT arc | 23' 9" | 22' 1.75" |
| 3PT corner | 22' 0" | 21' 7.75" |

Every data endpoint takes an optional `?league=nba\|wnba`, defaulting to `nba`
(`/api/health` and `/api/leagues` do not). `GET /api/leagues` reports which
leagues have data on disk, and drives the header toggle.

### How seasons are labelled

Both leagues use a plain year, but it means different things, because the NBA
season spans a new year and the WNBA season does not:

| label | NBA | WNBA |
|---|---|---|
| `2024` | the **2023-24** season (Oct 2023 - Apr 2024) | the **2024** season (May - Sep 2024) |

So an NBA season is named for the year it **ends**. `League.season_format` and
`League.season()` produce the `2024-25` style instead, but those are used only
by `etl/nba_etl.py`; data from `sdv_etl.py` is always a plain year.

Adding another league (G League is `LeagueID=20`) means adding one entry to
`LEAGUES` and running the ETL for it.

### Where the data comes from

stats.nba.com silently drops requests to its `/stats/` API from many IPs — the
connection and TLS handshake succeed, then nothing comes back. That makes
`etl/nba_etl.py` unusable on those networks, for either league.

`etl/sdv_etl.py` sidesteps it. sportsdataverse publishes ESPN-sourced data as
plain Parquet on GitHub — [hoopR](https://github.com/sportsdataverse/hoopR-nba-data)
for the NBA, [wehoop](https://github.com/sportsdataverse/wehoop-wnba-data) for
the WNBA. Both repos share a layout, so one script covers both leagues, and
there is no API to be rate-limited by.

```bash
python etl/sdv_etl.py --league nba
python etl/sdv_etl.py --league wnba --seasons 2015-2026
python etl/sdv_etl.py --league nba --shot-seasons 2020-2026
```

It writes six files per league, all suffixed (`_nba` / `_wnba`):

| file | feeds |
|---|---|
| `players_*` | Compare, Trends, Percentiles, Similar |
| `teams_*` | Teams |
| `teams_master_*` | franchise list |
| `gamelog_*` | Game Log |
| `shots_*` | Shot Chart |
| `player_bio_*` | Age Curves |

Every panel reads from disk; nothing hits the network at request time.

### Shot data coverage

Shots are the bulk of the data (~1 MB per NBA season), so `sdv_etl.py` pulls
only the **most recent 5 seasons** by default. Asking for a shot chart outside
that window returns an empty chart, not an error. To widen it:

```bash
python etl/sdv_etl.py --league nba --shot-seasons 2010-2026
```

Trade-offs versus stats.nba.com:

- **24 seasons instead of one.** Trends and Age Curves need history to say
  anything; a single season cannot show a trend.
- **No per-player `ortg`/`drtg`/`pace`.** Dean Oliver's individual formulas need
  possession-level data ESPN does not publish here, so those are computed per
  **team** instead. `available_metrics()` filters to columns actually present,
  so panels adapt rather than break.
- Player ids are **ESPN ids**, not stats.nba.com ids. The `nba_api` fallback in
  `backend/live.py` only fires for a league with no local file, and will not
  resolve these ids.
- The original stats.nba.com NBA pull is untouched at `data/players.parquet`
  (un-suffixed). `backend/data.py` prefers `players_nba.parquet` and falls back
  to it, so renaming files switches sources.

Data quirks the ETL handles:

- ESPN files All-Star games under `season_type = 2` next to real regular-season
  games. Any team-season with fewer than `MIN_TEAM_GAMES` games is dropped,
  which removes "Team USA"/"TEAM CLARK"/"EAST"/"WEST" without hardcoding names.
- Free throws carry a placeholder shot coordinate rather than a real location,
  and are excluded from shot charts.
- Shot coordinates use `coordinate_x`/`coordinate_y` (court-length and -width in
  feet), **not** the `_raw` pair — the raw values are integer-quantised and put
  roughly half of all three-pointers inside the arc.
- ESPN counts the WNBA Commissioner's Cup final as a regular-season game, so a
  win total can differ from the official standings by one.

## Notes

- Shot charts, game logs, and age curves are served from local Parquet. `backend/live.py` still holds `nba_api` fetchers, but they only fire for a league with no local file — with both leagues populated, nothing calls out.
- Player search (`/api/player-search`) is the exception: it reads `nba_api`'s bundled static rosters, which ship with the package and need no network. It returns names and ids only, no stats.
- For backward compatibility the NBA falls back to un-suffixed `data/players.parquet` if no `_nba` file exists, so a pre-existing pull keeps working. That original stats.nba.com data is still on disk and is the only source here with per-player `ortg`/`drtg`/`pace`.
- `app.py` (legacy Streamlit) reads the un-suffixed files directly, so it still shows the original single-season NBA data and is unaffected by any of this.
- WNBA three-point corner geometry (`three_point_corner` in `backend/leagues.py`) is derived from the FIBA line the league adopted in 2013 and has not been verified against a survey; if shot charts look off at the corners, that constant is the one to adjust.
- Four Factors' ORB% uses OREB / (OREB + DREB) as a proxy — the full formula requires opponent DREB.
- `MIN_GP_FOR_RADAR` in `backend/main.py` keeps low-sample players out of the pool used for radar percentile ranks; it falls back to the full pool if the filter leaves fewer than 30 players.
