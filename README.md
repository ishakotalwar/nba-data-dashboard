# Hoops Data Dashboard

![Player overview](docs/screenshot.png)

An analytics app for the **NBA and WNBA**, covering 2003 to the present. It is built
to answer basketball questions — *who had a season like 2025 SGA?*, *where is Curry
efficient relative to the league?*, *which player-seasons averaged 25+ points on 40%
from three?* — rather than to expose one chart per page.

Everything is served from local Parquet, so no request touches the network.

## Highlights

- **NBA and WNBA**, switched from one control, with every stat computed inside the selected league
- **2003–present**: 16,000+ player-seasons, 1,000+ team-seasons, 1.36M shot locations
- **Player-season as the unit** — "2016 Stephen Curry" is a first-class thing to select and compare
- **Historical similarity**: weighted cosine over every qualifying season, with explanations for *why* two seasons are alike
- **Shot analysis** by zone, against the league average for the same season
- **Stat explorer** for arbitrary filter stacks over the full history
- **Local Parquet API** — fast, offline, and immune to upstream rate limits

## Features

| Page | What it answers |
|---|---|
| **Players** | One player-season: bio, headline stats with league percentile and rank, career trend, recent games |
| **Compare** | Any player-seasons side by side — percentile radar, raw values, and the gap to that season's league average. A **Career** mode plots career trajectories against age |
| **Similarity** | Most similar seasons from anywhere in the dataset, with weighting presets (Overall / Scoring / Shooting / Playmaking / Defense / Custom) and per-result explanations |
| **Shot Analysis** | Hexbin and scatter shot charts, zone-by-zone accuracy vs. the league, and two player-seasons compared |
| **Teams** | League table, team profile over time, any two team-seasons head to head, and all-time leaderboards |
| **Explorer** | Filter every player-season by any metric with `>`, `>=`, `<`, `<=`, `=` or `between`, then sort and page through the results |

## Architecture

```text
      sportsdataverse
   (hoopR · wehoop on GitHub)
              │
              ▼
         Python ETL              etl/sdv_etl.py
              │
              ▼
       Local Parquet             data/*_{nba,wnba}.parquet
              │
              ▼
          FastAPI                backend/routers/*
              │
              ▼
    React + TypeScript           frontend/src
              │
              ▼
     Plotly analytics UI
```

Three rules keep it from sprawling:

- **`backend/leagues.py` is the only place leagues differ.** League id, season format, season start month, Parquet suffix and three-point geometry live there; nothing else branches on league.
- **`backend/analytics.py` holds the maths.** Percentiles, Four Factors and ranking are pure functions over DataFrames, so routers stay thin and the same numbers back every page.
- **`frontend/src/lib/metrics.ts` is the only place metrics are described.** Labels, formatting, category and direction come from one table used by every panel.

## Tech Stack

- **Backend**: FastAPI, pandas, scikit-learn, scipy
- **Frontend**: Vite, React, TypeScript, Tailwind, Radix UI, Plotly.js
- **Data**: Parquet on disk, built by a Python ETL from [sportsdataverse](https://github.com/sportsdataverse)

## Setup

```bash
git clone https://github.com/ishakotalwar/nba-data-dashboard.git
cd nba-data-dashboard

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python etl/sdv_etl.py --league nba      # 2003-present
python etl/sdv_etl.py --league wnba

cd frontend && npm install && cd ..
```

Run it in two terminals:

```bash
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

```bash
cd frontend
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api/*` to the backend on :8000.

The original Streamlit prototype is still there: `streamlit run app.py`.

## Data Pipeline

`etl/sdv_etl.py` downloads ESPN-sourced Parquet published by
[hoopR](https://github.com/sportsdataverse/hoopR-nba-data) (NBA) and
[wehoop](https://github.com/sportsdataverse/wehoop-wnba-data) (WNBA), then reshapes
it into six files per league:

| File | Feeds |
|---|---|
| `players_*` | Players, Compare, Similarity, Explorer |
| `teams_*` | Teams |
| `teams_master_*` | franchise list |
| `gamelog_*` | recent games |
| `shots_*` | Shot Analysis |
| `player_bio_*` | bio and age |

```bash
python etl/sdv_etl.py --league nba
python etl/sdv_etl.py --league wnba --seasons 2015-2026
python etl/sdv_etl.py --league nba --shot-seasons 2010-2026
```

Shots are the bulk of the data (~1 MB per NBA season), so only the most recent five
seasons are pulled by default; `--shot-seasons` widens that.

`etl/nba_etl.py` still targets stats.nba.com directly and is the better source when
reachable — it carries real per-player `ortg`/`drtg`/`pace` rather than derived
figures. See the note on blocking below.

## Data Coverage

| | NBA | WNBA |
|---|---|---|
| Seasons | 2003–2026 | 2003–2026 |
| Player-seasons | 12,073 | 3,983 |
| Distinct players | 2,480 | 993 |
| Team-seasons | 718 | 302 |
| Franchises (incl. relocations) | 36 | 23 |
| Shot locations | 1,171,163 (2022–2026) | 188,088 (2022–2026) |
| Game rows | 659,034 | 106,895 |

Roughly 12 MB of Parquet in total.

## Technical Notes / Limitations

**stats.nba.com is blocked from many networks.** Requests to its `/stats/` API
complete the TCP and TLS handshake and then never respond. That is why the default
pipeline is sportsdataverse rather than `nba_api`. `backend/live.py` keeps the
`nba_api` fetchers as a fallback for a league with no local file; with both leagues
populated nothing calls out.

**No per-player `ortg`/`drtg`/`pace`.** Dean Oliver's individual formulas need
possession-level data ESPN does not publish here, so those are computed per *team*.
`available_metrics()` filters to columns actually present, so panels adapt rather
than break. The original stats.nba.com pull is still on disk at the un-suffixed
`data/players.parquet`, and `backend/data.py` falls back to it when no `_nba` file
exists — renaming files switches sources.

**Player ids are ESPN ids**, not stats.nba.com ids, so the `nba_api` fallback cannot
resolve them.

**Season labels mean different things per league.** Both use a plain year, but an NBA
season is named for the year it *ends*:

| Label | NBA | WNBA |
|---|---|---|
| `2024` | the 2023-24 season (Oct 2023 – Apr 2024) | the 2024 season (May – Sep 2024) |

**ESPN data quirks the ETL handles:**

- All-Star games are filed under `season_type = 2` beside real regular-season games. Any team-season with fewer than `MIN_TEAM_GAMES` games is dropped, which removes "Team USA" / "TEAM CLARK" / "EAST" / "WEST" without hardcoding names.
- Free throws carry a placeholder shot coordinate rather than a real location, and are excluded from shot charts.
- Shot coordinates use `coordinate_x`/`coordinate_y`, **not** the `_raw` pair — the raw values are integer-quantized and put roughly half of all three-pointers inside the arc.
- ESPN counts the WNBA Commissioner's Cup final as a regular-season game, so a win total can differ from the official standings by one.
- One NBA team name arrives with an embedded carriage return (`NO/Oklahoma City\r\n Hornets`) and shows up as a broken entry in team pickers.

**Other caveats:**

- Four Factors' ORB% uses `OREB / (OREB + DREB)` as a proxy; the full formula needs opponent DREB.
- WNBA three-point corner geometry (`three_point_corner` in `backend/leagues.py`) is derived from the FIBA line adopted in 2013 and has not been verified against a survey.
- Cosine similarity over eight standardized features tends to cluster scores in the high 90s. The *ranking* is meaningful; the absolute percentage is not a probability.
- `MIN_GP_FOR_POOL` in `backend/analytics.py` keeps low-sample players out of percentile pools, falling back to the full pool if the filter leaves fewer than 30 players.
