# Full Court

**Live: <https://full-court-six.vercel.app>**

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
- **Two halves**: explore what happened, or what happens next — switchable from any page
- **Predictions**: team Elo ratings and win probabilities, plus projected player lines, each shown against its own backtest
- **Ask Full Court** — natural-language questions, answered from the data rather than from a model
- **Light and dark themes**, remembered per browser
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
| **Predictions · Teams** | Elo power ratings, win probability and projected margin for any matchup, with accuracy, Brier score and a calibration curve |
| **Predictions · Players** | Projected per-game line for next season — highest projected, biggest risers and fallers — with the seasons behind each number |
| **Ask Full Court** | A question in plain English — “which players since 2010 averaged 25+ points?” — parsed into a structured query, executed against local Parquet, with a button that opens the answer in the matching page |

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

- **Backend**: FastAPI, pandas, NumPy
- **Frontend**: Vite, React, TypeScript, Tailwind, Radix UI, Plotly.js
- **Data**: Parquet on disk, built by a Python ETL from [sportsdataverse](https://github.com/sportsdataverse)

## Setup

```bash
git clone https://github.com/ishakotalwar/full-court.git
cd full-court

python -m venv venv
source venv/bin/activate
pip install -r requirements-local.txt   # backend + Streamlit prototype + ETL

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

`requirements.txt` holds only what the FastAPI backend imports, because that
is the file the deployment installs and serverless bundles are size-capped.
`requirements-local.txt` pulls it in and adds Streamlit, Plotly, SciPy,
scikit-learn and the ETL's dependencies.

## Predictions

The app opens on a choice — explore stats, or explore predictions — and the
header switches between them from any page. Both halves cover the NBA and the
WNBA.

**Games** is a calendar of the schedule: pick a date, see that day's games and
what the model makes of each. The schedule is the one thing the historical data
cannot contain, so it has its own pipeline — `etl/schedule_etl.py` writes
`data/schedule_{nba,wnba}.parquet` and refreshes independently of the analytics
ETL. Times are converted to US Eastern before the calendar day is taken, since
a 02:00 UTC tip-off belongs to the previous evening.

The same pipeline fetches **rosters**, which the projections need as much as
the schedule: a rotation built from "whoever finished last season on this team"
puts about a quarter of the league on the wrong bench once trades, free agency
and the draft have happened. Rotations follow the published roster for the
season being predicted, and fall back to last season's teams only when no
roster exists yet.

Inside a season already under way the projections use that season's games —
ignoring them would forecast a live season from last year's form — while the
backtest still sees only seasons strictly before its target, so the reported
error stays honest. Rotation membership is decided by minutes rather than games
played, since a player at 30 minutes a night belongs in it whether they have
appeared 11 times or 41.

```bash
python etl/schedule_etl.py                # schedules + rosters, both leagues
python etl/schedule_etl.py --league wnba
```

**Teams** use Elo: a home-court term, a margin-of-victory update, and a pull
back toward the mean between seasons. Game results are rebuilt from the player
game logs, which carry no score but do name both teams and mark the home side.

| | NBA | WNBA |
|---|---|---|
| Games | 28,853 | 5,140 |
| Accuracy | 65.5% | 65.8% |
| Always pick home | 56.5% | 55.3% |
| Brier | 0.216 (vs 0.246) | 0.217 (vs 0.247) |

Scored from 2015 onward, each game predicted before it was played and before it
updated the ratings. Accuracy alone flatters a model on a base rate this high,
so Brier score, log loss and a calibration curve are shown alongside it.

**Players** use a Marcel-style projection: recent seasons weighted 12/3/1 and by
games played, regressed toward the league mean, then adjusted for age. The
weights were fitted against held-out seasons rather than chosen by taste — a
flatter blend loses to simply repeating last season for volume stats.

| Metric | NBA MAE | vs "same as last season" | WNBA MAE | vs baseline |
|---|---|---|---|---|
| Points | 2.42 | +4.4% | 2.53 | +6.8% |
| Rebounds | 0.84 | +6.1% | 0.94 | +11.7% |
| Assists | 0.66 | +5.5% | 0.67 | +8.7% |
| True shooting % | 0.037 | +15.2% | 0.043 | +16.5% |

Two data details the models depend on, both in `backend/leagues.py`:
franchises that changed abbreviation (`SEA`→`OKC`, `SAS`→`SA`→`LV`, and six more
in the WNBA) are folded onto one identity so a rating is not split in half, and
only a league whose season crosses New Year is labelled by the year it ends in.

## Ask Full Court

A question box, reachable from every page (the floating button, or ⌘K), that
resizes between three widths.

The design rule is that **a model may choose what to compute, never what the
answer is**:

```text
question -> parse to AskQuery -> validate -> existing analytics -> Parquet -> answer
```

`backend/ask_schema.py` defines the query language — five intents (`explorer`,
`similarity`, `compare`, `shot_analysis`, `team_explorer`), a closed operator
set, and the natural-language-to-column aliases. `backend/routers/ask.py`
executes it by calling the same router functions the UI calls, so no analytics
logic is duplicated. Every statistic on screen came out of a Parquet file.

It understands four shapes of question: **conditions** ("25+ points and 40%
from three"), **rankings** ("best WNBA defensive players", "top scorers since
2015"), **one player-season** (similarity, comparison, shot zones), and **team
leaderboards**.

Names are resolved without a model: exact match, then substring, then spelling
distance against every name and each part of it — so *Jokich*, *Yannis* and
*Steph Curry* all land on the right player, and a genuinely ambiguous name
returns a list to choose from instead of a guess. Role phrases resolve too —
*rim protector* to blocks, *floor general* to assists.

A ranking always names the metric it used and the games floor it applied, and
says so when the dataset forces a proxy: "best defensive players" ranks on
blocks, because there is no per-player defensive rating to rank on.

Parsing runs in two stages (`backend/ask_parse.py`):

1. **Rules** — regex over the shapes basketball questions take. Free, instant,
   offline, deterministic, and enough for all eight worked examples.
2. **Claude** — only for what the rules miss, with the schema enforced by
   structured output. Requires `anthropic` installed and `ANTHROPIC_API_KEY`
   set; without either, the feature runs on the rule parser alone. To enable it
   on a deployment, add `anthropic` to `requirements.txt` and set the key.

Questions the dataset cannot answer are declined rather than guessed — salary,
awards, playoffs, injuries, draft, a metric that isn't stored, a season out of
range, or shots before the 2021-22 coverage start. An ambiguous name returns a
clarification instead of a guess.

## Theme

Light and dark, toggled from the header and stored in `localStorage`; the
initial choice follows `prefers-color-scheme`. The palette lives in CSS
variables in `frontend/src/index.css`, and `tailwind.config.js` resolves every
colour through them, so a theme swap is one attribute on `<html>`. Plotly draws
to canvas and cannot read CSS variables, so `Plot.tsx` samples the palette and
redraws when the theme changes.

## Deployment

Deployed on Vercel as one project: the Vite build is the static site, and
`api/index.py` exposes the FastAPI app as a Python serverless function.
`vercel.json` serves the built files first and sends `/api/*` to the function.

Two constraints shape the setup:

- **The project's Framework Preset must be Vite, not FastAPI.** The FastAPI
  preset routes every request to the Python function, so the built frontend
  never gets served.
- **A serverless function is capped at 250 MB unzipped.** pyarrow, SciPy and
  scikit-learn together are more than that, so the backend uses NumPy in place
  of the two SciPy/scikit-learn call sites it had, and fastparquet (10 MB) in
  place of pyarrow (112 MB). That is why `requirements.txt` is deliberately
  small — Vercel installs it from the repository root.

The Parquet in `data/` is committed and ships with the deployment via
`includeFiles` in `vercel.json`, so the API has its data with no external calls.

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

Roughly 13 MB of Parquet in total.

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

| Stored label | NBA | WNBA |
|---|---|---|
| `2024` | the 2023-24 season (Oct 2023 – Apr 2024) | the 2024 season (May – Sep 2024) |

The stored label is what the API accepts and returns. The UI renders it through
`frontend/src/lib/season.ts` using the `season_format` reported by `/api/meta`,
so an NBA season shows as `2023-24` and a WNBA season as `2024`.

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
