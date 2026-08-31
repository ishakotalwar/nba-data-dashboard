"""Upcoming-game schedules, kept separate from the historical ETL.

`sdv_etl.py` builds the record of what happened; this builds the list of what
is *going* to happen, which is the one thing the analytics data cannot contain.
They refresh on different cadences — history changes once a night, a schedule
changes whenever a game is postponed — so they stay separate pipelines.

    python etl/schedule_etl.py                 # both leagues, current + next season
    python etl/schedule_etl.py --league wnba
    python etl/schedule_etl.py --seasons 2026 2027
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.leagues import LEAGUES, League  # noqa: E402

DATA = ROOT / "data"
TIMEOUT = 120

# Same repos as the historical ETL, different dataset and file naming.
REPOS = {
    "nba": ("sportsdataverse/hoopR-nba-data", "nba"),
    "wnba": ("sportsdataverse/wehoop-wnba-data", "wnba"),
}

# ESPN times are UTC. A 02:00Z tip-off is the previous evening in North
# America, so the calendar day has to come from a US timezone or half the
# schedule lands on the wrong date.
GAME_DAY_TZ = "America/New_York"

# Regular-season and playoff games. Excludes ALLSTAR and CC (Commissioner's
# Cup final), whose "teams" are one-off squads with no rating.
KEEP_TYPES = {"STD"}

COLUMNS = [
    "game_id", "league", "season", "date", "tipoff",
    "home", "away", "home_name", "away_name",
    "status", "completed",
]


def url_for(league: League, season: int) -> str:
    repo, seg = REPOS[league.key]
    return (f"https://raw.githubusercontent.com/{repo}/main/{seg}"
            f"/schedules/parquet/{seg}_schedule_{season}.parquet")


def fetch(league: League, season: int) -> pd.DataFrame | None:
    r = requests.get(url_for(league, season), timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def normalise(df: pd.DataFrame, league: League, season: int) -> pd.DataFrame:
    if "type_abbreviation" in df.columns:
        df = df[df["type_abbreviation"].isin(KEEP_TYPES)]

    when = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(GAME_DAY_TZ)
    completed = df.get("status_type_completed")
    state = df.get("status_type_state", pd.Series("", index=df.index))

    out = pd.DataFrame({
        "game_id": df["id"].astype(str),
        "league": league.key,
        "season": str(season),
        "date": when.dt.strftime("%Y-%m-%d"),
        "tipoff": when.dt.strftime("%H:%M"),
        # Franchises that changed abbreviation are folded onto today's code so
        # a scheduled game matches the team's rating.
        "home": df["home_abbreviation"].astype(str).map(league.canonical_team),
        "away": df["away_abbreviation"].astype(str).map(league.canonical_team),
        "home_name": df.get("home_display_name", df["home_abbreviation"]).astype(str),
        "away_name": df.get("away_display_name", df["away_abbreviation"]).astype(str),
        "status": state.astype(str),
        "completed": (completed.fillna(False).astype(bool) if completed is not None
                      else pd.Series(False, index=df.index)),
    })
    return out.dropna(subset=["date"])[COLUMNS]


def build(league: League, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        raw = fetch(league, season)
        if raw is None or raw.empty:
            print(f"  {league.label} {season}: not published")
            continue
        rows = normalise(raw, league, season)
        upcoming = int((~rows.completed).sum())
        print(f"  {league.label} {season}: {len(rows):,} games ({upcoming:,} not yet played)")
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["game_id"])
              .sort_values(["date", "tipoff"])
              .reset_index(drop=True))


def default_seasons(league: League) -> list[int]:
    """The season in progress and the one after it — enough for a calendar."""
    today = pd.Timestamp.today()
    current = today.year + 1 if today.month >= league.season_start_month and \
        league.season_start_month >= 7 else today.year
    return [current, current + 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", choices=sorted(LEAGUES), help="default: both")
    ap.add_argument("--seasons", nargs="+", type=int, help="default: current and next")
    args = ap.parse_args()

    for key in ([args.league] if args.league else sorted(LEAGUES)):
        lg = LEAGUES[key]
        seasons = args.seasons or default_seasons(lg)
        print(f"{lg.label} schedule (seasons {', '.join(map(str, seasons))})")
        df = build(lg, seasons)
        if df.empty:
            print(f"  nothing to write for {lg.label}")
            continue
        path = DATA / f"schedule{lg.suffix}.parquet"
        df.to_parquet(path, index=False)
        print(f"  wrote {path.relative_to(ROOT)} ({len(df):,} rows)\n")


if __name__ == "__main__":
    main()
