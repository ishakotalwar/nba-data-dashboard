"""Upcoming schedules, current rosters and injuries — the "what is true now"
pipeline, kept separate from the historical ETL.

`sdv_etl.py` builds the record of what happened; this builds the list of what
is *going* to happen, which is the one thing the analytics data cannot contain.
They refresh on different cadences — history changes once a night, a schedule
changes whenever a game is postponed — so they stay separate pipelines.

    python etl/schedule_etl.py                 # both leagues, current + next season
    python etl/schedule_etl.py --league wnba
    python etl/schedule_etl.py --seasons 2026 2027

Rosters matter as much as the schedule: a projection built from "whoever
finished last season on this team" puts a quarter of the league on the wrong
bench once trades and free agency have happened.
"""
from __future__ import annotations

import argparse
import io
import re
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

SCHEDULE_COLUMNS = [
    "game_id", "league", "season", "date", "tipoff",
    "home", "away", "home_name", "away_name",
    "status", "completed",
]

ROSTER_COLUMNS = [
    "season", "league", "team", "player_id", "player_name",
    "jersey", "position", "status",
]

INJURY_COLUMNS = [
    "league", "player_id", "player_name", "team",
    "status", "type", "detail", "comment", "fetched",
]

# ESPN's public injuries feed. Undocumented, so treat a failure as "no injury
# data" rather than an error: a missing flag is better than a broken page.
INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/{league}/injuries"


def url_for(league: League, season: int) -> str:
    repo, seg = REPOS[league.key]
    return (f"https://raw.githubusercontent.com/{repo}/main/{seg}"
            f"/schedules/parquet/{seg}_schedule_{season}.parquet")


def roster_url_for(league: League, season: int) -> str:
    repo, seg = REPOS[league.key]
    return (f"https://raw.githubusercontent.com/{repo}/main/{seg}"
            f"/rosters/parquet/rosters_{season}.parquet")


def fetch_url(url: str) -> pd.DataFrame | None:
    r = requests.get(url, timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def normalise_roster(df: pd.DataFrame, league: League, season: int) -> pd.DataFrame:
    out = pd.DataFrame({
        "season": str(season),
        "league": league.key,
        "team": df["team_abbreviation"].astype(str).map(league.canonical_team),
        "player_id": pd.to_numeric(df["athlete_id"], errors="coerce"),
        "player_name": df["display_name"].astype(str),
        "jersey": df.get("jersey", pd.Series("", index=df.index)).astype(str),
        "position": df.get("position_abbreviation", pd.Series("", index=df.index)).astype(str),
        "status": df.get("status_name", pd.Series("", index=df.index)).astype(str),
    })
    return out.dropna(subset=["player_id"])[ROSTER_COLUMNS]


def build_rosters(league: League, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        raw = fetch_url(roster_url_for(league, season))
        if raw is None or raw.empty:
            print(f"  {league.label} {season} roster: not published")
            continue
        rows = normalise_roster(raw, league, season)
        print(f"  {league.label} {season} roster: {len(rows):,} players "
              f"across {rows.team.nunique()} teams")
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=ROSTER_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def espn_athlete_id(athlete: dict) -> int | None:
    """The feed omits the athlete id, but embeds it in the headshot URL and in
    every player link, e.g. `.../players/full/3058895.png`."""
    headshot = (athlete.get("headshot") or {}).get("href", "")
    m = re.search(r"/(\d+)\.png", headshot)
    if m:
        return int(m.group(1))
    for link in athlete.get("links") or []:
        m = re.search(r"/id/(\d+)", link.get("href", "") or "")
        if m:
            return int(m.group(1))
    return None


def build_injuries(league: League) -> pd.DataFrame:
    """Current injury report. A snapshot only — the feed carries no history,
    so this says who is hurt now, not who was hurt for any past game."""
    try:
        r = requests.get(INJURY_URL.format(league=league.key), timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:  # noqa: BLE001 - any failure means "no injury data"
        print(f"  {league.label} injuries: unavailable ({type(e).__name__})")
        return pd.DataFrame(columns=INJURY_COLUMNS)

    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M")
    rows = []
    for team in payload.get("injuries", []):
        for entry in team.get("injuries", []):
            athlete = entry.get("athlete") or {}
            pid = espn_athlete_id(athlete)
            if pid is None:
                continue
            details = entry.get("details") or {}
            rows.append({
                "league": league.key,
                "player_id": pid,
                "player_name": athlete.get("displayName", ""),
                "team": team.get("displayName", ""),
                "status": entry.get("status", ""),
                "type": details.get("type", "") or "",
                "detail": details.get("detail", "") or "",
                "comment": (entry.get("shortComment") or "")[:400],
                "fetched": now,
            })
    df = pd.DataFrame(rows, columns=INJURY_COLUMNS)
    if df.empty:
        print(f"  {league.label} injuries: none reported")
    else:
        counts = df.status.value_counts().to_dict()
        print(f"  {league.label} injuries: {len(df)} players ({counts})")
    return df


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
    return out.dropna(subset=["date"])[SCHEDULE_COLUMNS]


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
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
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
        print(f"  wrote {path.relative_to(ROOT)} ({len(df):,} rows)")

        rosters = build_rosters(lg, seasons)
        if rosters.empty:
            print(f"  no rosters published for {lg.label}\n")
            continue
        rpath = DATA / f"roster{lg.suffix}.parquet"
        rosters.to_parquet(rpath, index=False)
        print(f"  wrote {rpath.relative_to(ROOT)} ({len(rosters):,} rows)")

        injuries = build_injuries(lg)
        ipath = DATA / f"injury{lg.suffix}.parquet"
        injuries.to_parquet(ipath, index=False)
        print(f"  wrote {ipath.relative_to(ROOT)} ({len(injuries):,} rows)\n")


if __name__ == "__main__":
    main()
