import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict, List

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from nba_api.stats.static import players as players_static
from nba_api.stats.static import teams as teams_static
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    teamyearbyyearstats,
    shotchartdetail,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

from backend.leagues import LEAGUES, League  # noqa: E402  (needs ROOT on sys.path)

REQUEST_PAUSE = 0.8  # avoid too many requests to API too quickly

# Sensible per-league defaults for the one-player shot-chart sample.
DEFAULT_SEASON_YEAR = 2024
DEMO_PLAYERS = {"nba": "Stephen Curry", "wnba": "A'ja Wilson"}



class NBARetryError(Exception):
    pass


@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.8, min=1, max=8),
)
def _first_df(endpoint_obj):
    """Call an nba_api endpoint object and return its first dataframe with retries."""
    dfs = endpoint_obj.get_data_frames()
    if not dfs:
        raise NBARetryError("NBA API returned no data frames")
    return dfs[0]


@dataclass
class PlayerIdentity:
    id: int
    full_name: str


def find_player(name: str, league: League) -> Optional[PlayerIdentity]:
    if league.key == "nba":
        hits = players_static.find_players_by_full_name(name)
    else:
        finder = getattr(players_static, f"find_{league.key}_players_by_full_name", None)
        hits = finder(name) if finder else None
    if not hits:
        return None
    return PlayerIdentity(id=hits[0]["id"], full_name=hits[0]["full_name"])


def fetch_teams_master(league: League) -> pd.DataFrame:
    if league.key == "nba":
        return pd.DataFrame(teams_static.get_teams())
    getter = getattr(teams_static, f"get_{league.key}_teams", None)
    if getter is None:
        raise SystemExit(f"nba_api has no static team list for the {league.label}")
    return pd.DataFrame(getter())


def fetch_team_year_by_year(team_id: int, league: League) -> pd.DataFrame:
    ep = teamyearbyyearstats.TeamYearByYearStats(
        team_id=team_id,
        league_id_nullable=league.league_id,
    )
    return _first_df(ep)


def _league_dash_player_stats_safe(season: str, per_mode: str, measure_type: str, league: League):
    try:
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense=measure_type,
            league_id_nullable=league.league_id,
        )
    except TypeError:
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_def=measure_type,
            league_id_nullable=league.league_id,
        )


def fetch_players_base_adv(season: str, league: League) -> pd.DataFrame:
    """
    League-wide player stats for a season:
    - Base: PTS, AST, REB, FG%, etc.
    - Advanced: TS%, USG%, ORtg, DRtg, Pace
    """
    base = _first_df(_league_dash_player_stats_safe(season, "PerGame", "Base", league))
    time.sleep(REQUEST_PAUSE)
    adv  = _first_df(_league_dash_player_stats_safe(season, "PerGame", "Advanced", league))

    keep_base = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
        "GP", "MIN", "PTS", "AST", "REB", "STL", "BLK", "TOV",
        "FG_PCT", "FG3_PCT", "FT_PCT",
    ]
    keep_adv = [
        "PLAYER_ID", "PLAYER_NAME", "TS_PCT", "USG_PCT",
        "OFF_RATING", "DEF_RATING", "PACE",
    ]

    base_small = base[[c for c in keep_base if c in base.columns]].copy()
    adv_small  = adv[[c for c in keep_adv  if c in adv.columns]].copy()

    merged = pd.merge(base_small, adv_small, on=["PLAYER_ID", "PLAYER_NAME"], how="left")
    merged["SEASON"] = season

    rename_map: Dict[str, str] = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "TEAM_ID": "team_id",
        "TEAM_ABBREVIATION": "team_abbr",
        "GP": "gp",
        "MIN": "min",
        "PTS": "pts",
        "AST": "ast",
        "REB": "reb",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "FG_PCT": "fg_pct",
        "FG3_PCT": "three_pct",
        "FT_PCT": "ft_pct",
        "TS_PCT": "ts_pct",
        "USG_PCT": "usg_pct",
        "OFF_RATING": "ortg",
        "DEF_RATING": "drtg",
        "PACE": "pace",
        "SEASON": "season",
    }
    merged = merged.rename(columns=rename_map)

    num_cols = [
        "gp", "min", "pts", "ast", "reb", "stl", "blk", "tov",
        "fg_pct", "three_pct", "ft_pct", "ts_pct", "usg_pct",
        "ortg", "drtg", "pace",
    ]
    for c in num_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged


def fetch_player_shots(player_id: int, season: str, league: League) -> pd.DataFrame:
    # ShotChartDetail takes `league_id`, not `league_id_nullable`; using the
    # nullable name raises TypeError and silently yields NBA data.
    ep = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_nullable=season,
        context_measure_simple="FGA",
        league_id=league.league_id,
    )
    raw = _first_df(ep)

    rename_map = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "TEAM_ID": "team_id",
        "GAME_ID": "game_id",
        "GAME_DATE": "game_date",
        "LOC_X": "x",
        "LOC_Y": "y",
        "SHOT_MADE_FLAG": "made",
        "SHOT_ZONE_BASIC": "zone",
        "SHOT_DISTANCE": "distance",
        "PERIOD": "period",
        "SHOT_CLOCK": "shot_clock", 
    }

    have = [c for c in rename_map if c in raw.columns]
    shots = raw[have].rename(columns={k: v for k, v in rename_map.items() if k in have})

    for _, v in rename_map.items():
        if v not in shots.columns:
            shots[v] = pd.NA

    shots["season"] = season

    shots["x"] = pd.to_numeric(shots["x"], errors="coerce")
    shots["y"] = pd.to_numeric(shots["y"], errors="coerce")
    shots["made"] = pd.to_numeric(shots["made"], errors="coerce").fillna(0).astype(int)

    return shots

def out_path(stem: str, league: League) -> str:
    return os.path.join(DATA_DIR, f"{stem}{league.suffix}.parquet")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Populate data/*.parquet for a league from stats.nba.com."
    )
    ap.add_argument("--league", default="nba", choices=sorted(LEAGUES),
                    help="which league to pull (default: nba)")
    ap.add_argument("--season", default=None,
                    help="season string; defaults to the league's 2024 season "
                         "(NBA 2024-25, WNBA 2024)")
    ap.add_argument("--demo-player", default=None,
                    help="player used for the sample shot chart")
    ap.add_argument("--skip-shots", action="store_true",
                    help="skip the shot-chart pull (the slowest, flakiest step)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    league = LEAGUES[args.league]
    season = args.season or league.season(DEFAULT_SEASON_YEAR)
    demo_player = args.demo_player or DEMO_PLAYERS.get(league.key)

    print(f"[{league.label}] season={season} league_id={league.league_id}")

    teams_master = fetch_teams_master(league)
    teams_master.to_parquet(out_path("teams_master", league), index=False)
    print(f"[{league.label}] teams_master: {len(teams_master)} teams")
    time.sleep(REQUEST_PAUSE)

    all_yby: List[pd.DataFrame] = []
    for _, t in teams_master.iterrows():
        try:
            yby = fetch_team_year_by_year(team_id=t["id"], league=league)
            yby["TEAM_ID"] = t["id"]
            yby["TEAM_NAME"] = t["full_name"]
            all_yby.append(yby)
            time.sleep(REQUEST_PAUSE)
        except Exception as e:
            print(f"team y/y failed for {t.get('full_name', t.get('id'))}: {e}")

    if all_yby:
        teams = pd.concat(all_yby, ignore_index=True)
        for need in ["TEAM_ID", "TEAM_NAME", "YEAR", "WINS", "LOSSES"]:
            if need not in teams.columns:
                teams[need] = pd.NA

        teams = teams.rename(columns={
            "TEAM_ID": "team_id",
            "TEAM_NAME": "team_name",
            "YEAR": "season",
            "WINS": "wins",
            "LOSSES": "losses",
        })
        teams.to_parquet(out_path("teams", league), index=False)
        print(f"[{league.label}] teams: {len(teams)} team-seasons")
    else:
        print("warning: no team y/y data assembled")

    players = fetch_players_base_adv(season, league)
    players.to_parquet(out_path("players", league), index=False)
    print(f"[{league.label}] players: {len(players)} rows for {season}")
    time.sleep(REQUEST_PAUSE)

    if args.skip_shots:
        print(f"[{league.label}] skipping shots")
        return

    match = find_player(demo_player, league)
    if not match:
        raise SystemExit(f"Could not find {league.label} player: {demo_player}")
    shots = fetch_player_shots(match.id, season, league)
    shots.to_parquet(out_path("shots", league), index=False)
    print(f"[{league.label}] shots: {len(shots)} attempts for {match.full_name}")


if __name__ == "__main__":
    main()
