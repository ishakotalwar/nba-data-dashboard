import os
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


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEASON = "2024-25"
DEMO_PLAYER = "Stephen Curry"

REQUEST_PAUSE = 0.8  # avoid too many requests to API too quickly
LEAGUE_ID = "00"     # NBA (00=NBA, 10=WNBA, 20=G League)



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


def find_player(name: str) -> Optional[PlayerIdentity]:
    hits = players_static.find_players_by_full_name(name)
    if not hits:
        return None
    return PlayerIdentity(id=hits[0]["id"], full_name=hits[0]["full_name"])


def fetch_teams_master() -> pd.DataFrame:
    return pd.DataFrame(teams_static.get_teams())


def fetch_team_year_by_year(team_id: int) -> pd.DataFrame:
    try:
        ep = teamyearbyyearstats.TeamYearByYearStats(
            team_id=team_id,
            league_id_nullable=LEAGUE_ID,
        )
    except TypeError:
        ep = teamyearbyyearstats.TeamYearByYearStats(team_id=team_id)
    return _first_df(ep)


def _league_dash_player_stats_safe(season: str, per_mode: str, measure_type: str):
    try:
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense=measure_type,
            league_id_nullable=LEAGUE_ID,
        )
    except TypeError:
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_def=measure_type,
            league_id_nullable=LEAGUE_ID,
        )


def fetch_players_base_adv(season: str) -> pd.DataFrame:
    """
    League-wide player stats for a season:
    - Base: PTS, AST, REB, FG%, etc.
    - Advanced: TS%, USG%, ORtg, DRtg, Pace
    """
    base = _first_df(_league_dash_player_stats_safe(season=season, per_mode="PerGame", measure_type="Base"))
    time.sleep(REQUEST_PAUSE)
    adv  = _first_df(_league_dash_player_stats_safe(season=season, per_mode="PerGame", measure_type="Advanced"))

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


def fetch_player_shots(player_id: int, season: str) -> pd.DataFrame:
    try:
        ep = shotchartdetail.ShotChartDetail(
            team_id=0, 
            player_id=player_id,
            season_nullable=season,
            context_measure_simple="FGA",
            league_id_nullable=LEAGUE_ID,
        )
    except TypeError:
        ep = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            context_measure_simple="FGA",
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

def main():
    teams_master = fetch_teams_master()
    teams_master.to_parquet(os.path.join(DATA_DIR, "teams_master.parquet"), index=False)
    time.sleep(REQUEST_PAUSE)
    all_yby: List[pd.DataFrame] = []
    for _, t in teams_master.iterrows():
        try:
            yby = fetch_team_year_by_year(team_id=t["id"])
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
        teams.to_parquet(os.path.join(DATA_DIR, "teams.parquet"), index=False)
    else:
        print("warning: no team y/y data assembled")
    players = fetch_players_base_adv(SEASON)
    players.to_parquet(os.path.join(DATA_DIR, "players.parquet"), index=False)
    time.sleep(REQUEST_PAUSE)

    match = find_player(DEMO_PLAYER)
    if not match:
        raise SystemExit(f"Could not find player: {DEMO_PLAYER}")
    shots = fetch_player_shots(player_id=match.id, season=SEASON)
    shots.to_parquet(os.path.join(DATA_DIR, "shots.parquet"), index=False)


if __name__ == "__main__":
    main()
