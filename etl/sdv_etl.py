"""Unified NBA + WNBA ETL backed by sportsdataverse Parquet releases.

stats.nba.com drops requests to its /stats/ API from many IPs -- the TCP and TLS
handshakes succeed and then nothing comes back -- which makes `nba_etl.py`
unusable on those networks. sportsdataverse publishes ESPN-sourced data as plain
Parquet on GitHub (hoopR for the NBA, wehoop for the WNBA), so there is no API to
be blocked by. Both repos share a layout, so one script covers both leagues.

Outputs, per league, into data/ with the league's suffix:
    players_*      season aggregates      (Compare, Trends, Percentiles, Similar)
    teams_*        team-season totals     (Teams)
    teams_master_* franchise list
    gamelog_*      per-game player rows   (Game Log)
    shots_*        shot coordinates       (Shot Chart)
    player_bio_*   birthdates             (Age Curves)

Usage:
    python etl/sdv_etl.py --league wnba
    python etl/sdv_etl.py --league nba --seasons 2015-2026
    python etl/sdv_etl.py --league nba --shot-seasons 2024-2026   # smaller pull
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Iterable

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

from backend.leagues import LEAGUES, League  # noqa: E402  (needs ROOT on sys.path)

# league key -> (github repo, path segment inside it)
REPOS = {
    "nba": ("sportsdataverse/hoopR-nba-data", "nba"),
    "wnba": ("sportsdataverse/wehoop-wnba-data", "wnba"),
}

REGULAR_SEASON = 2  # ESPN season_type
FIRST_SEASON = 2003
TIMEOUT = 120

# ESPN files All-Star games under season_type 2 alongside real regular-season
# games ("Team USA" vs "Team WNBA", "TEAM CLARK", "EAST"/"WEST", ...). Those
# squads play one game, so a minimum game count per team-season drops them
# without hardcoding names or ids.
MIN_TEAM_GAMES = 10

# Shot coordinates: `coordinate_x` runs along the court's length from centre
# (+-47 ft) and `coordinate_y` across its width (+-25 ft). Both leagues play on
# a 94 ft floor with the rim 5.25 ft off the baseline, so the hoop sits at
# |coordinate_x| = 47 - 5.25.
HOOP_X = 41.75
# Free throws carry a placeholder coordinate rather than a real location.
FT_SENTINEL = (25.0, 13.75)


def url_for(league: League, dataset: str, season: int) -> str:
    repo, seg = REPOS[league.key]
    return (f"https://raw.githubusercontent.com/{repo}/main/{seg}"
            f"/{dataset}/parquet/{dataset}_{season}.parquet")


def fetch_parquet(league: League, dataset: str, season: int) -> pd.DataFrame | None:
    """Download one season of a dataset; None if that season isn't published."""
    r = requests.get(url_for(league, dataset, season), timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def load_seasons(league: League, dataset: str, seasons: Iterable[int]) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        df = fetch_parquet(league, dataset, yr)
        if df is None:
            continue
        print(f"  {dataset} {yr}: {len(df):>7} rows")
        frames.append(df)
    if not frames:
        raise SystemExit(f"No {dataset} data downloaded for {league.label}.")
    return pd.concat(frames, ignore_index=True)


def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise ratio, NaN where the denominator is zero."""
    return num.div(den.where(den > 0))


def real_franchise_games(tb: pd.DataFrame) -> set:
    """game_ids in which both sides are actual franchises (no All-Star squads)."""
    reg = tb[tb["season_type"] == REGULAR_SEASON]
    played = reg.groupby(["season", "team_id"])["game_id"].nunique()
    valid = played[played >= MIN_TEAM_GAMES].index
    is_real = pd.MultiIndex.from_arrays([reg["season"], reg["team_id"]]).isin(valid)
    sides = reg[is_real].groupby("game_id")["team_id"].nunique()
    return set(sides[sides == 2].index)


PLAYER_NUM_COLS = [
    "minutes", "points", "assists", "rebounds", "steals", "blocks", "turnovers",
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
]


def _regular_player_rows(pb: pd.DataFrame, games: set) -> pd.DataFrame:
    p = pb[(pb["season_type"] == REGULAR_SEASON)
           & (~pb["did_not_play"].fillna(False))
           & (pb["game_id"].isin(games))
           & (pb["athlete_id"].notna())].copy()
    for c in PLAYER_NUM_COLS:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    return p


def build_players(p: pd.DataFrame) -> pd.DataFrame:
    """Per-player, per-season averages in the dashboard's schema."""
    keys = ["season", "athlete_id", "athlete_display_name"]
    tot = p.groupby(keys, as_index=False).agg(
        gp=("game_id", "nunique"),
        min_tot=("minutes", "sum"), pts_tot=("points", "sum"),
        ast_tot=("assists", "sum"), reb_tot=("rebounds", "sum"),
        stl_tot=("steals", "sum"), blk_tot=("blocks", "sum"),
        tov_tot=("turnovers", "sum"),
        fgm=("field_goals_made", "sum"), fga=("field_goals_attempted", "sum"),
        fg3m=("three_point_field_goals_made", "sum"),
        fg3a=("three_point_field_goals_attempted", "sum"),
        ftm=("free_throws_made", "sum"), fta=("free_throws_attempted", "sum"),
    )

    # Team of record: the team a player logged the most games for that season.
    tm = (p.groupby(["season", "athlete_id", "team_id", "team_abbreviation"], as_index=False)
            .agg(g=("game_id", "nunique"))
            .sort_values("g", ascending=False)
            .drop_duplicates(["season", "athlete_id"]).drop(columns="g"))
    tot = tot.merge(tm, on=["season", "athlete_id"], how="left")

    team_tot = p.groupby(["season", "team_id"], as_index=False).agg(
        tm_min=("minutes", "sum"), tm_fga=("field_goals_attempted", "sum"),
        tm_fta=("free_throws_attempted", "sum"), tm_tov=("turnovers", "sum"),
    )
    tot = tot.merge(team_tot, on=["season", "team_id"], how="left")

    out = pd.DataFrame({
        "player_id": tot["athlete_id"].astype("int64"),
        "player_name": tot["athlete_display_name"],
        "team_id": tot["team_id"], "team_abbr": tot["team_abbreviation"],
        "season": tot["season"].astype(str), "gp": tot["gp"],
    })
    for name, col in [("min", "min_tot"), ("pts", "pts_tot"), ("ast", "ast_tot"),
                      ("reb", "reb_tot"), ("stl", "stl_tot"), ("blk", "blk_tot"),
                      ("tov", "tov_tot")]:
        out[name] = (tot[col] / tot["gp"]).round(2)

    out["fg_pct"] = _rate(tot["fgm"], tot["fga"]).round(3)
    out["three_pct"] = _rate(tot["fg3m"], tot["fg3a"]).round(3)
    out["ft_pct"] = _rate(tot["ftm"], tot["fta"]).round(3)
    # True shooting: PTS / (2 * (FGA + 0.44 * FTA))
    out["ts_pct"] = _rate(tot["pts_tot"], 2 * (tot["fga"] + 0.44 * tot["fta"])).round(3)
    # Usage: share of team plays a player uses while on the floor.
    plays = tot["fga"] + 0.44 * tot["fta"] + tot["tov_tot"]
    tm_plays = tot["tm_fga"] + 0.44 * tot["tm_fta"] + tot["tm_tov"]
    out["usg_pct"] = _rate(plays * (tot["tm_min"] / 5.0), tot["min_tot"] * tm_plays).round(3)
    return out.sort_values(["season", "player_name"]).reset_index(drop=True)


def build_gamelog(p: pd.DataFrame) -> pd.DataFrame:
    """Per-game rows using nba_api's column names, so the API layer is unchanged."""
    g = pd.DataFrame({
        "player_id": p["athlete_id"].astype("int64"),
        "player_name": p["athlete_display_name"],
        "season": p["season"].astype(str),
        "GAME_DATE": pd.to_datetime(p["game_date"], errors="coerce"),
        "MATCHUP": p["team_abbreviation"].astype(str)
                   + p["home_away"].map({"home": " vs ", "away": " @ "}).fillna(" vs ")
                   + p["opponent_team_abbreviation"].astype(str),
        "MIN": p["minutes"], "PTS": p["points"], "REB": p["rebounds"],
        "AST": p["assists"], "STL": p["steals"], "BLK": p["blocks"],
        "TOV": p["turnovers"],
    })
    g["FG_PCT"] = _rate(p["field_goals_made"], p["field_goals_attempted"]).round(3)
    g["FG3_PCT"] = _rate(p["three_point_field_goals_made"],
                         p["three_point_field_goals_attempted"]).round(3)
    return g.sort_values(["player_id", "GAME_DATE"]).reset_index(drop=True)


def build_teams(tb: pd.DataFrame, games: set) -> pd.DataFrame:
    """Per-team, per-season totals plus ratings, in the dashboard's schema."""
    t = tb[(tb["season_type"] == REGULAR_SEASON) & (tb["game_id"].isin(games))].copy()
    for c in ["field_goals_made", "field_goals_attempted",
              "three_point_field_goals_made", "three_point_field_goals_attempted",
              "free_throws_made", "free_throws_attempted", "offensive_rebounds",
              "defensive_rebounds", "turnovers", "team_score", "opponent_team_score"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")

    # Each game has one row per side; join a game's two rows for opponent totals.
    opp = t[["game_id", "team_id", "field_goals_attempted", "free_throws_attempted",
             "turnovers", "offensive_rebounds"]].rename(columns={
        "team_id": "opp_id", "field_goals_attempted": "opp_fga",
        "free_throws_attempted": "opp_fta", "turnovers": "opp_tov",
        "offensive_rebounds": "opp_oreb"})
    t = t.merge(opp, on="game_id")
    t = t[t["team_id"] != t["opp_id"]]

    agg = t.groupby(["season", "team_id", "team_display_name"], as_index=False).agg(
        games=("game_id", "nunique"), wins=("team_winner", "sum"),
        FGM=("field_goals_made", "sum"), FGA=("field_goals_attempted", "sum"),
        FG3M=("three_point_field_goals_made", "sum"),
        FG3A=("three_point_field_goals_attempted", "sum"),
        FTM=("free_throws_made", "sum"), FTA=("free_throws_attempted", "sum"),
        OREB=("offensive_rebounds", "sum"), DREB=("defensive_rebounds", "sum"),
        TOV=("turnovers", "sum"), pts=("team_score", "sum"),
        opp_pts=("opponent_team_score", "sum"), opp_fga=("opp_fga", "sum"),
        opp_fta=("opp_fta", "sum"), opp_tov=("opp_tov", "sum"),
        opp_oreb=("opp_oreb", "sum"))
    agg["wins"] = agg["wins"].astype(int)
    agg["losses"] = agg["games"] - agg["wins"]

    # Possessions (Oliver): FGA + 0.44*FTA - OREB + TOV, averaged with the
    # opponent's estimate to smooth the two sides out.
    poss = agg["FGA"] + 0.44 * agg["FTA"] - agg["OREB"] + agg["TOV"]
    opp_poss = agg["opp_fga"] + 0.44 * agg["opp_fta"] - agg["opp_oreb"] + agg["opp_tov"]
    avg_poss = 0.5 * (poss + opp_poss)

    return pd.DataFrame({
        "team_id": agg["team_id"], "team_name": agg["team_display_name"],
        "season": agg["season"].astype(str),
        "wins": agg["wins"], "losses": agg["losses"],
        "FGM": agg["FGM"], "FGA": agg["FGA"], "FG3M": agg["FG3M"], "FTA": agg["FTA"],
        "TOV": agg["TOV"], "OREB": agg["OREB"], "DREB": agg["DREB"],
        "FG_PCT": _rate(agg["FGM"], agg["FGA"]).round(3),
        "FG3_PCT": _rate(agg["FG3M"], agg["FG3A"]).round(3),
        "FT_PCT": _rate(agg["FTM"], agg["FTA"]).round(3),
        "ortg": (100 * _rate(agg["pts"], avg_poss)).round(1),
        "drtg": (100 * _rate(agg["opp_pts"], avg_poss)).round(1),
        "pace": _rate(avg_poss, agg["games"]).round(1),
    }).sort_values(["season", "team_name"]).reset_index(drop=True)


def build_shots(sh: pd.DataFrame) -> pd.DataFrame:
    """Shot locations in the dashboard's court frame: hoop at (0, 0), tenths of
    a foot, y increasing toward half court."""
    # Team events (e.g. team rebounds) carry no athlete, so drop those too.
    s = sh.dropna(subset=["coordinate_x", "coordinate_y", "athlete_id_1"]).copy()
    # Drop free throws, which carry a placeholder location rather than a real one.
    s = s[~((s["coordinate_x_raw"] == FT_SENTINEL[0])
            & (s["coordinate_y_raw"] == FT_SENTINEL[1]))]
    along = s["coordinate_x"].astype(float)   # court length, +-47 ft
    across = s["coordinate_y"].astype(float)  # court width, +-25 ft
    # Fold both ends onto one half; flip width with the end so left/right holds.
    side = along.map(lambda v: -1.0 if v < 0 else 1.0)
    out = pd.DataFrame({
        "player_id": s["athlete_id_1"].astype("int64"),
        "player_name": s["athlete_name_1"],
        "season": s["season"].astype(str),
        "x": (across * side * 10.0).round(1),
        "y": ((HOOP_X - along.abs()) * 10.0).round(1),
        "made": s["scoring_play"].fillna(False).astype(int),
    })
    # Keep what lands on the half court the UI draws.
    return out[(out["x"].between(-250, 250)) & (out["y"].between(-52.5, 417.5))].reset_index(drop=True)


def build_bios(core: pd.DataFrame, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per player: birthdate plus the bio fields the overview shows."""
    c = core.dropna(subset=["athlete_id"]).copy()
    c["birthdate"] = pd.to_datetime(c.get("date_of_birth"), errors="coerce", utc=True).dt.tz_localize(None)
    # player_core carries both numeric `height`/`weight` and the formatted
    # `display_*` pair; drop the numeric ones so the rename below can't collide.
    c = c.drop(columns=[x for x in ("height", "weight") if x in c.columns])
    keep = {"athlete_id": "player_id", "display_name": "player_name",
            "display_height": "height", "display_weight": "weight",
            "birth_city": "birth_city", "birth_country": "birth_country"}
    have = {k: v for k, v in keep.items() if k in c.columns}
    bio = (c.sort_values("season")
             .drop_duplicates("athlete_id", keep="last")
             .rename(columns=have)[list(have.values()) + ["birthdate"]]
             .astype({"player_id": "int64"}).reset_index(drop=True))

    # Position only exists on the box scores; take the most recent one played.
    if rows is not None and "athlete_position_name" in rows.columns:
        pos = (rows.dropna(subset=["athlete_position_name"])
                   .sort_values(["season", "game_date"])
                   .drop_duplicates("athlete_id", keep="last")
                   [["athlete_id", "athlete_position_name"]]
                   .rename(columns={"athlete_id": "player_id",
                                    "athlete_position_name": "position"}))
        pos["player_id"] = pos["player_id"].astype("int64")
        bio = bio.merge(pos, on="player_id", how="left")
    return bio


def parse_seasons(spec: str | None) -> list[int]:
    latest = pd.Timestamp.today().year
    if not spec:
        return list(range(FIRST_SEASON, latest + 1))
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def write(df: pd.DataFrame, stem: str, league: League, note: str) -> None:
    path = os.path.join(DATA_DIR, f"{stem}{league.suffix}.parquet")
    df.to_parquet(path, index=False)
    print(f"wrote {os.path.basename(path):<26} {len(df):>7} rows  {note}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", default="wnba", choices=sorted(REPOS))
    ap.add_argument("--seasons", default=None,
                    help=f"season or inclusive range (default: {FIRST_SEASON}-present)")
    ap.add_argument("--shot-seasons", default=None,
                    help="seasons to pull shot coordinates for (default: all of "
                         "--seasons; narrow this to trade shot history for a "
                         "smaller download and a smaller file)")
    args = ap.parse_args(argv)

    league = LEAGUES[args.league]
    seasons = parse_seasons(args.seasons)
    # Shots default to the full range: a shorter one silently drops seasons the
    # shot chart was showing, since each run rewrites the whole file.
    shot_seasons = parse_seasons(args.shot_seasons) if args.shot_seasons else seasons
    repo = REPOS[league.key][0]
    print(f"[{league.label}] {repo}, seasons {seasons[0]}-{seasons[-1]}")

    print("player box scores…")
    pb = load_seasons(league, "player_box", seasons)
    print("team box scores…")
    tb = load_seasons(league, "team_box", seasons)

    games = real_franchise_games(tb)
    print(f"kept {len(games)} regular-season games "
          f"({tb['game_id'].nunique() - len(games)} All-Star/exhibition/postseason dropped)")
    rows = _regular_player_rows(pb, games)

    players = build_players(rows)
    write(players, "players", league,
          f"{players['player_name'].nunique()} players, {players['season'].nunique()} seasons")
    teams = build_teams(tb, games)
    write(teams, "teams", league, f"{teams['team_name'].nunique()} franchises")
    master = (teams[["team_id", "team_name"]].drop_duplicates("team_id")
              .rename(columns={"team_name": "full_name"}).reset_index(drop=True))
    write(master, "teams_master", league, "")
    write(build_gamelog(rows), "gamelog", league, "per-game rows")

    print(f"shot coordinates ({shot_seasons[0]}-{shot_seasons[-1]})…")
    try:
        shots = build_shots(load_seasons(league, "shots", shot_seasons))
        write(shots, "shots", league, f"{shots['season'].nunique()} seasons")
    except SystemExit as e:
        print(f"  skipped: {e}")

    print("player bios…")
    try:
        bios = build_bios(load_seasons(league, "player_core", seasons), rows)
        write(bios, "player_bio", league, "birthdates")
    except SystemExit as e:
        print(f"  skipped: {e}")


if __name__ == "__main__":
    main()
