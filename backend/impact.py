"""Box-score impact metrics, computed from the player-season table.

The possession-level metrics live in `etl/lineup_etl.py`, where the stints are.
What is here needs no play-by-play at all: PER is arithmetic over a season's
box scores and its league averages, which is both its appeal and its limit — it
cannot see defence beyond steals and blocks, and it cannot see who a player was
on the floor with.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data
from .leagues import DEFAULT, League

# Season totals the formula needs, as they are named in the player table.
COUNTS = ["min_tot", "fgm", "fga", "fg3m", "ftm", "fta", "oreb", "dreb", "pf",
          "ast_tot", "tov_tot", "stl_tot", "blk_tot", "pts_tot", "reb_tot"]


@lru_cache(maxsize=8)
def per(league: League = DEFAULT) -> pd.DataFrame | None:
    """Hollinger's Player Efficiency Rating, per player-season.

    Follows the published formula: an unadjusted per-minute rating built from
    the box score, corrected for the team's pace, then scaled so the league
    average comes out at 15. None when the player table predates the raw counts
    the formula needs — percentages cannot be turned back into attempts.
    """
    players = data.players(league)
    if not set(COUNTS).issubset(players.columns):
        return None
    df = data.clean_numeric(players.copy(), COUNTS)
    teams = data.clean_numeric(data.teams(league).copy(), ["pace"])

    out = []
    for season, group in df.groupby("season"):
        rated = _season_per(group, teams[teams["season"] == season])
        if rated is not None:
            out.append(rated)
    if not out:
        return None
    return pd.concat(out, ignore_index=True)


def _season_per(g: pd.DataFrame, season_teams: pd.DataFrame) -> pd.DataFrame | None:
    """One season's PER. Every league term is that season's own."""
    played = g[g["min_tot"].fillna(0) > 0].copy()
    if played.empty:
        return None
    total = {c: float(played[c].sum()) for c in COUNTS}
    if total["fga"] <= 0 or total["reb_tot"] <= 0 or total["pf"] <= 0:
        return None

    # League-wide terms: the value of a possession, the share of rebounds that
    # are defensive, and the assist correction applied to made shots.
    vop = total["pts_tot"] / (total["fga"] - total["oreb"] + total["tov_tot"]
                              + 0.44 * total["fta"])
    drb_pct = (total["reb_tot"] - total["oreb"]) / total["reb_tot"]
    factor = ((2 / 3) - (0.5 * (total["ast_tot"] / total["fgm"]))
              / (2 * (total["fgm"] / total["ftm"]))) if total["ftm"] else 2 / 3

    # A player's assist credit depends on how much of the team's scoring came
    # off assists, so the team's own ratio enters the per-player term.
    team = played.groupby("team_id")[["ast_tot", "fgm"]].sum()
    team_ratio = (team["ast_tot"] / team["fgm"].where(team["fgm"] > 0)).fillna(0)
    ratio = played["team_id"].map(team_ratio).fillna(0).to_numpy(dtype=float)

    p = {c: played[c].to_numpy(dtype=float) for c in COUNTS}
    fg_miss = p["fga"] - p["fgm"]
    ft_miss = p["fta"] - p["ftm"]
    uper = (
        p["fg3m"]
        + (2 / 3) * p["ast_tot"]
        + (2 - factor * ratio) * p["fgm"]
        + p["ftm"] * 0.5 * (1 + (1 - ratio) + (2 / 3) * ratio)
        - vop * p["tov_tot"]
        - vop * drb_pct * fg_miss
        - vop * 0.44 * (0.44 + 0.56 * drb_pct) * ft_miss
        + vop * (1 - drb_pct) * p["dreb"]
        + vop * drb_pct * p["oreb"]
        + vop * p["stl_tot"]
        + vop * drb_pct * p["blk_tot"]
        - p["pf"] * ((total["ftm"] / total["pf"])
                     - 0.44 * (total["fta"] / total["pf"]) * vop)
    ) / p["min_tot"]

    # Pace correction: a slow team's players get fewer chances per minute.
    pace = season_teams.set_index("team_id")["pace"]
    league_pace = float(pace.mean()) if len(pace) else np.nan
    team_pace = played["team_id"].map(pace).to_numpy(dtype=float)
    adjusted = uper * np.where(np.isfinite(team_pace) & (team_pace > 0),
                               league_pace / team_pace, 1.0)

    # Scaled so the league's minute-weighted average lands on 15, which is what
    # makes a PER of 20 mean the same thing in any season.
    weight = p["min_tot"]
    mean = float(np.average(adjusted, weights=weight)) if weight.sum() else np.nan
    scale = 15.0 / mean if mean and np.isfinite(mean) and mean != 0 else np.nan
    return pd.DataFrame({
        "season": played["season"].astype(str).to_numpy(),
        "player_id": played["player_id"].astype("int64").to_numpy(),
        "per": np.round(adjusted * scale, 2),
    })
