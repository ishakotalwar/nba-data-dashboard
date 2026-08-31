"""Team analytics: one team's history, a whole season's league table,
head-to-head team-season comparison, and all-time leaderboards."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import analytics, data, leagues

router = APIRouter(prefix="/api/teams", tags=["teams"])

SERIES_COLS = [
    "wins", "losses", "FG_PCT", "FT_PCT", "FG3_PCT", "ortg", "drtg", "pace",
    "FGM", "FGA", "FG3M", "FTA", "TOV", "OREB", "DREB",
]

# Metrics a leaderboard can rank on, and which direction is "best".
RANKABLE = {
    "win_pct": True, "wins": True, "ortg": True, "net": True, "pace": True,
    "eFG%": True, "ORB%": True, "FT rate": True,
    "drtg": False, "TOV%": False,
}


def _season_frame(lg, season: str | None = None) -> pd.DataFrame:
    tdf = data.teams(lg).copy()
    if season:
        tdf = tdf[tdf["season"] == str(season)]
    return tdf


def _enrich(sub: pd.DataFrame) -> pd.DataFrame:
    """Add derived rate stats used everywhere teams are ranked."""
    num = ["wins", "losses", "ortg", "drtg", "pace",
           "FGM", "FGA", "FG3M", "FTA", "TOV", "OREB", "DREB"]
    sub = data.clean_numeric(sub, num)
    ff = sub.apply(analytics.four_factors, axis=1, result_type="expand")
    out = sub[["team_name", "season", "wins", "losses", "ortg", "drtg", "pace"]].copy()
    games = sub["wins"] + sub["losses"]
    out["games"] = games
    out["win_pct"] = (sub["wins"] / games.where(games > 0)).round(3)
    out["net"] = (sub["ortg"] - sub["drtg"]).round(1)
    for k in ("eFG%", "TOV%", "ORB%", "FT rate"):
        out[k] = ff[k].round(3)
    return out


@router.get("")
def teams_list(league: str | None = None):
    lg = leagues.get(league)
    tdf = data.teams(lg)
    return {"league": lg.key, "teams": data.team_names(lg),
            "seasons": sorted(tdf["season"].astype(str).unique().tolist())}


@router.get("/series")
def team_series(team: str, league: str | None = None):
    lg = leagues.get(league)
    tdf = _season_frame(lg)
    tdf = tdf[tdf["team_name"] == team]
    if tdf.empty:
        raise HTTPException(404, "team not found")
    tdf = data.clean_numeric(tdf, SERIES_COLS).sort_values("season")
    cols = ["season"] + [c for c in SERIES_COLS if c in tdf.columns]
    return analytics.json_safe({"team": team, "rows": data.records(tdf[cols])})


@router.get("/factors")
def team_factors(team: str, season: str, league: str | None = None):
    lg = leagues.get(league)
    tdf = data.teams(lg).copy()
    needed = {"FGM", "FGA", "FG3M", "FTA", "TOV", "OREB", "DREB"}
    if not needed.issubset(set(tdf.columns)):
        raise HTTPException(400, f"Missing columns: {sorted(needed - set(tdf.columns))}")
    tdf = data.clean_numeric(tdf, list(needed))
    row = tdf[(tdf["team_name"] == team) & (tdf["season"] == str(season))]
    if row.empty:
        raise HTTPException(404, "team/season not found")
    lgf = tdf[tdf["season"] == str(season)].apply(analytics.four_factors, axis=1, result_type="expand")
    return analytics.json_safe({
        "team": team, "season": str(season),
        "team_ff": analytics.four_factors(row.iloc[0]),
        "league_avg": lgf.mean(numeric_only=True).to_dict(),
    })


@router.get("/league")
def teams_league(season: str, league: str | None = None):
    lg = leagues.get(league)
    sub = _season_frame(lg, season)
    if sub.empty:
        raise HTTPException(404, f"No {lg.label} team data for {season}")
    out = _enrich(sub).rename(columns={"team_name": "team"}).sort_values("net", ascending=False)
    avg = {c: float(pd.to_numeric(out[c], errors="coerce").mean())
           for c in ("ortg", "drtg", "net", "pace", "eFG%", "TOV%", "ORB%", "FT rate")}
    return analytics.json_safe({"season": str(season), "league": lg.key,
                                "rows": data.records(out), "league_avg": avg})


class TeamCompareRequest(BaseModel):
    a: dict  # {team, season}
    b: dict
    league: str | None = None


@router.post("/compare")
def teams_compare(req: TeamCompareRequest):
    """Two team-seasons side by side, each against its own season's average."""
    lg = leagues.get(req.league)
    out = {}
    for side, sel in (("a", req.a), ("b", req.b)):
        team, season = str(sel["team"]), str(sel["season"])
        sub = _season_frame(lg, season)
        if sub.empty:
            raise HTTPException(404, f"No {lg.label} data for {season}")
        enriched = _enrich(sub)
        row = enriched[enriched["team_name"] == team]
        if row.empty:
            raise HTTPException(404, f"{team} has no {season} season")
        r = row.iloc[0]
        avg = enriched.mean(numeric_only=True)
        out[side] = {
            "team": team, "season": season,
            "values": {k: (None if pd.isna(r[k]) else float(r[k]))
                       for k in ("wins", "losses", "win_pct", "ortg", "drtg", "net", "pace",
                                 "eFG%", "TOV%", "ORB%", "FT rate")},
            "vs_league": {k: (None if pd.isna(r[k]) or pd.isna(avg.get(k)) else round(float(r[k] - avg[k]), 3))
                          for k in ("win_pct", "ortg", "drtg", "net", "pace",
                                    "eFG%", "TOV%", "ORB%", "FT rate")},
        }
    return analytics.json_safe(out)


@router.get("/rankings")
def teams_rankings(metric: str = "net", league: str | None = None,
                   limit: int = 15, min_games: int = 20):
    """All-time leaderboard for one team metric, across every season on disk."""
    lg = leagues.get(league)
    if metric not in RANKABLE:
        raise HTTPException(400, f"Cannot rank on {metric!r}. Try: {', '.join(sorted(RANKABLE))}")
    enriched = _enrich(_season_frame(lg))
    enriched = enriched[enriched["games"].fillna(0) >= min_games]
    col = pd.to_numeric(enriched[metric], errors="coerce")
    enriched = enriched[col.notna()]
    ordered = enriched.sort_values(metric, ascending=not RANKABLE[metric]).head(limit)
    return analytics.json_safe({
        "metric": metric,
        "higher_is_better": RANKABLE[metric],
        "rows": data.records(ordered.rename(columns={"team_name": "team"})),
    })
