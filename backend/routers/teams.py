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


def _lineup_rows(df: pd.DataFrame) -> list[dict]:
    """Lineup totals -> rates, on the same per-100-possessions scale as ORtg."""
    out = df.copy()
    poss = out["poss"].where(out["poss"] > 0)
    out["ortg"] = (100 * out["pts_for"] / poss).round(1)
    out["drtg"] = (100 * out["pts_against"] / poss).round(1)
    out["net"] = (out["ortg"] - out["drtg"]).round(1)
    out["plus_minus"] = out["pts_for"] - out["pts_against"]
    # What fraction of everything this team played, so a big net rating over 80
    # minutes isn't read as a description of the season.
    out["share"] = (out["min"] / out["team_min"].where(out["team_min"] > 0)).round(4)
    # Stored in player-id order, which is meaningless to read. Sort by surname
    # so the same five looks the same everywhere it appears.
    out["players"] = [
        sorted(({"id": int(i), "name": n} for i, n in zip(ids.split("|"), names.split("|"))),
               key=lambda p: p["name"].rsplit(" ", 1)[-1])
        for ids, names in zip(out["player_ids"], out["player_names"])
    ]
    cols = ["team_name", "team_abbr", "season", "players", "games", "stints",
            "min", "share", "poss", "pts_for", "pts_against",
            "ortg", "drtg", "net", "plus_minus"]
    return data.records(out[cols].rename(columns={"team_name": "team"}))


@router.get("/lineups")
def team_lineups(season: str, team: str | None = None, league: str | None = None,
                 min_minutes: float = 50, limit: int = 250):
    """Five-man lineups for one season, either league-wide or for one team.

    Rebuilt from play-by-play substitutions by `etl/lineup_etl.py` — the box
    scores say what a player did, only the substitutions say who he did it
    alongside.
    """
    lg = leagues.get(league)
    df = data.lineups(lg)
    if df is None:
        raise HTTPException(404, f"No {lg.label} lineup data. "
                                 f"Run `python etl/lineup_etl.py --league {lg.key}`.")
    sub = df[df["season"] == str(season)]
    if sub.empty:
        raise HTTPException(404, f"No {lg.label} lineups for {season}. "
                                 f"Seasons on disk: {', '.join(data.lineup_seasons(lg)) or 'none'}")
    team_total = None
    if team:
        sub = sub[sub["team_name"] == team]
        if sub.empty:
            raise HTTPException(404, f"{team} has no {season} lineups")
        team_total = float(sub["team_min"].iloc[0])

    qualified = sub[sub["min"] >= float(min_minutes)]
    ordered = qualified.sort_values("min", ascending=False).head(limit)
    rows = _lineup_rows(ordered)
    return analytics.json_safe({
        "season": str(season), "league": lg.key, "team": team,
        "min_minutes": float(min_minutes),
        "rows": rows,
        # The floor hides lineups, and the ETL's own floor hides more. Both
        # shares are reported rather than left for the reader to wonder about.
        "shown_minutes": float(qualified["min"].sum()),
        "team_minutes": team_total,
        "lineups_total": int(len(sub)),
    })


# The row carrying a team's own totals, written by etl/lineup_etl.py.
TEAM_TOTAL_ID = 0


def _split(name: str, row: dict) -> dict:
    """One cell of the with/without square, as rates."""
    poss = row["poss"]
    net = (100 * (row["pts_for"] - row["pts_against"]) / poss) if poss > 0 else None
    return {
        "label": name,
        "min": round(row["min"], 1),
        "poss": round(poss, 1),
        "pts_for": int(row["pts_for"]),
        "pts_against": int(row["pts_against"]),
        "ortg": round(100 * row["pts_for"] / poss, 1) if poss > 0 else None,
        "drtg": round(100 * row["pts_against"] / poss, 1) if poss > 0 else None,
        "net": None if net is None else round(net, 1),
    }


def _combine(*parts) -> dict:
    """Add and subtract stored totals: (totals, sign) pairs."""
    out = {k: 0.0 for k in ("min", "poss", "pts_for", "pts_against")}
    for row, sign in parts:
        for k in out:
            out[k] += sign * (row[k] if row else 0.0)
    return out


@router.get("/wowy")
def team_wowy(season: str, team: str, league: str | None = None,
              player_a: int | None = None, player_b: int | None = None):
    """How a team played with and without one player, or a pair of them.

    Raw on-off, deliberately: no opponent or teammate adjustment, so it answers
    "what happened when he sat" rather than "how good is he". The Impact page
    is where the adjusted version lives. Built from every stint, so the splits
    add back up to the team's whole season.
    """
    lg = leagues.get(league)
    df = data.wowy(lg)
    if df is None:
        raise HTTPException(404, f"No {lg.label} with/without data. "
                                 f"Run `python etl/lineup_etl.py --league {lg.key}`.")
    sub = df[(df["season"] == str(season)) & (df["team_name"] == team)]
    if sub.empty:
        raise HTTPException(404, f"No {lg.label} {season} data for {team}")

    singles = sub[(sub["player_a"] == sub["player_b"])
                  & (sub["player_a"] != TEAM_TOTAL_ID)]
    roster = [{"player_id": int(r.player_a), "name": r.name_a, "min": round(r.min, 1)}
              for r in singles.sort_values("min", ascending=False).itertuples()]
    total = sub[sub["player_a"] == TEAM_TOTAL_ID].iloc[0].to_dict()

    def stored(a: int, b: int) -> dict | None:
        lo, hi = sorted((a, b))
        hit = sub[(sub["player_a"] == lo) & (sub["player_b"] == hi)]
        return hit.iloc[0].to_dict() if len(hit) else None

    rows: list[dict] = []
    if player_a:
        on_a = stored(player_a, player_a)
        if on_a is None:
            raise HTTPException(404, f"Player {player_a} did not play for {team} in {season}")
        name_a = on_a["name_a"]
        if player_b:
            on_b = stored(player_b, player_b)
            if on_b is None:
                raise HTTPException(404, f"Player {player_b} did not play for {team} in {season}")
            name_b = on_b["name_a"]
            both = stored(player_a, player_b)
            if both is None:
                raise HTTPException(
                    404, f"{name_a} and {name_b} never shared the floor for long enough")
            rows = [
                _split(f"{name_a} + {name_b}", _combine((both, 1))),
                _split(f"{name_a} without {name_b}", _combine((on_a, 1), (both, -1))),
                _split(f"{name_b} without {name_a}", _combine((on_b, 1), (both, -1))),
                # Everything the team played that neither of them was part of.
                _split("Neither", _combine((total, 1), (on_a, -1), (on_b, -1), (both, 1))),
            ]
        else:
            rows = [
                _split(f"{name_a} on", _combine((on_a, 1))),
                _split(f"{name_a} off", _combine((total, 1), (on_a, -1))),
            ]

    return analytics.json_safe({
        "season": str(season), "league": lg.key, "team": team,
        "roster": roster,
        "team_total": _split(team, total),
        "rows": rows,
    })


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
