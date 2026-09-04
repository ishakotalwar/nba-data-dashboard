"""Team analytics: one team's history, a whole season's league table,
head-to-head team-season comparison, and all-time leaderboards."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

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


# A group smaller than a five is every five that contains it, added up.
GROUP_SIZES = [2, 3, 4, 5]


def _groups_of(sub: pd.DataFrame, size: int) -> pd.DataFrame:
    """Roll five-man rows up into every group of `size` inside them.

    Each possession belongs to exactly one five, so a pair's minutes are the
    minutes of every five containing that pair — no double counting, and the
    totals stay exact. Games cannot be carried across: two fives may share a
    game, and the stored counts would add it twice, so the column is dropped
    rather than reported wrongly.
    """
    totals: dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    names: dict = {}
    teams: dict = {}
    for r in sub.itertuples():
        ids = [int(i) for i in r.player_ids.split("|")]
        for pid, name in zip(ids, r.player_names.split("|")):
            names[pid] = name
        for group in combinations(sorted(ids), size):
            key = (r.team_id, group)
            teams[key] = (r.team_name, r.team_abbr, r.team_min)
            slot = totals[key]
            slot[0] += r.min
            slot[1] += r.poss
            slot[2] += r.pts_for
            slot[3] += r.pts_against
            slot[4] += r.stints
    rows = []
    for (team_id, group), (mins, poss, pf, pa, stints) in totals.items():
        team_name, team_abbr, team_min = teams[(team_id, group)]
        rows.append({
            "season": sub["season"].iloc[0],
            "team_id": team_id,
            "team_name": team_name,
            "team_abbr": team_abbr,
            "player_ids": "|".join(str(p) for p in group),
            "player_names": "|".join(names[p] for p in group),
            "games": None,
            "stints": int(stints),
            "min": round(mins, 1),
            "team_min": team_min,
            "poss": round(poss, 1),
            "pts_for": int(pf),
            "pts_against": int(pa),
        })
    return pd.DataFrame(rows)


@router.get("/lineups")
def team_lineups(season: str, team: str | None = None, league: str | None = None,
                 size: int = 5, min_minutes: float = 50, limit: int = 250):
    """The best groups of players for one season, league-wide or for one team.

    Rebuilt from play-by-play substitutions by `etl/lineup_etl.py` — the box
    scores say what a player did, only the substitutions say who they did it
    alongside. A `size` below five rolls the fives up into the pairs, trios or
    quartets inside them.
    """
    lg = leagues.get(league)
    if size not in GROUP_SIZES:
        raise HTTPException(400, f"size must be one of {GROUP_SIZES}")
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

    if size < 5:
        sub = _groups_of(sub, size)

    qualified = sub[sub["min"] >= float(min_minutes)]
    ordered = qualified.sort_values("min", ascending=False).head(limit)
    rows = _lineup_rows(ordered)
    return analytics.json_safe({
        "season": str(season), "league": lg.key, "team": team,
        "size": size,
        "min_minutes": float(min_minutes),
        "rows": rows,
        # The floor hides lineups, and the ETL's own floor hides more. Both
        # shares are reported rather than left for the reader to wonder about.
        "shown_minutes": float(qualified["min"].sum()),
        "team_minutes": team_total,
        "lineups_total": int(len(sub)),
    })


# More than this and the table is 32 rows of mostly-empty combinations.
MAX_WOWY_PLAYERS = 4

# A combination has to have been on the floor this long to be worth a row.
MIN_COMBINATION_MINUTES = 1.0


def _rates(label: str, rows: pd.DataFrame, on: list[str], off: list[str]) -> dict:
    """One combination of players on the floor, as totals and rates."""
    poss = float(rows["poss"].sum())
    pf, pa = int(rows["pts_for"].sum()), int(rows["pts_against"].sum())
    net = (100 * (pf - pa) / poss) if poss > 0 else None
    return {
        "label": label,
        "on": on,
        "off": off,
        "min": round(float(rows["min"].sum()), 1),
        "poss": round(poss, 1),
        "pts_for": pf,
        "pts_against": pa,
        "ortg": round(100 * pf / poss, 1) if poss > 0 else None,
        "drtg": round(100 * pa / poss, 1) if poss > 0 else None,
        "net": None if net is None else round(net, 1),
    }


def _surname(name: str) -> str:
    """Everything after the first name, so "Jaren Jackson Jr." keeps its suffix.
    Combination labels stack up to four names and a full one each would run off
    the row."""
    first, _, rest = name.partition(" ")
    return rest or first


def _combination_label(on: list[str], off: list[str]) -> str:
    """Who was on the floor. Naming both halves — "A without B + C" — reads as
    if the absent names qualify the present one, so the label carries only the
    players who were out there and the table marks the rest as off."""
    if len(on) + len(off) == 1:
        return f"{on[0]} on" if on else f"{off[0]} off"
    if not on:
        return "None of them"
    return " + ".join(_surname(n) for n in on)


@router.get("/wowy")
def team_wowy(season: str, team: str, league: str | None = None,
              players: str | None = None):
    """How a team played with and without any group of its players.

    Every possession belongs to exactly one five, so a group of players is
    answered by adding up the lineups that contain them — which is why the
    lineup table stores every five a team used rather than only the ones that
    lasted. The combinations are exhaustive and add back to the team's season.

    Raw, deliberately: no adjustment for teammates or opponents, so it answers
    "what happened when they sat" rather than "how good are they". The Impact page
    is where the adjusted version lives.
    """
    lg = leagues.get(league)
    df = data.lineups(lg)
    if df is None:
        raise HTTPException(404, f"No {lg.label} lineup data. "
                                 f"Run `python etl/lineup_etl.py --league {lg.key}`.")
    sub = df[(df["season"] == str(season)) & (df["team_name"] == team)].copy()
    if sub.empty:
        raise HTTPException(404, f"No {lg.label} {season} lineups for {team}")

    # Who played, and how much, from the fives themselves.
    minutes: dict = {}
    names: dict = {}
    id_sets = []
    for ids, player_names, mins in zip(sub["player_ids"], sub["player_names"], sub["min"]):
        group = [int(i) for i in ids.split("|")]
        id_sets.append(set(group))
        for pid, name in zip(group, player_names.split("|")):
            names[pid] = name
            minutes[pid] = minutes.get(pid, 0.0) + float(mins)
    sub["five"] = id_sets

    roster = [{"player_id": pid, "name": names[pid], "min": round(mins, 1)}
              for pid, mins in sorted(minutes.items(), key=lambda kv: -kv[1])]
    total = _rates(team, sub, [], [])

    picked: list[int] = []
    if players:
        try:
            picked = [int(p) for p in players.split(",") if p.strip()]
        except ValueError:
            raise HTTPException(400, "players must be a comma-separated list of ids")
        unknown = [p for p in picked if p not in names]
        if unknown:
            raise HTTPException(404, f"No {team} {season} minutes for player(s): "
                                     f"{', '.join(str(u) for u in unknown)}")
        if len(picked) > MAX_WOWY_PLAYERS:
            raise HTTPException(400, f"At most {MAX_WOWY_PLAYERS} players at once — "
                                     f"beyond that the split is mostly empty rows.")

    rows: list[dict] = []
    if picked:
        # Each five falls into exactly one combination: which of the picked
        # players were part of it.
        key = sub["five"].map(lambda f: tuple(p for p in picked if p in f))
        for combination, group in sub.groupby(key, sort=False):
            on = [names[p] for p in combination]
            off = [names[p] for p in picked if p not in combination]
            entry = _rates(_combination_label(on, off), group, on, off)
            if entry["min"] >= MIN_COMBINATION_MINUTES:
                rows.append(entry)
        # Best first: the question is which combination played well, and
        # minutes are already a column for judging how much to trust it.
        rows.sort(key=lambda r: (r["net"] is not None, r["net"]), reverse=True)

    return analytics.json_safe({
        "season": str(season), "league": lg.key, "team": team,
        "players": picked,
        "max_players": MAX_WOWY_PLAYERS,
        "roster": roster,
        "team_total": total,
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
