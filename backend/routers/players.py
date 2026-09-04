"""Player-centric endpoints: roster, bio, one player-season, and career."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from .. import analytics, data, impact, leagues, live

router = APIRouter(prefix="/api", tags=["players"])

# Headline stats shown on the overview, in display order.
OVERVIEW_METRICS = ["pts", "reb", "ast", "fg_pct", "three_pct", "ft_pct", "ts_pct", "usg_pct"]


def _check_basis(per: str) -> None:
    if per not in data.RATE_BASES:
        raise HTTPException(400, f"Unknown rate basis {per!r}. Expected one of: "
                                 f"{', '.join(data.RATE_BASES)}")


def _player_rows(lg, player_id: int, per: str = "game") -> pd.DataFrame:
    df = data.players_at(per, lg)
    rows = df[pd.to_numeric(df["player_id"], errors="coerce") == player_id]
    if rows.empty:
        raise HTTPException(404, f"No {lg.label} player with id {player_id}")
    return rows.sort_values("season")


def _season_start(lg, season: str) -> pd.Timestamp | None:
    """First day of `season` for this league.

    A season that tips off in the second half of the calendar year is labelled
    by the year it finishes in (NBA: `2016` began October 2015); one that runs
    inside a single year is labelled by that year (WNBA: `2016` began May 2016).
    """
    start_year = lg.season_start_year(season)
    if start_year is None:
        return None
    return pd.Timestamp(year=start_year, month=lg.season_start_month, day=1)


def _bio(lg, player_id: int, season: str | None = None) -> dict:
    df = data._load_optional("player_bio", lg)
    if df is None:
        return {}
    row = df[df["player_id"].astype("int64") == player_id]
    if row.empty:
        return {}
    r = row.iloc[0]
    bd = pd.to_datetime(r.get("birthdate"), errors="coerce")
    out = {
        "height": r.get("height"),
        "weight": r.get("weight"),
        "position": r.get("position"),
        "birthdate": bd.date().isoformat() if pd.notna(bd) else None,
        "birthplace": ", ".join(
            [x for x in (r.get("birth_city"), r.get("birth_country")) if isinstance(x, str) and x]
        ) or None,
    }
    if pd.notna(bd):
        # Age during the season being viewed, not age today — otherwise every
        # historical season reports the player's present-day age.
        asof = _season_start(lg, season) if season else None
        age = int(((asof if asof is not None else pd.Timestamp.today()) - bd).days // 365.25)
        if age >= 0:
            out["age"] = age
    return out


@router.get("/players")
def players(league: str | None = None, q: str = "", limit: int = 50):
    """Roster for a league, optionally filtered — id, name and latest team."""
    lg = leagues.get(league)
    df = data.players(lg)
    latest = df.sort_values("season").drop_duplicates("player_name", keep="last")
    if q:
        latest = latest[latest["player_name"].str.contains(q, case=False, na=False)]
    latest = latest.sort_values("player_name").head(limit)
    return analytics.json_safe({
        "league": lg.key,
        "results": [
            {"player_id": int(r.player_id), "name": r.player_name,
             "team": getattr(r, "team_abbr", None), "last_season": r.season}
            for r in latest.itertuples()
        ],
    })


@router.get("/player/{player_id}")
def player(player_id: int, league: str | None = None):
    """Bio plus the seasons this player actually has data for."""
    lg = leagues.get(league)
    rows = _player_rows(lg, player_id)
    return analytics.json_safe({
        "player_id": player_id,
        "name": str(rows.iloc[-1]["player_name"]),
        "seasons": rows["season"].astype(str).tolist(),
        "teams": rows[["season", "team_abbr"]].to_dict("records"),
        "bio": _bio(lg, player_id),
    })


@router.get("/player/{player_id}/season/{season}")
def player_season(player_id: int, season: str, league: str | None = None,
                  per: str = "game"):
    """One player-season: headline stats with percentile and league rank."""
    lg = leagues.get(league)
    _check_basis(per)
    rows = _player_rows(lg, player_id, per)
    row = rows[rows["season"] == str(season)]
    if row.empty:
        raise HTTPException(404, f"No {lg.label} season {season} for player {player_id}")
    r = row.iloc[0]

    pool = analytics.gp_filtered_pool(analytics.season_pool(lg, season, per))
    available = [m for m in OVERVIEW_METRICS if m in rows.columns]
    stats = []
    for m in available:
        v = pd.to_numeric(pd.Series([r.get(m)]), errors="coerce").iloc[0]
        pct = analytics.percentile_series(pool, m)
        name_match = pool["player_name"] == r["player_name"]
        pctile = float(pct[name_match].iloc[0]) * 100 if name_match.any() else None
        stats.append({
            "metric": m,
            "value": None if pd.isna(v) else float(v),
            "percentile": None if pctile is None or pd.isna(pctile) else round(pctile, 1),
            "rank": analytics.rank_in_season(pool, m, None if pd.isna(v) else float(v)),
        })

    return analytics.json_safe({
        "player_id": player_id,
        "name": r["player_name"],
        "season": str(season),
        "team": r.get("team_abbr"),
        "gp": r.get("gp"),
        "min": r.get("min"),
        "per": per,
        "pool_size": int(len(pool)),
        "bio": _bio(lg, player_id, str(season)),
        "stats": stats,
    })


@router.get("/player/{player_id}/career")
def player_career(player_id: int, league: str | None = None, recent: int = 10,
                  per: str = "game"):
    """Season-by-season rows for the trend chart, plus the most recent games.

    The per-game log at the end is always per game: a single game has no rate
    to restate it on.
    """
    lg = leagues.get(league)
    _check_basis(per)
    rows = _player_rows(lg, player_id, per)
    mets = data.available_metrics(lg)
    cols = ["season", "team_abbr", "gp", "min"] + [m for m in mets if m in rows.columns]
    career = rows[[c for c in cols if c in rows.columns]].copy()

    games: list[dict] = []
    glog = data.gamelog(lg)
    if glog is not None:
        g = glog[pd.to_numeric(glog["player_id"], errors="coerce") == player_id]
        if not g.empty:
            g = g.sort_values("GAME_DATE").tail(recent)
            for _, x in g.iterrows():
                games.append({
                    "date": x["GAME_DATE"].isoformat() if pd.notna(x["GAME_DATE"]) else None,
                    "season": x.get("season"),
                    "matchup": x.get("MATCHUP"),
                    "min": x.get("MIN"), "pts": x.get("PTS"), "reb": x.get("REB"),
                    "ast": x.get("AST"), "stl": x.get("STL"), "blk": x.get("BLK"),
                    "tov": x.get("TOV"), "fg_pct": x.get("FG_PCT"), "three_pct": x.get("FG3_PCT"),
                })

    return analytics.json_safe({
        "player_id": player_id,
        "name": str(rows.iloc[-1]["player_name"]),
        "metrics": [m for m in mets if m in rows.columns],
        "seasons": data.records(career),
        "recent_games": games,
    })


@router.get("/player-search")
def player_search(q: str = "", limit: int = 12, league: str | None = None):
    """Static roster search from nba_api's bundled data (names and ids only)."""
    return {"results": live.search_players(q, limit=limit, league=leagues.get(league))}


@router.get("/players/ratings")
def player_ratings(season: str, league: str | None = None, team: str | None = None,
                   season_type: str = "regular", min_share: float = 0.55,
                   metric: str = "rapm", limit: int = 250):
    """Impact ratings for one season, best first.

    Each player's number is the points per 100 possessions he is responsible for
    once the other nine on the floor are regressed out, split into the offensive
    and defensive halves and the possession outcomes underneath them — see
    `etl/lineup_etl.py`.

    The floor is a share of his team's possessions rather than a count of them,
    because a count means different things in different leagues and in a
    postseason. It matters more than a floor usually does: a rating over a
    fraction of a season is a real estimate with real uncertainty, and part-time
    players whose stints happened to go well are exactly who floats to the top
    of a leaderboard that lets them in. Percentiles are against everyone who
    clears the same bar, so they move with it.
    """
    lg = leagues.get(league)
    df = data.ratings(lg)
    if df is None:
        raise HTTPException(404, f"No {lg.label} rating data. "
                                 f"Run `python etl/lineup_etl.py --league {lg.key}`.")
    if metric not in data.IMPACT_METRICS:
        raise HTTPException(400, f"Unknown metric {metric!r}. Expected one of: "
                                 f"{', '.join(data.IMPACT_METRICS)}")
    if season_type not in data.RATING_SEASON_TYPES:
        raise HTTPException(400, f"Unknown season type {season_type!r}. Expected one "
                                 f"of: {', '.join(data.RATING_SEASON_TYPES)}")
    sub = df[(df["season"] == str(season)) & (df["season_type"] == season_type)].copy()
    if sub.empty:
        raise HTTPException(
            404, f"No {lg.label} {season_type} ratings for {season}. Seasons on disk: "
                 f"{', '.join(data.rating_seasons(lg, season_type)) or 'none'}")

    # PER is box-score arithmetic and lives outside the stint data, so it is
    # joined on rather than stored beside the fitted numbers.
    box = impact.per(lg)
    if box is not None:
        sub = sub.merge(box[box["season"] == str(season)], on=["season", "player_id"],
                        how="left")

    # Five players are on the floor for each of a team's possessions, so its
    # players' possessions sum to five times the team's own.
    poss = pd.to_numeric(sub["poss"], errors="coerce").fillna(0)
    team_poss = sub.assign(poss=poss).groupby("team_id")["poss"].transform("sum") / 5
    sub["share"] = (poss / team_poss.where(team_poss > 0)).round(4)
    if team:
        sub = sub[sub["team_name"] == team]

    qualified = sub[sub["share"].fillna(0) >= float(min_share)].copy()
    ranked = data.IMPACT_METRICS[metric]["column"]
    for col in set(data.RATING_COLUMNS) | {ranked}:
        if col in qualified.columns:
            qualified[f"pct_{col}"] = (
                pd.to_numeric(qualified[col], errors="coerce").rank(pct=True) * 100
            ).round(0)
    if ranked not in qualified.columns:
        raise HTTPException(404, f"No {data.IMPACT_METRICS[metric]['label']} on disk "
                                 f"for {lg.label} {season}. Re-run the lineup ETL.")
    ordered = qualified.sort_values(ranked, ascending=False, na_position="last").head(limit)
    return analytics.json_safe({
        "season": str(season), "league": lg.key, "team": team,
        "season_type": season_type,
        "metric": metric,
        "metric_column": ranked,
        "min_share": float(min_share),
        "qualified": int(len(qualified)),
        "pool": int(len(sub)),
        "rows": data.records(ordered),
    })
