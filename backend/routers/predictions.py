"""Forecasting endpoints — team game outcomes and player season projections.

Everything here is arithmetic over local Parquet (see `backend/predictions.py`).
Each model reports its own backtest alongside its output, so the UI can show
how well it actually does rather than asking to be believed.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import analytics, data, leagues, predictions

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("/teams")
def teams(league: str | None = None):
    """Power ratings for every team, plus how the model scored historically."""
    lg = leagues.get(league)
    if data.gamelog(lg) is None:
        raise HTTPException(404, f"No {lg.label} game logs on disk to rate teams from.")
    ratings = predictions.team_ratings(lg)
    games = predictions.games(lg)
    return analytics.json_safe({
        "league": lg.key,
        "season_format": lg.season_format,
        "ratings": ratings,
        "backtest": predictions.team_backtest(lg),
        "calibration": predictions.calibration(lg),
        "games_used": int(len(games)),
        "through": str(games.GAME_DATE.max().date()) if len(games) else None,
        "home_advantage_elo": predictions.HOME_ADVANTAGE,
        "points_per_elo": round(predictions.points_per_elo(lg), 1),
    })


class MatchupRequest(BaseModel):
    home: str
    away: str
    league: str | None = None


@router.post("/matchup")
def matchup(req: MatchupRequest):
    """Win probability and projected margin for one hypothetical game."""
    lg = leagues.get(req.league)
    if req.home == req.away:
        raise HTTPException(400, "Pick two different teams.")
    try:
        return analytics.json_safe({**predictions.predict_game(lg, req.home, req.away),
                                    "league": lg.key})
    except KeyError as e:
        raise HTTPException(404, f"No {lg.label} rating for team {e.args[0]!r}.") from e


@router.get("/players")
def players(league: str | None = None, metric: str = "pts", limit: int = 25,
            order: str = "desc"):
    """Projected leaderboard for next season, and the projection's own error."""
    lg = leagues.get(league)
    if metric not in predictions.PROJECTED_METRICS and not metric.endswith("_delta"):
        raise HTTPException(
            400,
            f"Cannot project {metric!r}. Try: {', '.join(predictions.PROJECTED_METRICS)}",
        )
    if order not in ("asc", "desc"):
        raise HTTPException(400, "order must be 'asc' or 'desc'")

    rows = predictions.projected_leaderboard(lg, metric=metric,
                                             limit=min(limit, 100), order=order)
    base_metric = metric[:-6] if metric.endswith("_delta") else metric
    return analytics.json_safe({
        "league": lg.key,
        "season_format": lg.season_format,
        "metric": metric,
        "rows": rows,
        "metrics": list(predictions.PROJECTED_METRICS),
        "accuracy": predictions.player_backtest(lg, base_metric, seasons_back=3),
        "model": "marcel-style weighted blend",
    })


@router.get("/player/{player_id}")
def player(player_id: int, league: str | None = None):
    """One player's projection, with the seasons and weights behind it."""
    lg = leagues.get(league)
    try:
        return analytics.json_safe({
            **predictions.project_player(lg, player_id),
            "league": lg.key,
            "season_format": lg.season_format,
            # For spotting a retired player, whose own next season is in the past.
            "league_target_season": str(predictions.next_season(lg)),
            "accuracy": predictions.player_backtest(lg, "pts", seasons_back=3),
        })
    except KeyError as e:
        raise HTTPException(404, f"No {lg.label} player with id {e.args[0]}.") from e
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


def _schedule_frame(lg):
    df = data.schedule(lg)
    if df is None or df.empty:
        raise HTTPException(
            404,
            f"No {lg.label} schedule on disk. Run `python etl/schedule_etl.py "
            f"--league {lg.key}` to fetch upcoming games.",
        )
    return df


def _predict_row(lg, home: str, away: str) -> dict | None:
    """Elo prediction for a scheduled game, or None when a team has no rating
    yet — an expansion side's first game has nothing to predict from."""
    try:
        return predictions.predict_game(lg, home, away)
    except KeyError:
        return None


@router.get("/calendar")
def calendar(league: str | None = None):
    """Every date that has games, with counts, so the grid can mark them."""
    lg = leagues.get(league)
    df = _schedule_frame(lg)

    counts = (df.assign(upcoming=df.status.eq("pre"))
                .groupby("date", as_index=False)
                .agg(games=("game_id", "size"), upcoming=("upcoming", "sum")))
    counts = counts.sort_values("date")

    # Open on the next day that still has games to play; failing that, the most
    # recent day with any, so the calendar is never empty out of season.
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    future = counts[(counts.date >= today) & (counts.upcoming > 0)]
    default = (future.iloc[0]["date"] if len(future)
               else (counts.iloc[-1]["date"] if len(counts) else None))

    return analytics.json_safe({
        "league": lg.key,
        "dates": data.records(counts),
        "default_date": default,
        "today": today,
        "first": counts.iloc[0]["date"] if len(counts) else None,
        "last": counts.iloc[-1]["date"] if len(counts) else None,
    })


@router.get("/schedule")
def schedule(date: str, league: str | None = None):
    """Games on one date, each with the model's read on it."""
    lg = leagues.get(league)
    df = _schedule_frame(lg)
    day = df[df.date == str(date)].sort_values("tipoff")

    games = []
    for row in day.itertuples():
        prediction = _predict_row(lg, row.home, row.away)
        games.append({
            "game_id": row.game_id,
            "date": row.date,
            "tipoff": row.tipoff,
            "home": row.home,
            "away": row.away,
            "home_name": row.home_name,
            "away_name": row.away_name,
            "status": row.status,
            "completed": bool(row.completed),
            "prediction": prediction,
        })
    return analytics.json_safe({
        "league": lg.key,
        "date": str(date),
        "games": games,
        "model": "elo",
        "note": None if all(g["prediction"] for g in games) else
                "Some teams have no rating yet, so those games are unpredicted.",
    })


@router.get("/game/{game_id}")
def game(game_id: str, league: str | None = None, top: int = predictions.ROTATION_SIZE):
    """One scheduled game: the team model's read, plus a projected line for
    every rotation player, and what they actually did if it has been played."""
    lg = leagues.get(league)
    df = _schedule_frame(lg)
    row = df[df.game_id.astype(str) == str(game_id)]
    if row.empty:
        raise HTTPException(404, f"No {lg.label} game with id {game_id!r}.")
    g = row.iloc[0]

    lines = predictions.game_player_lines(lg, g.home, g.away, int(g.season),
                                          top=max(1, min(int(top), 15)))
    actual = (predictions.game_actual_lines(lg, g.date, g.home, g.away)
              if bool(g.completed) else {})

    # Match the projection to the box score by player, so a line can be read
    # against what happened rather than beside it.
    for side, rows in lines["players"].items():
        played = {p["player_id"]: p for p in actual.get(side, [])}
        for r in rows:
            r["actual"] = played.get(r["player_id"])

    return analytics.json_safe({
        "league": lg.key,
        "game_id": str(g.game_id),
        "date": g.date,
        "home": g.home, "away": g.away,
        "home_name": g.home_name, "away_name": g.away_name,
        "completed": bool(g.completed),
        "prediction": _predict_row(lg, g.home, g.away),
        **lines,
        "metrics": list(predictions.GAME_LINE_METRICS),
        "accuracy": predictions.game_line_backtest(lg, "pts"),
        "model": "marcel-style projection, adjusted for opponent and venue",
        "roster_source": predictions.roster_source(lg, int(g.season)),
        "roster_note": (
            "Rotations follow the published roster for this season, ordered by "
            "last season's minutes. Rookies and players with no prior games "
            "have nothing to project from and are left out."
            if predictions.roster_source(lg, int(g.season)) == "roster"
            else "No roster is published for this season yet, so rotations are "
                 "last season's — trades and signings since are not reflected."
        ),
    })
