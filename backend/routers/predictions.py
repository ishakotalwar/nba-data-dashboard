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
        return analytics.json_safe({**predictions.project_player(lg, player_id),
                                    "league": lg.key,
                                    "season_format": lg.season_format})
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
