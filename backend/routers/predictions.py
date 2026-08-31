"""Forecasting endpoints — team game outcomes and player season projections.

Everything here is arithmetic over local Parquet (see `backend/predictions.py`).
Each model reports its own backtest alongside its output, so the UI can show
how well it actually does rather than asking to be believed.
"""
from __future__ import annotations

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
