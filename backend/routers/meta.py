"""Health, league discovery, and per-league metadata."""
from __future__ import annotations

from fastapi import APIRouter

from .. import data, leagues
from . import shots

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
def health():
    return {"ok": True}


@router.get("/leagues")
def league_list():
    """Which leagues exist and which have Parquet on disk."""
    return {
        "leagues": [
            {"key": lg.key, "label": lg.label, "available": data.has_data(lg)}
            for lg in leagues.LEAGUES.values()
        ],
        "default": leagues.DEFAULT.key,
    }


@router.get("/meta")
def meta(league: str | None = None):
    lg = leagues.get(league)
    return {
        "league": lg.key,
        "league_label": lg.label,
        "season_format": lg.season_format,
        "players": data.player_names(lg),
        "player_ids": data.player_ids(lg),
        "teams": data.team_names(lg),
        "seasons": data.seasons(lg),
        # Lineups start later than the rest: they need play-by-play, and the
        # early substitution logs are too sparse to rebuild a five from.
        "lineup_seasons": data.lineup_seasons(lg),
        "metrics": data.available_metrics(lg),
        "invert_metrics": sorted(data.INVERT_METRICS),
        # Everything the shot chart needs to redraw the same zones the
        # backend classifies shots into.
        "court": {"arc": lg.three_point_arc, "corner": lg.three_point_corner,
                  "rim": shots.RESTRICTED_RADIUS, "paint_width": shots.PAINT_HALF_WIDTH,
                  "paint_depth": shots.PAINT_DEPTH, "paint_near": shots.PAINT_NEAR_RADIUS,
                  "wing_angle": shots.WING_ANGLE},
    }
