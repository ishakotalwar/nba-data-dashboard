"""Shot locations, zone aggregation, and player-season shot comparison.

Coordinates are in the dashboard's court frame: tenths of a foot, hoop at the
origin, y increasing toward half court. Zone boundaries come from the league's
own three-point geometry in `leagues.py`, so the WNBA's shorter line is handled
without a second code path.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .. import analytics, data, leagues, live

router = APIRouter(prefix="/api", tags=["shots"])

# Which games a shot chart draws from. Everything else in the app is regular
# season only, so that stays the default; the playoffs are a different sample
# and worth seeing on their own, not silently folded in.
SEASON_TYPES = {
    "regular": "Regular season",
    "playoffs": "Playoffs",
    "both": "Both",
}

RESTRICTED_RADIUS = 40.0    # 4 ft from the rim
PAINT_HALF_WIDTH = 80.0     # the lane is 16 ft wide
PAINT_DEPTH = 137.5         # to the free-throw line
PAINT_NEAR_RADIUS = 80.0    # 8 ft: past this the lane is floater range

# Above the break splits into wings and the top of the arc at these angles from
# the rim, measured from the right baseline: a 60-degree band across the top,
# with the wings filling the gap down to where the corner begins.
WING_ANGLE = 60.0

ZONE_ORDER = [
    "Rim", "Paint", "Short Midrange", "Long Midrange",
    "Left Corner 3", "Left Wing 3", "Top of Arc 3", "Right Wing 3", "Right Corner 3",
]


def classify(x: pd.Series, y: pd.Series, league) -> pd.Series:
    """Assign each shot to one of ZONE_ORDER."""
    dist = np.hypot(x, y)
    arc, corner = league.three_point_arc, league.three_point_corner
    # Corner threes sit below where the arc meets the straight corner line.
    junction_y = float(np.sqrt(max(arc * arc - corner * corner, 0.0)))

    is_corner3 = (x.abs() >= corner) & (y <= junction_y)
    is_three = is_corner3 | (dist >= arc)

    zone = pd.Series("Long Midrange", index=x.index, dtype=object)
    zone[is_three & is_corner3 & (x < 0)] = "Left Corner 3"
    zone[is_three & is_corner3 & (x >= 0)] = "Right Corner 3"

    # Angle from the rim: 0 along the right baseline, 180 along the left.
    angle = np.degrees(np.arctan2(y, x))
    above = is_three & ~is_corner3
    zone[above & (angle < 90 - WING_ANGLE / 2)] = "Right Wing 3"
    zone[above & (angle > 90 + WING_ANGLE / 2)] = "Left Wing 3"
    zone[above & (angle >= 90 - WING_ANGLE / 2) & (angle <= 90 + WING_ANGLE / 2)] = "Top of Arc 3"

    # The lane splits at 8 ft: everything closer is a shot at the basket,
    # everything beyond it out to the free-throw line is a longer look.
    in_paint = ~is_three & (x.abs() <= PAINT_HALF_WIDTH) & (y <= PAINT_DEPTH)
    zone[in_paint & (dist > PAINT_NEAR_RADIUS)] = "Short Midrange"
    zone[in_paint & (dist <= PAINT_NEAR_RADIUS)] = "Paint"
    zone[~is_three & (dist <= RESTRICTED_RADIUS)] = "Rim"
    return zone


@lru_cache(maxsize=16)
def _zoned(league_key: str) -> pd.DataFrame:
    """Every shot with a zone column. Cached — this is the expensive step."""
    lg = leagues.LEAGUES[league_key]
    df = data.shots(lg)
    if df is None:
        raise HTTPException(404, f"No {lg.label} shot data on disk")
    out = df.copy()
    out["zone"] = classify(out["x"], out["y"], lg)
    return out


def _check_season_type(season_type: str) -> None:
    if season_type not in SEASON_TYPES:
        raise HTTPException(400, f"Unknown season type {season_type!r}. Expected one "
                                 f"of: {', '.join(SEASON_TYPES)}")


def _of_type(df: pd.DataFrame, season_type: str) -> pd.DataFrame:
    """Narrow to regular season or playoffs. Files built before shots carried a
    season type have no column to filter on, and are left whole."""
    if season_type == "both" or "season_type" not in df.columns:
        return df
    return df[df["season_type"] == season_type]


@lru_cache(maxsize=64)
def _league_zone_rates(league_key: str, season: str,
                       season_type: str = "regular") -> dict[str, float]:
    """League-wide FG% per zone for one season, on the same games as the player.

    The comparison only means something if both sides are drawn from the same
    kind of game: playoff defenses are better, so a player's playoff shooting
    against a regular-season league average would read as a decline he didn't
    have.
    """
    df = _of_type(_zoned(league_key), season_type)
    sub = df[df["season"] == str(season)]
    if sub.empty:
        return {}
    g = sub.groupby("zone")["made"].mean()
    return {str(k): float(v) for k, v in g.items()}


def _player_shots(league_key: str, player_id: int, season: str,
                  season_type: str = "regular") -> pd.DataFrame:
    df = _of_type(_zoned(league_key), season_type)
    ids = pd.to_numeric(df["player_id"], errors="coerce")
    return df[(ids == player_id) & (df["season"] == str(season))]


def _zone_table(league_key: str, player_id: int, season: str,
                season_type: str = "regular") -> list[dict]:
    sub = _player_shots(league_key, player_id, season, season_type)
    lg_rates = _league_zone_rates(league_key, season, season_type)
    total = len(sub)
    rows = []
    for zone in ZONE_ORDER:
        z = sub[sub["zone"] == zone]
        fga = int(len(z))
        fgm = int(z["made"].sum()) if fga else 0
        pct = (fgm / fga) if fga else None
        lg = lg_rates.get(zone)
        rows.append({
            "zone": zone,
            "fga": fga,
            "fgm": fgm,
            "fg_pct": pct,
            "share": (fga / total) if total else None,
            "league_fg_pct": lg,
            "diff": (pct - lg) if (pct is not None and lg is not None) else None,
        })
    return rows


@router.get("/shots/zones")
def shot_zones(player_id: int, season: str, league: str | None = None,
               season_type: str = "regular"):
    lg = leagues.get(league)
    _check_season_type(season_type)
    sub = _player_shots(lg.key, player_id, season, season_type)
    if sub.empty:
        raise HTTPException(404, f"No {lg.label} {SEASON_TYPES[season_type].lower()} "
                                 f"shots for player {player_id} in {season}")
    return analytics.json_safe({
        "player_id": player_id,
        "player_name": str(sub.iloc[0]["player_name"]),
        "season": str(season),
        "season_type": season_type,
        "total_fga": int(len(sub)),
        "fg_pct": float(sub["made"].mean()),
        "zones": _zone_table(lg.key, player_id, season, season_type),
    })


class ShotCompareRequest(BaseModel):
    a: dict  # {player_id, season}
    b: dict
    league: str | None = None
    season_type: str = "regular"


@router.post("/shots/compare")
def shot_compare(req: ShotCompareRequest):
    """Two player-seasons side by side, zone for zone."""
    lg = leagues.get(req.league)
    _check_season_type(req.season_type)
    out = {}
    for side, sel in (("a", req.a), ("b", req.b)):
        pid, season = int(sel["player_id"]), str(sel["season"])
        sub = _player_shots(lg.key, pid, season, req.season_type)
        if sub.empty:
            raise HTTPException(404, f"No {lg.label} "
                                     f"{SEASON_TYPES[req.season_type].lower()} shots "
                                     f"for player {pid} in {season}")
        out[side] = {
            "player_id": pid,
            "player_name": str(sub.iloc[0]["player_name"]),
            "season": season,
            "total_fga": int(len(sub)),
            "fg_pct": float(sub["made"].mean()),
            "zones": _zone_table(lg.key, pid, season, req.season_type),
        }
    return analytics.json_safe(out)


@router.get("/shots")
def shots(
    player_id: int,
    season: str,
    mode: str = Query("hex", pattern="^(scatter|hex)$"),
    league: str | None = None,
    season_type: str = "regular",
):
    """Raw shot locations, as points or hex-binned zones."""
    lg = leagues.get(league)
    _check_season_type(season_type)
    local = data.shots(lg)
    if local is not None:
        sub = _of_type(local, season_type)
        ids = pd.to_numeric(sub["player_id"], errors="coerce")
        sdf = sub[(ids == player_id) & (sub["season"] == str(season))]
    else:
        try:
            sdf = live.fetch_shots(player_id, season, lg)
        except Exception as e:
            raise HTTPException(502, live.friendly_upstream_message(e, lg)) from e

    if sdf.empty:
        return {"player_id": player_id, "season": str(season), "mode": mode,
                "season_type": season_type, "shots": [], "hexes": []}

    name = str(sdf.iloc[0].get("player_name", ""))
    base = {"player_id": player_id, "player": name, "season": str(season),
            "season_type": season_type,
            "count": int(len(sdf)), "fg_pct": float(sdf["made"].mean())}

    if mode == "scatter":
        return analytics.json_safe({
            **base, "mode": "scatter",
            "shots": sdf[["x", "y", "made"]].head(3000).to_dict(orient="records"),
        })

    x, y, m = sdf["x"].to_numpy(), sdf["y"].to_numpy(), sdf["made"].to_numpy()
    xb, yb = np.linspace(-250, 250, 26), np.linspace(-52.5, 417.5, 24)
    cnt, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    made, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=m.astype(float))
    # Empty bins stay NaN, matching the mean-of-nothing this replaced.
    pct = np.divide(made, cnt, out=np.full_like(made, np.nan), where=cnt > 0)
    cx, cy = 0.5 * (xb[:-1] + xb[1:]), 0.5 * (yb[:-1] + yb[1:])
    hexes = [
        {"x": float(cx[i]), "y": float(cy[j]), "count": int(cnt[i, j]), "pct": float(pct[i, j])}
        for i in range(cnt.shape[0]) for j in range(cnt.shape[1]) if cnt[i, j] >= 2
    ]
    return analytics.json_safe({**base, "mode": "hex", "hexes": hexes})
