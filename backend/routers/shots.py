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

RESTRICTED_RADIUS = 40.0    # 4 ft from the rim
PAINT_HALF_WIDTH = 80.0     # the lane is 16 ft wide
PAINT_DEPTH = 137.5         # to the free-throw line

ZONE_ORDER = [
    "Restricted Area", "Paint", "Midrange",
    "Left Corner 3", "Right Corner 3", "Above the Break 3",
]


def classify(x: pd.Series, y: pd.Series, league) -> pd.Series:
    """Assign each shot to one of ZONE_ORDER."""
    dist = np.hypot(x, y)
    arc, corner = league.three_point_arc, league.three_point_corner
    # Corner threes sit below where the arc meets the straight corner line.
    junction_y = float(np.sqrt(max(arc * arc - corner * corner, 0.0)))

    is_corner3 = (x.abs() >= corner) & (y <= junction_y)
    is_three = is_corner3 | (dist >= arc)

    zone = pd.Series("Midrange", index=x.index, dtype=object)
    zone[is_three & is_corner3 & (x < 0)] = "Left Corner 3"
    zone[is_three & is_corner3 & (x >= 0)] = "Right Corner 3"
    zone[is_three & ~is_corner3] = "Above the Break 3"
    in_paint = ~is_three & (x.abs() <= PAINT_HALF_WIDTH) & (y <= PAINT_DEPTH)
    zone[in_paint] = "Paint"
    zone[~is_three & (dist <= RESTRICTED_RADIUS)] = "Restricted Area"
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


@lru_cache(maxsize=64)
def _league_zone_rates(league_key: str, season: str) -> dict[str, float]:
    """League-wide FG% per zone for one season."""
    df = _zoned(league_key)
    sub = df[df["season"] == str(season)]
    if sub.empty:
        return {}
    g = sub.groupby("zone")["made"].mean()
    return {str(k): float(v) for k, v in g.items()}


def _player_shots(league_key: str, player_id: int, season: str) -> pd.DataFrame:
    df = _zoned(league_key)
    ids = pd.to_numeric(df["player_id"], errors="coerce")
    return df[(ids == player_id) & (df["season"] == str(season))]


def _zone_table(league_key: str, player_id: int, season: str) -> list[dict]:
    sub = _player_shots(league_key, player_id, season)
    lg_rates = _league_zone_rates(league_key, season)
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
def shot_zones(player_id: int, season: str, league: str | None = None):
    lg = leagues.get(league)
    sub = _player_shots(lg.key, player_id, season)
    if sub.empty:
        raise HTTPException(404, f"No {lg.label} shots for player {player_id} in {season}")
    return analytics.json_safe({
        "player_id": player_id,
        "player_name": str(sub.iloc[0]["player_name"]),
        "season": str(season),
        "total_fga": int(len(sub)),
        "fg_pct": float(sub["made"].mean()),
        "zones": _zone_table(lg.key, player_id, season),
    })


class ShotCompareRequest(BaseModel):
    a: dict  # {player_id, season}
    b: dict
    league: str | None = None


@router.post("/shots/compare")
def shot_compare(req: ShotCompareRequest):
    """Two player-seasons side by side, zone for zone."""
    lg = leagues.get(req.league)
    out = {}
    for side, sel in (("a", req.a), ("b", req.b)):
        pid, season = int(sel["player_id"]), str(sel["season"])
        sub = _player_shots(lg.key, pid, season)
        if sub.empty:
            raise HTTPException(404, f"No {lg.label} shots for player {pid} in {season}")
        out[side] = {
            "player_id": pid,
            "player_name": str(sub.iloc[0]["player_name"]),
            "season": season,
            "total_fga": int(len(sub)),
            "fg_pct": float(sub["made"].mean()),
            "zones": _zone_table(lg.key, pid, season),
        }
    return analytics.json_safe(out)


@router.get("/shots")
def shots(
    player_id: int,
    season: str,
    mode: str = Query("hex", pattern="^(scatter|hex)$"),
    league: str | None = None,
):
    """Raw shot locations, as points or hex-binned zones."""
    lg = leagues.get(league)
    local = data.shots(lg)
    if local is not None:
        ids = pd.to_numeric(local["player_id"], errors="coerce")
        sdf = local[(ids == player_id) & (local["season"] == str(season))]
    else:
        try:
            sdf = live.fetch_shots(player_id, season, lg)
        except Exception as e:
            raise HTTPException(502, live.friendly_upstream_message(e, lg)) from e

    if sdf.empty:
        return {"player_id": player_id, "season": str(season), "mode": mode, "shots": [], "hexes": []}

    name = str(sdf.iloc[0].get("player_name", ""))
    base = {"player_id": player_id, "player": name, "season": str(season),
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
