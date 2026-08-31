"""Comparison across player-seasons.

The unit of comparison is (player_id, season) — "2016 Stephen Curry" — not a
player, so seasons from different eras and different players sit side by side.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import analytics, data, leagues

router = APIRouter(prefix="/api", tags=["compare"])


class Selection(BaseModel):
    player_id: int
    season: str | None = None


class CompareRequest(BaseModel):
    selections: list[Selection]
    metrics: list[str] = []
    league: str | None = None
    mode: str = "season"  # "season" | "career"


def _label(name: str, season: str) -> str:
    return f"{season} {name}"


@router.post("/compare")
def compare(req: CompareRequest):
    lg = leagues.get(req.league)
    if not req.selections:
        return {"mode": req.mode, "rows": [], "metrics": []}
    if req.mode not in ("season", "career"):
        raise HTTPException(400, "mode must be 'season' or 'career'")

    df = data.players(lg)
    mets = [m for m in req.metrics if m in data.available_metrics(lg)]
    if not mets:
        mets = [m for m in ("pts", "reb", "ast", "ts_pct") if m in data.available_metrics(lg)]

    if req.mode == "career":
        return _career(lg, df, req.selections, mets)
    return _single_season(lg, df, req.selections[:5], mets)


def _single_season(lg, df: pd.DataFrame, selections: list[Selection], mets: list[str]) -> dict:
    ids = pd.to_numeric(df["player_id"], errors="coerce")
    rows, radar = [], {}
    league_avg: dict[str, dict[str, float]] = {}

    for sel in selections:
        sub = df[ids == sel.player_id]
        if sub.empty:
            continue
        season = sel.season or str(sub["season"].max())
        row = sub[sub["season"] == str(season)]
        if row.empty:
            continue
        r = row.iloc[0]
        name = str(r["player_name"])
        key = _label(name, str(season))

        pool = analytics.gp_filtered_pool(analytics.season_pool(lg, season))
        if season not in league_avg:
            league_avg[season] = {
                m: float(pd.to_numeric(pool[m], errors="coerce").mean())
                for m in mets if m in pool.columns
            }

        entry = {"key": key, "player_id": sel.player_id, "player_name": name,
                 "season": str(season), "team": r.get("team_abbr"),
                 "gp": r.get("gp"), "values": {}, "percentiles": {}, "vs_league": {}}
        vals = []
        for m in mets:
            v = pd.to_numeric(pd.Series([r.get(m)]), errors="coerce").iloc[0]
            v = None if pd.isna(v) else float(v)
            entry["values"][m] = v
            pct = analytics.percentile_series(pool, m)
            match = pool["player_name"] == name
            pv = float(pct[match].iloc[0]) if match.any() else None
            entry["percentiles"][m] = None if pv is None or pd.isna(pv) else round(pv * 100, 1)
            avg = league_avg[season].get(m)
            entry["vs_league"][m] = (
                None if v is None or avg is None or pd.isna(avg) else round(v - avg, 3)
            )
            vals.append(0.5 if pv is None or pd.isna(pv) else round(pv, 4))
        rows.append(entry)
        radar[key] = vals

    return analytics.json_safe({
        "mode": "season", "metrics": mets, "rows": rows,
        "radar": {"features": mets, "values": radar},
        "league_avg": league_avg,
    })


def _career(lg, df: pd.DataFrame, selections: list[Selection], mets: list[str]) -> dict:
    """Career trajectories: each metric against the player's age that season."""
    ids = pd.to_numeric(df["player_id"], errors="coerce")
    births = data.birthdates(lg)
    curves = []
    for sel in selections[:5]:
        sub = df[ids == sel.player_id].sort_values("season")
        if sub.empty:
            continue
        name = str(sub.iloc[-1]["player_name"])
        bd = births.get(int(sel.player_id))
        points = []
        for _, r in sub.iterrows():
            season = str(r["season"])
            age = None
            if bd is not None and pd.notna(bd):
                try:
                    start = pd.Timestamp(year=int(season.split("-")[0]),
                                         month=lg.season_start_month, day=1)
                    age = round((start - bd).days / 365.25, 2)
                except Exception:
                    age = None
            point = {"season": season, "age": age, "team": r.get("team_abbr"), "gp": r.get("gp")}
            for m in mets:
                v = pd.to_numeric(pd.Series([r.get(m)]), errors="coerce").iloc[0]
                point[m] = None if pd.isna(v) else float(v)
            points.append(point)
        curves.append({
            "player_id": int(sel.player_id), "player_name": name,
            "has_age": bd is not None and pd.notna(bd),
            "points": points,
        })
    return analytics.json_safe({"mode": "career", "metrics": mets, "curves": curves})
