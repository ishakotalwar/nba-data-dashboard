"""Query historical player-seasons.

Answers questions of the shape "which player-seasons since 2003 averaged 25+
points while shooting 40%+ from three", with arbitrary filter stacks, sorting
and pagination over the whole dataset.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import analytics, data, leagues

router = APIRouter(prefix="/api", tags=["explorer"])

OPS = {">", ">=", "<", "<=", "=", "between"}
MAX_PAGE_SIZE = 200


class Filter(BaseModel):
    metric: str
    op: str
    value: float
    value2: float | None = None  # upper bound for "between"


class ExplorerRequest(BaseModel):
    league: str | None = None
    season_from: str | None = None
    season_to: str | None = None
    min_gp: int = 0
    min_min: float = 0
    team: str | None = None
    player: str | None = None
    filters: list[Filter] = []
    sort: str = "pts"
    dir: str = "desc"
    page: int = 1
    page_size: int = 25


@router.get("/explorer/fields")
def fields(league: str | None = None):
    """What can be filtered and sorted on, for this league's data."""
    lg = leagues.get(league)
    df = data.players(lg)
    numeric = [c for c in df.columns
               if c not in ("player_id", "team_id") and pd.api.types.is_numeric_dtype(df[c])]
    return {
        "league": lg.key,
        "metrics": data.available_metrics(lg),
        "numeric_fields": numeric,
        "seasons": data.seasons(lg),
        "teams": sorted(df["team_abbr"].dropna().unique().tolist()) if "team_abbr" in df else [],
        "operators": sorted(OPS),
    }


@router.post("/explorer")
def explorer(req: ExplorerRequest):
    lg = leagues.get(req.league)
    df = data.players(lg).copy()

    if req.season_from:
        df = df[df["season"] >= str(req.season_from)]
    if req.season_to:
        df = df[df["season"] <= str(req.season_to)]
    if req.min_gp and "gp" in df.columns:
        df = df[pd.to_numeric(df["gp"], errors="coerce").fillna(0) >= req.min_gp]
    if req.min_min and "min" in df.columns:
        df = df[pd.to_numeric(df["min"], errors="coerce").fillna(0) >= req.min_min]
    if req.team and "team_abbr" in df.columns:
        df = df[df["team_abbr"] == req.team]
    if req.player:
        df = df[df["player_name"].str.contains(req.player, case=False, na=False)]

    for f in req.filters:
        if f.op not in OPS:
            raise HTTPException(400, f"Unsupported operator {f.op!r}")
        if f.metric not in df.columns:
            raise HTTPException(400, f"Unknown field {f.metric!r}")
        col = pd.to_numeric(df[f.metric], errors="coerce")
        if f.op == ">":
            df = df[col > f.value]
        elif f.op == ">=":
            df = df[col >= f.value]
        elif f.op == "<":
            df = df[col < f.value]
        elif f.op == "<=":
            df = df[col <= f.value]
        elif f.op == "=":
            df = df[col == f.value]
        elif f.op == "between":
            hi = f.value2 if f.value2 is not None else f.value
            lo, hi = min(f.value, hi), max(f.value, hi)
            df = df[col.between(lo, hi)]

    total = int(len(df))
    sort_key = req.sort if req.sort in df.columns else "pts"
    if sort_key in df.columns:
        df = df.sort_values(
            sort_key, ascending=(req.dir == "asc"), na_position="last", kind="mergesort"
        )

    page_size = max(1, min(req.page_size, MAX_PAGE_SIZE))
    page = max(1, req.page)
    start = (page - 1) * page_size
    rows = df.iloc[start : start + page_size]

    cols = ["player_id", "player_name", "season", "team_abbr", "gp", "min"] + [
        m for m in data.available_metrics(lg) if m in df.columns
    ]
    seen, ordered = set(), []
    for c in cols:
        if c in rows.columns and c not in seen:
            ordered.append(c)
            seen.add(c)

    return analytics.json_safe({
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, -(-total // page_size)),
        "columns": [c for c in ordered if c != "player_id"],
        "rows": data.records(rows[ordered]),
    })
