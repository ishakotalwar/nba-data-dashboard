"""Shared analytics helpers.

Pure functions over DataFrames, with no FastAPI or league-specific branching —
league differences live in `leagues.py`, and the routers do the HTTP work.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import data

# Minimum games played to include a row in the pool used to compute percentile
# ranks. Keeps low-sample bench players from polluting the axes.
MIN_GP_FOR_POOL = 15

# Features used for similarity, in a fixed order.
SIMILARITY_FEATURES = ["pts", "ast", "reb", "tov", "ts_pct", "usg_pct", "ortg", "drtg"]


def json_safe(obj: Any) -> Any:
    """NaN/inf -> None, numpy scalars -> python, Timestamps -> ISO strings."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def gp_filtered_pool(df: pd.DataFrame, min_gp: int = MIN_GP_FOR_POOL) -> pd.DataFrame:
    """Drop low-sample rows, falling back to the full pool if that empties it."""
    if "gp" not in df.columns:
        return df
    gp = pd.to_numeric(df["gp"], errors="coerce").fillna(0)
    filt = df[gp >= min_gp]
    return filt if len(filt) >= 30 else df


def percentile_series(pool: pd.DataFrame, metric: str) -> pd.Series:
    """Percentile rank in [0,1] for one metric; larger is always better
    (drtg/tov are inverted so a lower raw value ranks higher)."""
    col = pd.to_numeric(pool.get(metric), errors="coerce")
    return col.rank(pct=True, ascending=(metric not in data.INVERT_METRICS))


def percentile_map(pool: pd.DataFrame, feats: list[str]) -> dict[str, dict[str, float]]:
    """Per-metric {player_name: percentile rank in [0,1]}."""
    out: dict[str, dict[str, float]] = {}
    if pool.empty or "player_name" not in pool.columns:
        return out
    for f in feats:
        out[f] = dict(zip(pool["player_name"], percentile_series(pool, f)))
    return out


def four_factors(row: pd.Series) -> dict:
    """Dean Oliver's Four Factors. ORB% uses OREB/(OREB+DREB) as a proxy —
    the real formula needs opponent DREB."""
    fga = float(row.get("FGA") or 0)
    fta = float(row.get("FTA") or 0)
    fgm = float(row.get("FGM") or 0)
    fg3m = float(row.get("FG3M") or 0)
    tov = float(row.get("TOV") or 0)
    oreb = float(row.get("OREB") or 0)
    dreb = float(row.get("DREB") or 0)
    poss = fga + 0.44 * fta + tov
    return {
        "eFG%": ((fgm + 0.5 * fg3m) / fga) if fga else None,
        "TOV%": (tov / poss) if poss else None,
        "ORB%": (oreb / (oreb + dreb)) if (oreb + dreb) else None,
        "FT rate": (fta / fga) if fga else None,
    }


def season_pool(league, season: str, per: str = "game") -> pd.DataFrame:
    """Every player row for one season, on the requested rate basis.

    The basis has to reach this far in: percentiles compare a player against
    everyone else, so ranking per-36 numbers inside a per-game pool would score
    a bench player against starters' raw totals.
    """
    df = data.players_at(per, league)
    return df[df["season"] == str(season)].copy()


def rank_in_season(pool: pd.DataFrame, metric: str, value: float) -> int | None:
    """1-based league rank for a value within a season pool, best = 1."""
    col = pd.to_numeric(pool.get(metric), errors="coerce").dropna()
    if col.empty or value is None or pd.isna(value):
        return None
    if metric in data.INVERT_METRICS:
        return int((col < value).sum()) + 1
    return int((col > value).sum()) + 1
