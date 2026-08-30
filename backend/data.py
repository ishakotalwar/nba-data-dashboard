"""Parquet loading + in-memory derived state."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CANDIDATE_METRICS = [
    "ts_pct", "usg_pct", "ortg", "drtg",
    "pts", "ast", "reb", "stl", "blk", "tov",
    "fg_pct", "three_pct", "ft_pct", "per", "bpm",
]
INVERT_METRICS = {"drtg", "tov"}


def _load(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python etl/nba_etl.py` first.")
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def players() -> pd.DataFrame:
    df = _load("players.parquet")
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    return df


@lru_cache(maxsize=1)
def teams() -> pd.DataFrame:
    df = _load("teams.parquet")
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    return df


@lru_cache(maxsize=1)
def available_metrics() -> list[str]:
    cols = set(players().columns)
    return [m for m in CANDIDATE_METRICS if m in cols]


@lru_cache(maxsize=1)
def player_names() -> list[str]:
    return sorted(players().get("player_name", pd.Series(dtype=str)).dropna().unique().tolist())


@lru_cache(maxsize=1)
def team_names() -> list[str]:
    return sorted(teams().get("team_name", pd.Series(dtype=str)).dropna().unique().tolist())


@lru_cache(maxsize=1)
def seasons() -> list[str]:
    return sorted(players().get("season", pd.Series(dtype=str)).dropna().unique().tolist())


def allowed_seasons(lo: str | None, hi: str | None) -> list[str]:
    s = seasons()
    if not s or lo is None or hi is None:
        return s
    try:
        i, j = s.index(lo), s.index(hi)
        if i > j:
            i, j = j, i
        return s[i : j + 1]
    except ValueError:
        return s


def clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list[dict] with NaN -> None for JSON safety."""
    return (
        df.replace({np.nan: None, np.inf: None, -np.inf: None})
          .to_dict(orient="records")
    )
