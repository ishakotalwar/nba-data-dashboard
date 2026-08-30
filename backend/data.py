"""Parquet loading + in-memory derived state, per league."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .leagues import DEFAULT, League

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CANDIDATE_METRICS = [
    "ts_pct", "usg_pct", "ortg", "drtg",
    "pts", "ast", "reb", "stl", "blk", "tov",
    "fg_pct", "three_pct", "ft_pct", "per", "bpm",
]
INVERT_METRICS = {"drtg", "tov"}


def _load(stem: str, league: League) -> pd.DataFrame:
    """Load `<stem><suffix>.parquet`, falling back to the pre-league
    `<stem>.parquet` name so existing NBA data keeps working unmigrated."""
    candidates = [DATA_DIR / f"{stem}{league.suffix}.parquet"]
    if league is DEFAULT:
        candidates.append(DATA_DIR / f"{stem}.parquet")
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)
    tried = " or ".join(p.name for p in candidates)
    raise FileNotFoundError(
        f"Missing {tried} in {DATA_DIR}. "
        f"Run `python etl/nba_etl.py --league {league.key}` first."
    )


def _with_season_str(df: pd.DataFrame) -> pd.DataFrame:
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    return df


@lru_cache(maxsize=8)
def players(league: League = DEFAULT) -> pd.DataFrame:
    return _with_season_str(_load("players", league))


@lru_cache(maxsize=8)
def teams(league: League = DEFAULT) -> pd.DataFrame:
    return _with_season_str(_load("teams", league))


@lru_cache(maxsize=8)
def available_metrics(league: League = DEFAULT) -> list[str]:
    cols = set(players(league).columns)
    return [m for m in CANDIDATE_METRICS if m in cols]


@lru_cache(maxsize=8)
def player_names(league: League = DEFAULT) -> list[str]:
    return sorted(players(league).get("player_name", pd.Series(dtype=str)).dropna().unique().tolist())


@lru_cache(maxsize=8)
def team_names(league: League = DEFAULT) -> list[str]:
    return sorted(teams(league).get("team_name", pd.Series(dtype=str)).dropna().unique().tolist())


@lru_cache(maxsize=8)
def seasons(league: League = DEFAULT) -> list[str]:
    return sorted(players(league).get("season", pd.Series(dtype=str)).dropna().unique().tolist())


def _load_optional(stem: str, league: League) -> pd.DataFrame | None:
    """Like _load, but None when the file isn't there. For datasets a league may
    simply not have (shots, game logs, birthdates) rather than an error state."""
    try:
        return _load(stem, league)
    except FileNotFoundError:
        return None


@lru_cache(maxsize=8)
def shots(league: League = DEFAULT) -> pd.DataFrame | None:
    """Local shot coordinates, or None if this league has none on disk."""
    return _load_optional("shots", league)


@lru_cache(maxsize=8)
def gamelog(league: League = DEFAULT) -> pd.DataFrame | None:
    """Local per-game rows, or None if this league has none on disk."""
    df = _load_optional("gamelog", league)
    return _with_season_str(df) if df is not None else None


@lru_cache(maxsize=8)
def birthdates(league: League = DEFAULT) -> dict[int, pd.Timestamp]:
    """player_id -> birthdate, empty when this league has no bio file."""
    df = _load_optional("player_bio", league)
    if df is None or df.empty:
        return {}
    return dict(zip(df["player_id"].astype(int), pd.to_datetime(df["birthdate"])))


def has_data(league: League) -> bool:
    """True if this league's player Parquet is present and loadable."""
    try:
        players(league)
        return True
    except Exception:
        return False


def allowed_seasons(lo: str | None, hi: str | None, league: League = DEFAULT) -> list[str]:
    s = seasons(league)
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
