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

# Counting stats, which mean different things at different rates. Everything
# else in the table is already a rate (`fg_pct`) or a total (`gp`), and is left
# alone whichever basis is asked for.
COUNTING_STATS = ["pts", "ast", "reb", "stl", "blk", "tov"]

# How to express a counting stat. Stored per game; the rest divide out playing
# time so a bench player and a starter can be read on one scale. Per 75 is the
# possession count a starter actually uses in a game, so its numbers stay near
# the per-game ones they replace, where per 100 inflates everything.
RATE_BASES = {
    "game": "Per game",
    "per36": "Per 36 minutes",
    "per75": "Per 75 possessions",
    "per100": "Per 100 possessions",
}

# The possession bases and what each scales to.
POSSESSION_BASES = {"per75": 75.0, "per100": 100.0}

# Which games an impact rating was fit on. A postseason is its own sample, not
# a continuation of the season, so the two are never pooled.
RATING_SEASON_TYPES = {"regular": "Regular season", "playoffs": "Playoffs"}

# The impact columns a page can rank on, in the order they nest: the parts,
# then each side's total, then the whole.
RATING_PARTS = ["field_goals", "free_throws", "second_chance", "turnovers"]
RATING_COLUMNS = ([f"{side}_{part}" for side in ("off", "def") for part in RATING_PARTS]
                  + ["off_rating", "def_rating", "rapm", "on_off"])


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
def player_ids(league: League = DEFAULT) -> dict[str, int]:
    """player_name -> id. These are ESPN athlete ids for sdv-sourced data, which
    is what the headshot CDN is keyed by. Latest season wins on a name clash."""
    df = players(league)
    if "player_id" not in df.columns:
        return {}
    latest = df.sort_values("season").drop_duplicates("player_name", keep="last")
    return {str(n): int(i) for n, i in zip(latest["player_name"], latest["player_id"])
            if pd.notna(i)}


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
def lineups(league: League = DEFAULT) -> pd.DataFrame | None:
    """Five-man lineup totals, or None if `etl/lineup_etl.py` hasn't run.

    Rebuilt from play-by-play substitutions, so it only covers the seasons that
    ETL was pointed at — a shorter range than the box-score data, since ESPN's
    older substitution logs are too sparse to close lineups reliably.
    """
    df = _load_optional("lineup", league)
    return _with_season_str(df) if df is not None else None


@lru_cache(maxsize=8)
def lineup_seasons(league: League = DEFAULT) -> list[str]:
    """Seasons with lineup data, newest last. Empty when the ETL hasn't run."""
    df = lineups(league)
    if df is None or df.empty:
        return []
    return sorted(df["season"].unique().tolist())


@lru_cache(maxsize=8)
def ratings(league: League = DEFAULT) -> pd.DataFrame | None:
    """Player impact ratings, or None if `etl/lineup_etl.py` hasn't run.

    Built from the same stints as `lineups`, so it covers the same seasons.
    """
    df = _load_optional("rating", league)
    return _with_season_str(df) if df is not None else None


@lru_cache(maxsize=16)
def rating_seasons(league: League = DEFAULT, season_type: str = "regular") -> list[str]:
    """Seasons with impact ratings of this kind, newest last.

    The playoff list is shorter: a postseason too small to regress anything out
    is skipped by the ETL rather than fit to noise.
    """
    df = ratings(league)
    if df is None or df.empty:
        return []
    if "season_type" in df.columns:
        df = df[df["season_type"] == season_type]
    elif season_type != "regular":
        return []
    return sorted(df["season"].unique().tolist())


@lru_cache(maxsize=8)
def wowy(league: League = DEFAULT) -> pd.DataFrame | None:
    """What a team did with each player and pair on the floor, or None if
    `etl/lineup_etl.py` hasn't run. Built from every stint, so it covers a
    team's whole season rather than the lineups long enough to store."""
    df = _load_optional("wowy", league)
    return _with_season_str(df) if df is not None else None


@lru_cache(maxsize=8)
def schedule(league: League = DEFAULT) -> pd.DataFrame | None:
    """Scheduled games for this league, or None if the schedule ETL hasn't run.

    Built by `etl/schedule_etl.py`, separately from the historical data: this
    is the only file that says anything about games not yet played.
    """
    return _load_optional("schedule", league)


@lru_cache(maxsize=8)
def roster(league: League = DEFAULT) -> pd.DataFrame | None:
    """Who is actually on each team, by season, or None if not fetched yet.

    Built by `etl/schedule_etl.py`. Without it, "this team's players" can only
    mean whoever finished the previous season there, which is wrong for about a
    quarter of the league once trades and free agency have happened.
    """
    return _load_optional("roster", league)


@lru_cache(maxsize=8)
def injuries(league: League = DEFAULT) -> pd.DataFrame | None:
    """The current injury report, or None if it hasn't been fetched.

    A snapshot with no history — it says who is hurt *now*, so it can flag a
    projection but cannot be backtested against past games.
    """
    return _load_optional("injury", league)


@lru_cache(maxsize=8)
def birthdates(league: League = DEFAULT) -> dict[int, pd.Timestamp]:
    """player_id -> birthdate, empty when this league has no bio file."""
    df = _load_optional("player_bio", league)
    if df is None or df.empty:
        return {}
    return dict(zip(df["player_id"].astype(int), pd.to_datetime(df["birthdate"])))


@lru_cache(maxsize=24)
def players_at(basis: str = "game", league: League = DEFAULT) -> pd.DataFrame:
    """`players`, with the counting stats restated on `basis`.

    Per-36 divides by minutes played. The possession bases divide by the
    possessions the player's team used while he was on the floor, which nobody
    publishes: it is his minutes times his team's pace, so they inherit two
    approximations. The team's pace is Oliver's estimate rather than a count
    (ESPN gives no possession data), and a player is assumed to have played at
    his team's average pace rather than his own, which box scores cannot
    separate.
    Someone traded mid-season is priced at the pace of the team he played the
    most games for, since that is the only team the row records.

    Percentiles, filters and comparisons all read this, so a rate basis changes
    the pool a player is ranked against and not just the number displayed.
    """
    if basis not in RATE_BASES:
        raise ValueError(f"Unknown rate basis {basis!r}. Expected one of: "
                         f"{', '.join(RATE_BASES)}")
    df = players(league)
    if basis == "game":
        return df

    out = clean_numeric(df.copy(), COUNTING_STATS + ["min", "gp"])
    minutes = out["min"].where(out["min"] > 0)
    if basis == "per36":
        factor = 36.0 / minutes
    else:
        scale = POSSESSION_BASES[basis]
        # pace is possessions per game, so a game's worth of minutes for one
        # player is the team's floor time divided by the five on it.
        pace = _team_pace(league).reindex(
            pd.MultiIndex.from_arrays([out["season"], out["team_id"]])).to_numpy()
        per_player_minutes = league.team_minutes / 5.0
        possessions = pd.Series(pace, index=out.index) * (minutes / per_player_minutes)
        factor = scale / possessions.where(possessions > 0)

    for stat in COUNTING_STATS:
        if stat in out.columns:
            out[stat] = (out[stat] * factor).round(2)
    return out


@lru_cache(maxsize=8)
def _team_pace(league: League = DEFAULT) -> pd.Series:
    """(season, team_id) -> possessions per game, for per-100 conversion."""
    tdf = teams(league)
    pace = pd.to_numeric(tdf["pace"], errors="coerce")
    return pd.Series(pace.to_numpy(),
                     index=pd.MultiIndex.from_arrays([tdf["season"], tdf["team_id"]]))


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
