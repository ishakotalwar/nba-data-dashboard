"""nba_api live fetchers with retry + in-process caching, per league."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .leagues import DEFAULT, NBA, League

try:
    from nba_api.stats.endpoints import commonplayerinfo, playergamelog, shotchartdetail
    from nba_api.stats.static import players as players_static
except Exception:  # pragma: no cover
    shotchartdetail = None
    playergamelog = None
    commonplayerinfo = None
    players_static = None

NBA_TIMEOUT = 45  # per-attempt; tenacity retries below


class ShotFetchError(RuntimeError):
    pass


class NBAUpstreamError(RuntimeError):
    """Upstream NBA API is unavailable or timing out."""


@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1, max=6),
)
def _call(endpoint_cls, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("timeout", NBA_TIMEOUT)
    try:
        ep = endpoint_cls(**kwargs)
    except TypeError:
        kwargs.pop("timeout", None)
        ep = endpoint_cls(**kwargs)
    dfs = ep.get_data_frames()
    if not dfs:
        raise ShotFetchError("NBA API returned no data")
    return dfs[0]


def friendly_upstream_message(exc: BaseException, league: League = DEFAULT) -> str:
    s = str(exc) or exc.__class__.__name__
    low = s.lower()
    if "timed out" in low or "timeout" in low:
        return (
            "stats.nba.com didn't respond in time. It rate-limits and sometimes blocks "
            "non-US IPs — try again in a moment, or use a US VPN."
        )
    if "connection" in low and ("refused" in low or "reset" in low):
        return "stats.nba.com refused the connection. Try again shortly."
    return f"stats.nba.com error ({league.label}): {s}"


# --- static roster lookups -------------------------------------------------

def _static_pool(league: League) -> list[dict]:
    """All known players for a league, from nba_api's bundled static data."""
    if players_static is None:
        return []
    if league is NBA:
        return players_static.get_players()
    getter = getattr(players_static, f"get_{league.key}_players", None)
    return getter() if getter else []


def find_player_id(name: str, league: League = DEFAULT) -> int | None:
    if players_static is None or not name:
        return None
    if league is NBA:
        hits = players_static.find_players_by_full_name(name) or []
    else:
        finder = getattr(players_static, f"find_{league.key}_players_by_full_name", None)
        hits = (finder(name) or []) if finder else []
    return hits[0]["id"] if hits else None


def search_players(q: str, limit: int = 20, league: League = DEFAULT) -> list[dict]:
    if not q:
        return []
    q = q.lower()
    matches = [p for p in _static_pool(league) if q in p["full_name"].lower()]
    # prefer active, then alphabetical
    matches.sort(key=lambda p: (not p.get("is_active", False), p["full_name"]))
    return [
        {"id": p["id"], "name": p["full_name"], "active": p.get("is_active", False)}
        for p in matches[:limit]
    ]


# --- live endpoint fetchers ------------------------------------------------

@lru_cache(maxsize=256)
def fetch_shots(player_id: int, season: str, league: League = DEFAULT) -> pd.DataFrame:
    if shotchartdetail is None:
        raise ShotFetchError("nba_api not installed.")
    # ShotChartDetail takes `league_id` (not `league_id_nullable`) — passing the
    # wrong name silently falls back to NBA data, which is wrong for the WNBA.
    raw = _call(
        shotchartdetail.ShotChartDetail,
        team_id=0,
        player_id=player_id,
        season_nullable=season,
        context_measure_simple="FGA",
        league_id=league.league_id,
    )
    rename = {
        "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
        "TEAM_ID": "team_id", "GAME_ID": "game_id", "GAME_DATE": "game_date",
        "LOC_X": "x", "LOC_Y": "y", "SHOT_MADE_FLAG": "made",
        "SHOT_ZONE_BASIC": "zone", "SHOT_DISTANCE": "distance",
        "PERIOD": "period", "SHOT_CLOCK": "shot_clock",
    }
    have = [c for c in rename if c in raw.columns]
    df = raw[have].rename(columns={k: v for k, v in rename.items() if k in have})
    for want in rename.values():
        if want not in df.columns:
            df[want] = pd.NA
    df["season"] = str(season)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["made"] = pd.to_numeric(df["made"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["x", "y"]).copy()
    if not df.empty:
        xmax, ymax = float(df["x"].abs().max()), float(df["y"].abs().max())
        if xmax <= 60 and ymax <= 120:
            df["x"] *= 10.0
            df["y"] *= 10.0
    return df


@lru_cache(maxsize=256)
def fetch_gamelog(player_id: int, season: str, league: League = DEFAULT) -> pd.DataFrame:
    if playergamelog is None:
        raise ShotFetchError("nba_api not installed.")
    raw = _call(
        playergamelog.PlayerGameLog,
        player_id=player_id,
        season=season,
        league_id_nullable=league.league_id,
    )
    if "GAME_DATE" in raw.columns:
        raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"], errors="coerce")
        raw = raw.sort_values("GAME_DATE").reset_index(drop=True)
    return raw


@lru_cache(maxsize=2048)
def fetch_birthdate(player_id: int, league: League = DEFAULT):
    if commonplayerinfo is None:
        return None
    try:
        df = _call(
            commonplayerinfo.CommonPlayerInfo,
            player_id=player_id,
            league_id_nullable=league.league_id,
        )
        return pd.to_datetime(df.iloc[0].get("BIRTHDATE"), errors="coerce")
    except Exception:
        return None
