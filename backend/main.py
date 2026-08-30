"""FastAPI backend for the basketball data dashboard (NBA + WNBA)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import binned_statistic_2d
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from . import data, leagues, live

# Minimum games played to include a row in the pool used to compute radar
# percentile ranks. Keeps low-sample bench players from polluting the axes.
MIN_GP_FOR_RADAR = 15


def _percentile_map(pool: pd.DataFrame, feats: list[str]) -> dict[str, dict[str, float]]:
    """Per-metric {player_name: percentile_rank in [0,1]}; larger = better
    (drtg/tov are inverted so lower raw value becomes higher percentile)."""
    out: dict[str, dict[str, float]] = {}
    if pool.empty or "player_name" not in pool.columns:
        return out
    for f in feats:
        col = pd.to_numeric(pool.get(f), errors="coerce")
        rk = col.rank(pct=True, ascending=(f not in data.INVERT_METRICS))
        out[f] = dict(zip(pool["player_name"], rk))
    return out


def _gp_filtered_pool(df: pd.DataFrame) -> pd.DataFrame:
    if "gp" not in df.columns:
        return df
    gp = pd.to_numeric(df["gp"], errors="coerce").fillna(0)
    filt = df[gp >= MIN_GP_FOR_RADAR]
    # fall back to the full pool if the filter empties it
    return filt if len(filt) >= 30 else df

app = FastAPI(title="Hoops Data Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FileNotFoundError)
def _missing_data(request: Request, exc: FileNotFoundError):
    """A league with no Parquet on disk is a expected, actionable state --
    report it as 404 with the ETL command rather than a 500."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/leagues")
def league_list():
    """Which leagues exist and which actually have Parquet data on disk."""
    return {
        "leagues": [
            {"key": lg.key, "label": lg.label, "available": data.has_data(lg)}
            for lg in leagues.LEAGUES.values()
        ],
        "default": leagues.DEFAULT.key,
    }


@app.get("/api/meta")
def meta(league: str | None = None):
    lg = leagues.get(league)
    return {
        "league": lg.key,
        "league_label": lg.label,
        "players": data.player_names(lg),
        "teams": data.team_names(lg),
        "seasons": data.seasons(lg),
        "metrics": data.available_metrics(lg),
        "invert_metrics": sorted(data.INVERT_METRICS),
        "court": {"arc": lg.three_point_arc, "corner": lg.three_point_corner},
    }


@app.get("/api/compare")
def compare(
    players: str = Query(..., description="comma-separated player names"),
    metrics: str = Query(..., description="comma-separated metric keys"),
    seasonLo: str | None = None,
    seasonHi: str | None = None,
    league: str | None = None,
):
    lg = leagues.get(league)
    names = [p for p in players.split(",") if p]
    mets = [m for m in metrics.split(",") if m and m in data.available_metrics(lg)]
    if not names or not mets:
        return {"rows": [], "single_season": False, "season_label": None}
    df = data.players(lg)
    df = df[df["player_name"].isin(names)]
    allowed = data.allowed_seasons(seasonLo, seasonHi, lg)
    if allowed:
        df = df[df["season"].isin(allowed)]
    if df.empty:
        return {"rows": [], "single_season": False, "season_label": None}
    df = data.clean_numeric(df, mets)
    cols = ["player_name", "season"] + mets
    df = df[cols].sort_values(["player_name", "season"])

    single = df["season"].nunique() == 1
    out = {
        "rows": _json_safe(data.records(df)),
        "metrics": mets,
        "players": names,
        "single_season": bool(single),
        "season_label": df["season"].iloc[0] if single else None,
    }

    # Radar uses league-percentile ranks so real profile differences show up
    # instead of everyone landing in the same mid-range band.
    if single:
        season = out["season_label"]
        all_players = data.players(lg)
        pool_cols = list(set(mets + ["gp", "player_name"]) & set(all_players.columns))
        raw_pool = all_players[all_players["season"] == season][pool_cols]
        pool = _gp_filtered_pool(data.clean_numeric(raw_pool, mets))
        pct = _percentile_map(pool, mets)
        norm = {}
        for _, row in df.iterrows():
            vals = []
            for m in mets:
                v = pct.get(m, {}).get(row["player_name"])
                if v is None or pd.isna(v):
                    v = 0.5
                vals.append(round(float(v), 4))
            norm[row["player_name"]] = vals
        out["radar"] = {"features": mets, "values": norm}
    return out


@app.get("/api/trends")
def trends(
    player: str,
    metrics: str,
    leagueAvg: bool = True,
    league: str | None = None,
):
    lg = leagues.get(league)
    mets = [m for m in metrics.split(",") if m and m in data.available_metrics(lg)]
    if not mets:
        return {"player": player, "metrics": [], "series": []}
    df = data.players(lg)
    pdf = data.clean_numeric(df[df["player_name"] == player], mets)
    pdf = pdf[["season"] + mets].sort_values("season")
    series = []
    for _, row in pdf.iterrows():
        for m in mets:
            series.append({"season": row["season"], "metric": m, "value": row[m], "source": player})
    if leagueAvg:
        avg = data.clean_numeric(df, mets).groupby("season")[mets].mean(numeric_only=True).reset_index()
        for _, row in avg.iterrows():
            for m in mets:
                series.append({"season": row["season"], "metric": m, "value": row[m], "source": "League avg"})
    return _json_safe({"player": player, "metrics": mets, "series": series})


@app.get("/api/percentiles")
def percentiles(player: str, season: str, league: str | None = None):
    lg = leagues.get(league)
    df = data.players(lg)
    mets = data.available_metrics(lg)
    pool = df[df["season"] == season].copy()
    sub = pool[pool["player_name"] == player]
    if sub.empty:
        raise HTTPException(404, "Player/season not found")
    row = sub.iloc[0]
    rows = []
    for m in mets:
        col = pd.to_numeric(pool[m], errors="coerce")
        if col.notna().sum() < 10:
            continue
        rank = col.rank(pct=True, ascending=(m not in data.INVERT_METRICS))
        idx = pool.index[pool["player_name"] == player]
        if not len(idx):
            continue
        pct = float(rank.loc[idx[0]])
        val = float(row[m]) if pd.notna(row[m]) else None
        rows.append({"metric": m, "value": val, "percentile": round(pct * 100, 1)})
    rows.sort(key=lambda r: r["percentile"])
    return _json_safe({"player": player, "season": season, "rows": rows})


class SimilarRequest(BaseModel):
    anchor: str
    k: int = 5
    weights: dict[str, float] = {}
    seasonLo: str | None = None
    seasonHi: str | None = None
    league: str | None = None


@app.post("/api/similar")
def similar(req: SimilarRequest):
    lg = leagues.get(req.league)
    desired = ["pts", "ast", "reb", "tov", "ts_pct", "usg_pct", "ortg", "drtg"]
    cols = data.players(lg).columns
    feats = [f for f in desired if f in cols]
    if not feats:
        raise HTTPException(400, "No similarity features available")
    df = data.players(lg).copy()
    allowed = data.allowed_seasons(req.seasonLo, req.seasonHi, lg)
    if allowed:
        df = df[df["season"].isin(allowed)]
    agg_cols = feats + (["gp"] if "gp" in df.columns else [])
    agg = df.groupby("player_name", as_index=False)[agg_cols].mean(numeric_only=True)
    agg = agg.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if req.anchor not in agg["player_name"].values:
        raise HTTPException(404, "Anchor player not found in range")
    X = agg[feats].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)
    w = np.array([float(req.weights.get(f, 1.0)) for f in feats], dtype=float)
    Xz = Xz * w
    sim = cosine_similarity(Xz)
    idx = int(agg.index[agg["player_name"] == req.anchor][0])
    scores = sim[idx]
    order = np.argsort(-scores)
    matches = []
    for i in order:
        if int(i) == idx:
            continue
        matches.append(
            {"player_name": agg.iloc[int(i)]["player_name"], "similarity": float(scores[int(i)])}
        )
        if len(matches) >= req.k:
            break

    radar_pool = _gp_filtered_pool(agg)
    pool_sorted = {
        f: np.sort(pd.to_numeric(radar_pool[f], errors="coerce").dropna().to_numpy())
        for f in feats
    }

    def norm_row(pname: str) -> list[float]:
        r = agg.loc[agg["player_name"] == pname, feats].iloc[0]
        out = []
        for f in feats:
            arr = pool_sorted[f]
            v = float(r[f])
            if arr.size == 0 or pd.isna(v):
                out.append(0.5)
                continue
            rank = float(np.searchsorted(arr, v, side="right")) / len(arr)
            if f in data.INVERT_METRICS:
                rank = 1.0 - rank
            out.append(round(max(0.0, min(1.0, rank)), 4))
        return out

    radar = {
        "features": feats,
        "anchor": {"name": req.anchor, "values": norm_row(req.anchor)},
        "peers": [{"name": m["player_name"], "values": norm_row(m["player_name"])} for m in matches],
    }
    return _json_safe({"features": feats, "matches": matches, "radar": radar})


@app.get("/api/gamelog")
def gamelog(
    player: str,
    season: str,
    stat: str = "PTS",
    window: int = 10,
    league: str | None = None,
):
    lg = leagues.get(league)
    local = data.gamelog(lg)
    if local is not None:
        glog = local[(local["player_name"] == player) & (local["season"] == str(season))].copy()
        if glog.empty and player not in set(local["player_name"]):
            raise HTTPException(404, f"No {lg.label} game log for '{player}'")
    else:
        pid = live.find_player_id(player, lg)
        if not pid:
            raise HTTPException(404, f"Player '{player}' not found in the {lg.label}")
        try:
            glog = live.fetch_gamelog(pid, season, lg)
        except Exception as e:
            raise HTTPException(502, live.friendly_upstream_message(e, lg)) from e
    if glog.empty or stat not in glog.columns:
        return {"player": player, "season": season, "stat": stat, "window": window, "games": []}
    g = glog.copy()
    g[stat] = pd.to_numeric(g[stat], errors="coerce")
    g["rolling"] = g[stat].rolling(window, min_periods=1).mean()
    games = []
    for _, r in g.iterrows():
        games.append(
            {
                "date": r.get("GAME_DATE").isoformat() if pd.notna(r.get("GAME_DATE")) else None,
                "matchup": r.get("MATCHUP"),
                "value": r[stat],
                "rolling": r["rolling"],
            }
        )
    return _json_safe(
        {
            "player": player,
            "season": season,
            "stat": stat,
            "window": window,
            "season_avg": float(g[stat].mean()) if g[stat].notna().any() else None,
            "games": games,
        }
    )


class AgeRequest(BaseModel):
    players: list[str]
    metric: str
    league: str | None = None


@app.post("/api/age-curves")
def age_curves(req: AgeRequest):
    lg = leagues.get(req.league)
    if req.metric not in data.available_metrics(lg):
        raise HTTPException(400, "metric unavailable")
    local_bd = data.birthdates(lg)
    curves = []
    for pname in req.players[:5]:
        sub = data.players(lg)
        sub = sub[sub["player_name"] == pname].copy()
        if sub.empty:
            curves.append({"player": pname, "points": [], "error": "not found"})
            continue
        pid = int(sub["player_id"].iloc[0])
        bd = local_bd.get(pid) if local_bd else live.fetch_birthdate(pid, lg)
        if bd is None or pd.isna(bd):
            curves.append({"player": pname, "points": [], "error": "no birthdate"})
            continue
        points = []
        for _, r in sub.sort_values("season").iterrows():
            s = str(r["season"])
            try:
                y = int(s.split("-")[0])
                start = pd.Timestamp(year=y, month=lg.season_start_month, day=1)
                age = round((start - bd).days / 365.25, 2)
            except Exception:
                continue
            v = pd.to_numeric(r.get(req.metric), errors="coerce")
            if pd.notna(v):
                points.append({"season": s, "age": age, "value": float(v)})
        curves.append({"player": pname, "points": points})
    return _json_safe({"metric": req.metric, "curves": curves})


def _four_factors(row: pd.Series) -> dict:
    fga = float(row.get("FGA") or 0)
    fta = float(row.get("FTA") or 0)
    fgm = float(row.get("FGM") or 0)
    fg3m = float(row.get("FG3M") or 0)
    tov = float(row.get("TOV") or 0)
    oreb = float(row.get("OREB") or 0)
    dreb = float(row.get("DREB") or 0)
    efg = ((fgm + 0.5 * fg3m) / fga) if fga else None
    poss = fga + 0.44 * fta + tov
    tovp = (tov / poss) if poss else None
    orb = (oreb / (oreb + dreb)) if (oreb + dreb) else None
    ftr = (fta / fga) if fga else None
    return {"eFG%": efg, "TOV%": tovp, "ORB%": orb, "FT rate": ftr}


@app.get("/api/teams/series")
def team_series(team: str, league: str | None = None):
    tdf = data.teams(leagues.get(league)).copy()
    tdf = tdf[tdf["team_name"] == team]
    if tdf.empty:
        raise HTTPException(404, "team not found")
    cols = [
        "wins", "losses", "FG_PCT", "FT_PCT", "FG3_PCT", "ortg", "drtg", "pace",
        "FGM", "FGA", "FG3M", "FTA", "TOV", "OREB", "DREB",
    ]
    tdf = data.clean_numeric(tdf, cols).sort_values("season")
    return _json_safe(
        {
            "team": team,
            "rows": data.records(tdf[["season"] + [c for c in cols if c in tdf.columns]]),
        }
    )


@app.get("/api/teams/factors")
def team_factors(team: str, season: str, league: str | None = None):
    tdf = data.teams(leagues.get(league)).copy()
    needed = {"FGM", "FGA", "FG3M", "FTA", "TOV", "OREB", "DREB"}
    if not needed.issubset(set(tdf.columns)):
        raise HTTPException(400, f"Missing columns: {sorted(needed - set(tdf.columns))}")
    tdf = data.clean_numeric(tdf, list(needed))
    team_row = tdf[(tdf["team_name"] == team) & (tdf["season"] == season)]
    if team_row.empty:
        raise HTTPException(404, "team/season not found")
    team_ff = _four_factors(team_row.iloc[0])
    lg = tdf[tdf["season"] == season]
    lg_ff_rows = lg.apply(_four_factors, axis=1, result_type="expand")
    lg_avg = lg_ff_rows.mean(numeric_only=True).to_dict()
    return _json_safe({"team": team, "season": season, "team_ff": team_ff, "league_avg": lg_avg})


@app.get("/api/shots")
def shots(
    player: str,
    season: str,
    mode: str = Query("scatter", pattern="^(scatter|hex)$"),
    league: str | None = None,
):
    lg = leagues.get(league)
    local = data.shots(lg)
    if local is not None:
        sdf = local[(local["player_name"] == player) & (local["season"] == str(season))]
        if sdf.empty and player not in set(local["player_name"]):
            raise HTTPException(404, f"No {lg.label} shot data for '{player}'")
    else:
        pid = live.find_player_id(player, lg)
        if not pid:
            raise HTTPException(404, f"Player '{player}' not found in the {lg.label}")
        try:
            sdf = live.fetch_shots(pid, season, lg)
        except Exception as e:
            raise HTTPException(502, live.friendly_upstream_message(e, lg)) from e
    if sdf.empty:
        return {"player": player, "season": season, "mode": mode, "shots": [], "hexes": []}
    if mode == "scatter":
        # cap at 2500 points to keep payload light
        out = sdf[["x", "y", "made"]].head(2500)
        return _json_safe(
            {
                "player": player,
                "season": season,
                "mode": "scatter",
                "count": int(len(sdf)),
                "fg_pct": float(sdf["made"].mean()),
                "shots": out.to_dict(orient="records"),
            }
        )
    # hexbin mode
    x = sdf["x"].to_numpy()
    y = sdf["y"].to_numpy()
    m = sdf["made"].to_numpy()
    xb = np.linspace(-250, 250, 26)
    yb = np.linspace(-52.5, 417.5, 24)
    cnt = binned_statistic_2d(x, y, np.ones_like(x), statistic="count", bins=[xb, yb]).statistic
    pct = binned_statistic_2d(x, y, m, statistic="mean", bins=[xb, yb]).statistic
    cx = 0.5 * (xb[:-1] + xb[1:])
    cy = 0.5 * (yb[:-1] + yb[1:])
    hexes = []
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            c = int(cnt[i, j])
            if c < 2:
                continue
            p = float(pct[i, j])
            hexes.append({"x": float(cx[i]), "y": float(cy[j]), "count": c, "pct": p})
    return _json_safe(
        {
            "player": player,
            "season": season,
            "mode": "hex",
            "count": int(len(sdf)),
            "fg_pct": float(sdf["made"].mean()),
            "hexes": hexes,
        }
    )


@app.get("/api/player-search")
def player_search(q: str = "", limit: int = 12, league: str | None = None):
    return {"results": live.search_players(q, limit=limit, league=leagues.get(league))}
