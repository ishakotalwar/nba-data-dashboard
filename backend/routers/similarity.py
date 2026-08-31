"""Player-season similarity.

Every qualifying player-season in the dataset is a candidate, so the answer to
"who had a season like this one" can come from any year, not just the current
one. Similarity is weighted cosine over z-scored features; the same normalized
features drive the per-result explanations, so the numbers and the prose agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

from .. import analytics, data, leagues

router = APIRouter(prefix="/api", tags=["similarity"])

# Weight presets. "Overall" is uniform; the rest tilt toward one skill.
PRESETS: dict[str, dict[str, float]] = {
    "Overall": {},
    "Scoring": {"pts": 2.5, "usg_pct": 1.8, "ts_pct": 1.5, "ortg": 1.0,
                "ast": 0.5, "reb": 0.5, "tov": 0.5, "drtg": 0.5},
    "Shooting": {"ts_pct": 2.5, "ortg": 1.5, "pts": 1.2, "usg_pct": 0.8,
                 "ast": 0.4, "reb": 0.4, "tov": 0.4, "drtg": 0.4},
    "Playmaking": {"ast": 2.5, "tov": 1.8, "usg_pct": 1.2, "ortg": 1.0,
                   "pts": 0.8, "ts_pct": 0.6, "reb": 0.4, "drtg": 0.4},
    "Defense": {"drtg": 2.5, "reb": 1.5, "tov": 0.6, "pts": 0.4, "ast": 0.4,
                "ts_pct": 0.3, "usg_pct": 0.3, "ortg": 0.4},
}

# Plain-language names for what a feature represents, used in explanations.
FEATURE_PHRASES = {
    "pts": "scoring volume",
    "ast": "assist rate",
    "reb": "rebounding",
    "tov": "turnovers",
    "ts_pct": "shooting efficiency",
    "usg_pct": "usage",
    "ortg": "offensive rating",
    "drtg": "defensive rating",
}

MIN_GP_DEFAULT = 20


class SimilarityRequest(BaseModel):
    player_id: int
    season: str
    league: str | None = None
    preset: str = "Overall"
    weights: dict[str, float] = {}
    k: int = 8
    min_gp: int = MIN_GP_DEFAULT
    same_season_only: bool = False


@router.get("/similarity/presets")
def presets():
    return {"presets": list(PRESETS), "features": analytics.SIMILARITY_FEATURES,
            "phrases": FEATURE_PHRASES}


@router.post("/similarity")
def similarity(req: SimilarityRequest):
    lg = leagues.get(req.league)
    df = data.players(lg)
    feats = [f for f in analytics.SIMILARITY_FEATURES if f in df.columns]
    if not feats:
        raise HTTPException(400, "No similarity features available")

    pool = df.copy()
    if "gp" in pool.columns:
        pool = pool[pd.to_numeric(pool["gp"], errors="coerce").fillna(0) >= req.min_gp]
    if req.same_season_only:
        pool = pool[pool["season"] == str(req.season)]
    pool = pool.dropna(subset=feats).reset_index(drop=True)
    if pool.empty:
        raise HTTPException(404, "No qualifying player-seasons")

    ids = pd.to_numeric(pool["player_id"], errors="coerce")
    anchor_mask = (ids == req.player_id) & (pool["season"] == str(req.season))
    if not anchor_mask.any():
        raise HTTPException(
            404,
            f"{req.season} for player {req.player_id} is not in the qualifying pool "
            f"(needs at least {req.min_gp} games)",
        )
    anchor_i = int(np.flatnonzero(anchor_mask.to_numpy())[0])

    # z-score across the whole pool, then weight
    X = StandardScaler().fit_transform(pool[feats].to_numpy(dtype=float))
    base = PRESETS.get(req.preset, {})
    w = np.array([float(req.weights.get(f, base.get(f, 1.0))) for f in feats], dtype=float)
    Xw = X * w

    a = Xw[anchor_i]
    denom = np.linalg.norm(Xw, axis=1) * (np.linalg.norm(a) or 1.0)
    sims = np.divide(Xw @ a, np.where(denom == 0, np.nan, denom))

    order = np.argsort(-np.nan_to_num(sims, nan=-np.inf))
    matches = []
    for i in order:
        i = int(i)
        if i == anchor_i:
            continue  # a season is not similar to itself
        r = pool.iloc[i]
        # Explanation: per-feature distance in the same normalized space.
        gaps = np.abs(X[i] - X[anchor_i])
        ranked = sorted(zip(feats, gaps), key=lambda t: t[1])
        matches.append({
            "player_id": int(r["player_id"]),
            "player_name": str(r["player_name"]),
            "season": str(r["season"]),
            "team": r.get("team_abbr"),
            "similarity": float(sims[i]),
            "values": {f: float(r[f]) for f in feats},
            "most_similar": [FEATURE_PHRASES.get(f, f) for f, _ in ranked[:4]],
            "biggest_difference": [FEATURE_PHRASES.get(f, f) for f, _ in ranked[-1:]],
        })
        if len(matches) >= req.k:
            break

    ar = pool.iloc[anchor_i]
    return analytics.json_safe({
        "anchor": {
            "player_id": int(ar["player_id"]), "player_name": str(ar["player_name"]),
            "season": str(ar["season"]), "team": ar.get("team_abbr"),
            "values": {f: float(ar[f]) for f in feats},
        },
        "features": feats,
        "preset": req.preset,
        "weights": {f: float(w[i]) for i, f in enumerate(feats)},
        "pool_size": int(len(pool)),
        "matches": matches,
        # Radar uses percentile-of-pool so shapes are comparable across eras.
        "radar": _radar(pool, feats, [anchor_i] + [
            int(np.flatnonzero((ids == m["player_id"]) & (pool["season"] == m["season"]))[0])
            for m in matches
        ]),
    })


def _radar(pool: pd.DataFrame, feats: list[str], idxs: list[int]) -> dict:
    ranks = {f: analytics.percentile_series(pool, f).to_numpy() for f in feats}
    out = []
    for i in idxs:
        r = pool.iloc[i]
        out.append({
            "name": f"{r['season']} {r['player_name']}",
            "values": [round(float(ranks[f][i]), 4) for f in feats],
        })
    return {"features": feats, "series": out}
