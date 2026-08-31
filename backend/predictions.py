"""Forecasting built on the same local Parquet everything else reads.

Two models, deliberately simple and both backtested against a naive baseline so
the numbers on screen can be judged rather than taken on faith:

* **Teams** — Elo with a home-court term and a margin-of-victory update. It is
  the standard baseline for game prediction: interpretable, needs nothing but
  results, and doubles as a power ranking.
* **Players** — a Marcel-style projection: a weighted blend of recent seasons,
  regressed toward the league mean by playing time, with an age adjustment.

Neither is fed by a model that could invent numbers; both are arithmetic over
`data/*.parquet`. Every projection is reproducible from the inputs shown.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data
from .leagues import DEFAULT, League

# --- Elo -------------------------------------------------------------------
K = 20.0                # update size per game
HOME_ADVANTAGE = 65.0   # in Elo points; ~58% home win rate falls out of this
SEASON_REGRESSION = 0.25  # pulled back toward 1500 between seasons
BASE_RATING = 1500.0
ELO_TEST_FROM = 2015    # ratings need a few seasons to settle before judging

# --- Player projection -----------------------------------------------------
# Fitted on held-out seasons (see player_backtest), not chosen by taste. The
# heavy recency weighting is the finding that mattered: NBA per-game production
# is persistent year to year, and a flatter blend loses to "same as last
# season" for volume stats. Regression toward the mean is what earns its keep
# on rate stats, where a single season is noisy.
SEASON_WEIGHTS = (12, 3, 1)  # most recent season first
REGRESSION_GAMES = 10        # games of league-average play blended in
AGE_PEAK = 27.0
AGE_SLOPE_YOUNG = 0.003      # improvement per year below the peak
AGE_SLOPE_OLD = 0.003        # decline per year above it
PROJECTED_METRICS = ("pts", "reb", "ast", "stl", "blk", "tov", "ts_pct")
MIN_SEASONS_FOR_PROJECTION = 1
PROJECTION_MIN_GP = 20       # a season must be this long to inform a projection


@lru_cache(maxsize=4)
def games(league: League = DEFAULT) -> pd.DataFrame:
    """One row per completed game, rebuilt from the player game logs.

    The logs carry no score, but `MATCHUP` names both teams and marks home with
    "vs", so summing each side's points reconstructs the result. Games where
    only one side's box score is present are dropped rather than guessed at.
    """
    log = data.gamelog(league).copy()
    matchup = log["MATCHUP"].astype(str)
    # Relocations and rebrands are folded onto one abbreviation so a franchise
    # carries a single rating history rather than restarting at 1500.
    log["team"] = matchup.str.split().str[0].map(league.canonical_team)
    log["opp"] = matchup.str.split().str[-1].map(league.canonical_team)
    log["is_home"] = matchup.str.contains(" vs ")

    side = (log.groupby(["GAME_DATE", "team", "opp", "is_home"], as_index=False)
               .agg(pts=("PTS", "sum")))
    home = side[side.is_home].rename(columns={"team": "home", "opp": "away", "pts": "home_pts"})
    away = side[~side.is_home].rename(columns={"team": "away", "opp": "home", "pts": "away_pts"})
    out = home.merge(away[["GAME_DATE", "home", "away", "away_pts"]],
                     on=["GAME_DATE", "home", "away"], how="inner")

    out = out.drop(columns=["is_home"]).sort_values("GAME_DATE").reset_index(drop=True)
    # Only a season that crosses New Year is labelled by the year it ends in.
    # Applying the NBA rule to the WNBA dated its summer games a year ahead.
    if league.season_start_month >= 7:
        out["season"] = np.where(out.GAME_DATE.dt.month >= league.season_start_month,
                                 out.GAME_DATE.dt.year + 1, out.GAME_DATE.dt.year)
    else:
        out["season"] = out.GAME_DATE.dt.year
    out["margin"] = out.home_pts - out.away_pts
    out["home_win"] = out.margin > 0
    return out


def _elo_expected(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((home_rating + HOME_ADVANTAGE) - away_rating) / 400))


@lru_cache(maxsize=4)
def _elo_run(league: League = DEFAULT) -> tuple[dict[str, float], pd.DataFrame]:
    """Walk the games in order, returning final ratings and every prediction
    made along the way. Predictions are recorded *before* the game updates the
    ratings, so the record is a genuine out-of-sample history."""
    df = games(league)
    ratings: dict[str, float] = {}
    prev_season = None
    log: list[dict] = []

    for row in df.itertuples():
        if row.season != prev_season:
            for team in ratings:
                ratings[team] = BASE_RATING + (ratings[team] - BASE_RATING) * (1 - SEASON_REGRESSION)
            prev_season = row.season

        home_r = ratings.get(row.home, BASE_RATING)
        away_r = ratings.get(row.away, BASE_RATING)
        p_home = _elo_expected(home_r, away_r)
        actual = 1.0 if row.home_win else 0.0

        log.append({"season": int(row.season), "date": row.GAME_DATE,
                    "home": row.home, "away": row.away, "p_home": p_home,
                    "home_win": bool(row.home_win), "margin": float(row.margin)})

        # Margin of victory scales the update, damped for lopsided matchups so
        # a blowout by an already-strong team moves the rating less.
        mov = abs(row.margin)
        multiplier = ((mov + 3) ** 0.8) / (7.5 + 0.006 * abs((home_r + HOME_ADVANTAGE) - away_r))
        delta = K * multiplier * (actual - p_home)
        ratings[row.home] = home_r + delta
        ratings[row.away] = away_r - delta

    return ratings, pd.DataFrame(log)


@lru_cache(maxsize=4)
def points_per_elo(league: League = DEFAULT) -> float:
    """How many Elo points a point of margin is worth, fitted rather than
    assumed, so projected margins come from this league's own data."""
    _, log = _elo_run(league)
    if log.empty:
        return 25.0
    diff = np.log10(log.p_home.clip(1e-6, 1 - 1e-6) / (1 - log.p_home.clip(1e-6, 1 - 1e-6))) * 400
    slope = np.polyfit(diff, log.margin, 1)[0]
    return float(1 / slope) if slope else 25.0


def team_ratings(league: League = DEFAULT, active_only: bool = True) -> list[dict]:
    """Current power ranking, with each team's record in its latest season.

    Teams that no longer exist are excluded by default: they still carry a
    rating from the seasons they played, but they are not part of a ranking
    of who is good *now*."""
    ratings, _ = _elo_run(league)
    df = games(league)
    if df.empty:
        return []
    latest = int(df.season.max())
    season = df[df.season == latest]

    records: dict[str, list[int]] = {}
    for row in season.itertuples():
        for team, won in ((row.home, row.home_win), (row.away, not row.home_win)):
            rec = records.setdefault(team, [0, 0])
            rec[0 if won else 1] += 1

    out = []
    for team, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
        if active_only and team not in records:
            continue  # folded franchises keep a rating but not a place in the table
        wins, losses = records.get(team, [0, 0])
        out.append({
            "team": team,
            "elo": round(rating, 1),
            "wins": wins,
            "losses": losses,
            "games": wins + losses,
        })
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out


def predict_game(league: League, home: str, away: str) -> dict:
    """Win probability and projected margin for a hypothetical matchup."""
    ratings, _ = _elo_run(league)
    home_r = ratings.get(home)
    away_r = ratings.get(away)
    if home_r is None or away_r is None:
        missing = home if home_r is None else away
        raise KeyError(missing)

    p_home = _elo_expected(home_r, away_r)
    edge = (home_r + HOME_ADVANTAGE) - away_r
    return {
        "home": home, "away": away,
        "home_elo": round(home_r, 1), "away_elo": round(away_r, 1),
        "home_win_prob": round(p_home, 4),
        "away_win_prob": round(1 - p_home, 4),
        "projected_margin": round(edge / points_per_elo(league), 1),
        "home_advantage_elo": HOME_ADVANTAGE,
    }


def team_backtest(league: League = DEFAULT) -> dict:
    """How the Elo model actually did, per season, against always-pick-home.

    Accuracy alone flatters a model on a base rate this high, so Brier score
    and log loss are reported too — they judge the probability, not the pick.
    """
    _, log = _elo_run(league)
    if log.empty:
        return {"seasons": [], "overall": None}
    test = log[log.season >= ELO_TEST_FROM]
    if test.empty:
        test = log

    def metrics(frame: pd.DataFrame) -> dict:
        y = frame.home_win.astype(float).to_numpy()
        p = frame.p_home.to_numpy()
        picked = (p > 0.5).astype(float)
        base_rate = y.mean()
        return {
            "games": int(len(frame)),
            "accuracy": round(float((picked == y).mean()), 4),
            "baseline_accuracy": round(float(max(base_rate, 1 - base_rate)), 4),
            "brier": round(float(((p - y) ** 2).mean()), 4),
            "baseline_brier": round(float(((base_rate - y) ** 2).mean()), 4),
            "log_loss": round(float(-(y * np.log(np.clip(p, 1e-9, 1))
                                      + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))).mean()), 4),
        }

    seasons = [{"season": str(int(s)), **metrics(frame)}
               for s, frame in test.groupby("season")]
    return {"seasons": seasons, "overall": metrics(test),
            "test_from": str(ELO_TEST_FROM), "model": "elo"}


def calibration(league: League = DEFAULT, bins: int = 10) -> list[dict]:
    """Predicted probability vs. how often those games were actually won —
    the check that matters for a forecast people will read as a percentage."""
    _, log = _elo_run(league)
    test = log[log.season >= ELO_TEST_FROM]
    if test.empty:
        return []
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(test.p_home, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        rows = test[idx == b]
        if len(rows) < 25:
            continue
        out.append({
            "bin": f"{edges[b]:.0%}–{edges[b+1]:.0%}",
            "predicted": round(float(rows.p_home.mean()), 4),
            "actual": round(float(rows.home_win.mean()), 4),
            "games": int(len(rows)),
        })
    return out


# --------------------------------------------------------------------------
# Player projections
# --------------------------------------------------------------------------

def _age_multiplier(age: float | None) -> float:
    """A gentle curve: still improving before 27, declining after."""
    if age is None or not np.isfinite(age):
        return 1.0
    if age < AGE_PEAK:
        return 1.0 + (AGE_PEAK - age) * AGE_SLOPE_YOUNG
    return 1.0 - (age - AGE_PEAK) * AGE_SLOPE_OLD


def _season_start_year(league: League, season: str) -> int | None:
    return league.season_start_year(season)


@lru_cache(maxsize=4)
def _player_history(league: League = DEFAULT) -> pd.DataFrame:
    df = data.players(league).copy()
    df["season_int"] = pd.to_numeric(df["season"], errors="coerce")
    df["gp_num"] = pd.to_numeric(df.get("gp"), errors="coerce").fillna(0)
    return df.dropna(subset=["season_int"]).sort_values(["player_id", "season_int"])


@lru_cache(maxsize=4)
def _league_means(league: League = DEFAULT) -> dict[str, float]:
    df = _player_history(league)
    qualified = df[df.gp_num >= PROJECTION_MIN_GP]
    return {m: float(pd.to_numeric(qualified[m], errors="coerce").mean())
            for m in PROJECTED_METRICS if m in df.columns}


def project_player(league: League, player_id: int, target_season: str | None = None) -> dict:
    """Projected per-game line for a player's next season.

    Weighted blend of up to three prior seasons, regressed toward the league
    mean in proportion to how little the player actually played, then nudged by
    an age curve. Both the inputs and the weights are returned so the number
    can be checked by hand.
    """
    hist = _player_history(league)
    rows = hist[pd.to_numeric(hist["player_id"], errors="coerce") == player_id]
    if rows.empty:
        raise KeyError(player_id)

    name = str(rows.iloc[-1]["player_name"])
    latest_season = int(rows.iloc[-1]["season_int"])
    target = int(target_season) if target_season else latest_season + 1

    prior = rows[rows.season_int < target].tail(len(SEASON_WEIGHTS))
    if prior.empty or len(prior) < MIN_SEASONS_FOR_PROJECTION:
        raise ValueError(f"no seasons before {target} for {name}")

    # Most recent season gets the largest weight; a season is also worth more
    # the more of it was actually played.
    ordered = list(prior.itertuples())[::-1]
    weights, used = [], []
    for i, row in enumerate(ordered):
        base = SEASON_WEIGHTS[i] if i < len(SEASON_WEIGHTS) else 1
        base = max(base, 1)  # never let a season count for nothing
        weights.append(base * max(float(row.gp_num), 1.0))
        used.append(row)

    means = _league_means(league)
    age = None
    bio_age = _projected_age(league, player_id, target)
    if bio_age is not None:
        age = bio_age

    projected: dict[str, float | None] = {}
    inputs: dict[str, list] = {}
    for metric in PROJECTED_METRICS:
        if metric not in hist.columns:
            continue
        values, wts, seasons = [], [], []
        for w, row in zip(weights, used):
            v = pd.to_numeric(pd.Series([getattr(row, metric, None)]), errors="coerce").iloc[0]
            if pd.notna(v):
                values.append(float(v))
                wts.append(w)
                seasons.append(str(int(row.season_int)))
        if not values or sum(wts) <= 0:
            projected[metric] = None
            continue

        weighted = float(np.average(values, weights=wts))
        # Regress toward the league mean: a player with few games carries less
        # signal, so REGRESSION_GAMES of average play is blended in.
        played = sum(float(r.gp_num) for r in used)
        mean = means.get(metric, weighted)
        shrunk = ((weighted * played) + (mean * REGRESSION_GAMES)) / (played + REGRESSION_GAMES)
        adjusted = shrunk * (_age_multiplier(age) if metric != "tov" else 1.0)
        projected[metric] = round(adjusted, 3)
        inputs[metric] = [{"season": s, "value": round(v, 3)} for s, v in zip(seasons, values)]

    last = used[0]
    return {
        "player_id": int(player_id),
        "player_name": name,
        "target_season": str(target),
        "based_on": [str(int(r.season_int)) for r in used],
        "age_at_target": None if age is None else round(age, 1),
        "age_multiplier": round(_age_multiplier(age), 4),
        "projected": projected,
        "last_season": {
            "season": str(int(last.season_int)),
            **{m: (None if pd.isna(getattr(last, m, None)) else
                   round(float(getattr(last, m)), 3))
               for m in PROJECTED_METRICS if m in hist.columns},
        },
        "inputs": inputs,
        "model": "marcel-style weighted blend",
    }


def _projected_age(league: League, player_id: int, target_season: int) -> float | None:
    bio = data._load_optional("player_bio", league)
    if bio is None:
        return None
    row = bio[bio["player_id"].astype("int64") == int(player_id)]
    if row.empty:
        return None
    birth = pd.to_datetime(row.iloc[0].get("birthdate"), errors="coerce")
    if pd.isna(birth):
        return None
    start_year = _season_start_year(league, str(target_season))
    if start_year is None:
        return None
    start = pd.Timestamp(year=start_year, month=league.season_start_month, day=1)
    return float((start - birth).days / 365.25)


def player_backtest(league: League = DEFAULT, metric: str = "pts",
                    seasons_back: int = 6) -> dict:
    """Projection error against the honest baseline: "same as last season".

    A projection only earns its complexity if it beats that, so both are scored
    on the same held-out seasons.
    """
    hist = _player_history(league)
    if metric not in hist.columns:
        raise KeyError(metric)
    all_seasons = sorted(hist.season_int.dropna().unique())
    targets = [int(s) for s in all_seasons[-seasons_back:]]

    proj_err, naive_err, n = [], [], 0
    for target in targets:
        actual = hist[(hist.season_int == target) & (hist.gp_num >= PROJECTION_MIN_GP)]
        for row in actual.itertuples():
            pid = int(row.player_id)
            try:
                p = project_player(league, pid, str(target))
            except (KeyError, ValueError):
                continue
            predicted = p["projected"].get(metric)
            previous = p["last_season"].get(metric)
            truth = pd.to_numeric(pd.Series([getattr(row, metric, None)]), errors="coerce").iloc[0]
            if predicted is None or previous is None or pd.isna(truth):
                continue
            proj_err.append(abs(predicted - float(truth)))
            naive_err.append(abs(float(previous) - float(truth)))
            n += 1

    if not proj_err:
        return {"metric": metric, "players": 0}
    return {
        "metric": metric,
        "players": n,
        "seasons": [str(t) for t in targets],
        "projection_mae": round(float(np.mean(proj_err)), 3),
        "baseline_mae": round(float(np.mean(naive_err)), 3),
        "improvement": round(float(1 - np.mean(proj_err) / np.mean(naive_err)), 4),
        "baseline": "same as last season",
    }


@lru_cache(maxsize=4)
def project_all(league: League = DEFAULT) -> list[dict]:
    """Every player with a recent, substantial season, projected forward.

    Restricted to players who actually appeared in the latest season on file —
    a projection for someone who last played in 2011 is noise, not a forecast.
    """
    hist = _player_history(league)
    if hist.empty:
        return []
    latest = int(hist.season_int.max())
    active = hist[(hist.season_int == latest) & (hist.gp_num >= PROJECTION_MIN_GP)]

    out = []
    for pid in active["player_id"].dropna().unique():
        try:
            p = project_player(league, int(pid))
        except (KeyError, ValueError):
            continue
        row = {
            "player_id": p["player_id"],
            "player_name": p["player_name"],
            "target_season": p["target_season"],
            "age_at_target": p["age_at_target"],
            "last_season": p["last_season"]["season"],
        }
        for metric, value in p["projected"].items():
            row[metric] = value
            previous = p["last_season"].get(metric)
            row[f"{metric}_last"] = previous
            row[f"{metric}_delta"] = (None if value is None or previous is None
                                      else round(value - previous, 3))
        out.append(row)
    return out


def projected_leaderboard(league: League = DEFAULT, metric: str = "pts",
                          limit: int = 25, order: str = "desc") -> list[dict]:
    """Top projected performers, or the biggest movers when ordering by delta."""
    rows = [r for r in project_all(league) if r.get(metric) is not None]
    key = metric if not metric.endswith("_delta") else metric
    rows.sort(key=lambda r: (r.get(key) is None, r.get(key)), reverse=(order == "desc"))
    return rows[:limit]
