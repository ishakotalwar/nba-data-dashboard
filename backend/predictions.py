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
LIVE_MIN_GAMES = 3           # inside a season in progress, just a noise screen
MAX_MINUTES = 38.0           # nobody plays the whole game
MINUTE_CONCENTRATION = 2.0   # fitted by eye against real rotations; see _fit_to_team_minutes
# Statuses that mean the player will not feature, so they get no line at all.
UNAVAILABLE = __import__("re").compile(r"out|suspend|inactive", __import__("re").I)


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


def next_season(league: League = DEFAULT) -> int:
    """The season every projection targets by default: the one after the last
    season on file."""
    hist = _player_history(league)
    return int(hist.season_int.max()) + 1


def project_player(league: League, player_id: int, target_season: str | None = None,
                   include_target: bool = False) -> dict:
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

    # Strictly before the target by default: a backtest must not see the season
    # it is predicting. `include_target` is for a season already under way,
    # where games played so far are legitimately known — and where ignoring
    # them means projecting a live season from last year's form.
    mask = rows.season_int <= target if include_target else rows.season_int < target
    prior = rows[mask].tail(len(SEASON_WEIGHTS))
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


@lru_cache(maxsize=8)
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


# --- Player lines for a single game ----------------------------------------
# A game line is the season projection above, put through the two things that
# separate one night from a player's average night and can be read off the data
# we hold: who they are playing, and where.
GAME_LINE_METRICS = ("pts", "reb", "ast", "stl", "blk", "tov")
ROTATION_SIZE = 8
HOME_SCORING_EDGE = 0.015  # home teams score ~1.5% more than the same team away
DEFENCE_SHRINK = 0.5       # points allowed also encodes pace, so only half of
                           # the gap from average is treated as real defence


@lru_cache(maxsize=8)
def _defence_factors(league: League, season: int) -> dict[str, float]:
    """Points each team allowed per game, relative to the league average.

    Measured over the most recent season completed before `season`, so a game
    is never predicted using its own result.
    """
    g = games(league)
    prior = g[g.season < season]
    if prior.empty:
        return {}
    prior = prior[prior.season == prior.season.max()]
    allowed = pd.concat([
        prior[["home", "away_pts"]].rename(columns={"home": "team", "away_pts": "pts"}),
        prior[["away", "home_pts"]].rename(columns={"away": "team", "home_pts": "pts"}),
    ])
    per_team = allowed.groupby("team")["pts"].mean()
    league_mean = float(per_team.mean())
    if not league_mean:
        return {}
    return {str(t): 1.0 + DEFENCE_SHRINK * (float(v) / league_mean - 1.0)
            for t, v in per_team.items()}


# Counting stats a team can only produce so many of in a game; a rate like
# true shooting is a property of the player, not a share of a team total.
TEAM_TOTAL_METRICS = ("pts", "reb", "ast", "stl", "blk", "tov")
# How far a team's projected total is pulled toward the league average. Rates
# taken from five players' separate pasts assume five separate balls, so the
# raw sum runs hot; the exponent keeps better teams ahead of worse ones while
# closing most of the gap to what a team actually produces.
TEAM_TOTAL_SHRINK = 0.35


@lru_cache(maxsize=8)
def _league_team_totals(league: League) -> dict[str, float]:
    """What one team actually puts up in a game, averaged over the league."""
    log = data.gamelog(league)
    if log is None or log.empty:
        return {}
    frame = log.copy()
    frame["team"] = frame["MATCHUP"].astype(str).str.split().str[0]
    cols = {"pts": "PTS", "reb": "REB", "ast": "AST",
            "stl": "STL", "blk": "BLK", "tov": "TOV"}
    have = {k: v for k, v in cols.items() if v in frame.columns}
    if not have:
        return {}
    per_game = frame.groupby(["GAME_DATE", "team"])[list(have.values())].sum()
    return {k: float(per_game[v].mean()) for k, v in have.items()}


@lru_cache(maxsize=8)
def _injury_index(league: League) -> dict[int, dict]:
    """player_id -> their current injury entry, empty when none is on disk."""
    df = data.injuries(league)
    if df is None or df.empty:
        return {}
    out: dict[int, dict] = {}
    for r in df.itertuples():
        try:
            pid = int(r.player_id)
        except (TypeError, ValueError):
            continue
        out[pid] = {
            "status": str(getattr(r, "status", "") or ""),
            "type": str(getattr(r, "type", "") or ""),
            "detail": str(getattr(r, "detail", "") or ""),
            "comment": str(getattr(r, "comment", "") or ""),
        }
    return out


@lru_cache(maxsize=8)
def _roster_teams(league: League, season: int) -> dict[int, str] | None:
    """player_id -> the team they are on for `season`, or None if no roster
    for that season has been fetched."""
    df = data.roster(league)
    if df is None or df.empty:
        return None
    rows = df[df["season"].astype(str) == str(season)]
    if rows.empty:
        return None
    return {int(r.player_id): league.canonical_team(str(r.team))
            for r in rows.itertuples() if pd.notna(r.player_id)}


def roster_source(league: League, season: int) -> str:
    """Which of the two roster sources a projection for `season` is using."""
    return "roster" if _roster_teams(league, int(season)) is not None else "last-season"


@lru_cache(maxsize=8)
def _rotations(league: League, target_season: int) -> dict[str, list[dict]]:
    """Each team's rotation for `target_season`, every player projected.

    The roster is whoever finished the previous season on the team, ordered by
    minutes — the schedule carries no roster, and trades and signings made
    after that season are not in the data.
    """
    hist = _player_history(league)
    # A season with games already played is the best description of who is
    # playing and how well; only fall back to earlier seasons before tip-off.
    live = bool((hist.season_int == target_season).any())
    pool = hist[hist.season_int <= target_season] if live else hist[hist.season_int < target_season]
    if pool.empty:
        return {}
    latest = int(pool.season_int.max())
    # Games played measures availability, not role: a player at 30 minutes a
    # night is in the rotation whether they have appeared 11 times or 41. So
    # within a live season the games floor only screens out one-off cameos,
    # and the minutes sort plus the top-N cut decide who actually shows.
    # Filtering on games here dropped players like an MVP missing time injured.
    floor = LIVE_MIN_GAMES if live else PROJECTION_MIN_GP
    active = pool[(pool.season_int == latest) & (pool.gp_num >= floor)]

    # Where the player actually is *now*. Falls back to last season's team when
    # no roster has been published for the target season yet.
    signed = _roster_teams(league, target_season)
    injury_index = _injury_index(league)

    out: dict[str, list[dict]] = {}
    for row in active.itertuples():
        pid = int(row.player_id)
        if signed is not None:
            team = signed.get(pid)
            if team is None:
                continue  # not on any roster: retired, unsigned, or overseas
        else:
            team = str(getattr(row, "team_abbr", "") or "")
        if not team:
            continue
        injury = injury_index.get(pid)
        if injury and UNAVAILABLE.search(injury.get("status", "")):
            continue  # ruled out: no line, and their minutes go to team-mates

        try:
            p = project_player(league, int(row.player_id), str(target_season),
                               include_target=live)
        except (KeyError, ValueError):
            continue
        out.setdefault(league.canonical_team(team), []).append({
            "player_id": p["player_id"],
            "player_name": p["player_name"],
            # Flag only. With no archive of past injury reports there is
            # nothing to fit an availability model on, so the honest move
            # is to show the status and leave the projection alone.
            "injury": injury,
            "minutes": round(float(getattr(row, "min", 0.0) or 0.0), 1),
            "games_played": int(row.gp_num),
            "based_on": p["based_on"],
            "projected": {m: p["projected"].get(m) for m in GAME_LINE_METRICS},
        })
    for team in out:
        out[team].sort(key=lambda r: r["minutes"], reverse=True)
        _fit_to_team_minutes(league, out[team])
    return out


def _fit_to_team_minutes(league: League, players: list[dict]) -> None:
    """Rescale a rotation so it fits in one game, in place.

    Per-game averages come from each player's own past, where they had their
    own minutes and their own role. Summed over a squad that has since been
    rebuilt, they describe a team that would need far more than 48 minutes a
    night — Philadelphia's top eight projected 142 points across 243 minutes.

    Minutes are the fixed resource, so they are shared out in proportion to
    what each player has earned, and every rate is carried across at the
    player's own per-minute production. A team-mate ruled out is simply absent
    from the split, so his minutes flow to the others.
    """
    baseline = [max(float(r["minutes"] or 0.0), 0.0) for r in players]
    total = sum(baseline)
    if total <= 0:
        return

    budget = float(league.team_minutes)
    # Weighting by minutes squared rather than minutes: a coach concentrates a
    # rotation on his best players, so a starter keeps most of his workload
    # while the deep bench absorbs the squeeze. A flat split left a stacked
    # team's stars on 25 minutes, which no rotation looks like.
    weights = [b ** MINUTE_CONCENTRATION for b in baseline]
    pool = sum(weights) or 1.0
    allotted = [w / pool * budget for w in weights]

    # Nobody plays a whole game. Anything over the cap goes back to the others.
    for _ in range(3):
        spill = 0.0
        free = []
        for i, m in enumerate(allotted):
            if m > MAX_MINUTES:
                spill += m - MAX_MINUTES
                allotted[i] = MAX_MINUTES
            else:
                free.append(i)
        if spill <= 0.01 or not free:
            break
        free_pool = sum(weights[i] for i in free) or 1.0
        for i in free:
            allotted[i] += spill * weights[i] / free_pool

    # The last redistribution can push someone back over, so clamp once more.
    # A short-handed side then simply does not fill the budget, which is the
    # honest outcome: there is nobody left to give the minutes to.
    allotted = [min(m, MAX_MINUTES) for m in allotted]

    for row, was, now in zip(players, baseline, allotted):
        factor = (now / was) if was > 0 else 0.0
        row["baseline_minutes"] = round(was, 1)
        row["minutes"] = round(now, 1)
        row["projected"] = {k: (None if v is None else v * factor)
                            for k, v in row["projected"].items()}

    _shrink_team_totals(league, players)
    for row in players:
        row["projected"] = {k: (None if v is None else round(v, 2))
                            for k, v in row["projected"].items()}


def _shrink_team_totals(league: League, players: list[dict]) -> None:
    """Pull each team total toward what a team really produces, in place.

    Minutes alone do not fix a squad assembled from other people's usage: five
    players who each shot 20 times a night cannot all keep shooting 20 times.
    Each counting stat is scaled so the team's total moves most of the way to
    the league average, keeping better rosters ahead of worse ones instead of
    flattening every team onto the same line.
    """
    norms = _league_team_totals(league)
    for metric, average in norms.items():
        if metric not in TEAM_TOTAL_METRICS or average <= 0:
            continue
        raw = sum((r["projected"].get(metric) or 0.0) for r in players)
        if raw <= 0:
            continue
        target = average * (raw / average) ** TEAM_TOTAL_SHRINK
        factor = target / raw
        for r in players:
            v = r["projected"].get(metric)
            if v is not None:
                r["projected"][metric] = v * factor


def _scaled(value: float | None, factor: float) -> float | None:
    return None if value is None else round(float(value) * factor, 1)


def game_player_lines(league: League, home: str, away: str, season: int,
                      top: int = ROTATION_SIZE) -> dict:
    """Projected per-game lines for both rotations in one scheduled game."""
    rotations = _rotations(league, int(season))
    defence = _defence_factors(league, int(season))
    out: dict[str, list[dict]] = {}
    context: dict[str, dict] = {}
    for side, team, opponent in (("home", home, away), ("away", away, home)):
        opp_factor = defence.get(league.canonical_team(opponent), 1.0)
        venue = 1.0 + (HOME_SCORING_EDGE if side == "home" else -HOME_SCORING_EDGE)
        scoring = opp_factor * venue
        # Rebounds, assists and the rest move with the opponent but not with
        # the venue, and less than scoring does.
        other = 1.0 + (opp_factor - 1.0) / 2
        rows = []
        for player in rotations.get(league.canonical_team(team), [])[:top]:
            proj = player["projected"]
            rows.append({
                **{k: player[k] for k in
                   ("player_id", "player_name", "minutes", "games_played",
                    "based_on", "injury")},
                **{m: _scaled(proj.get(m), scoring if m == "pts" else other)
                   for m in GAME_LINE_METRICS},
            })
        out[side] = rows
        context[side] = {
            "team": team,
            "opponent_defence": round(opp_factor, 3),
            "venue_factor": round(venue, 3),
        }
    return {"players": out, "adjustments": context}


def game_actual_lines(league: League, date: str, home: str, away: str) -> dict:
    """What each rotation actually did, for a game that has been played."""
    log = data.gamelog(league)
    if log is None:
        return {}
    day = log[log["GAME_DATE"].dt.strftime("%Y-%m-%d") == str(date)].copy()
    if day.empty:
        return {}
    matchup = day["MATCHUP"].astype(str)
    day["team"] = matchup.str.split().str[0].map(league.canonical_team)
    out: dict[str, list[dict]] = {}
    for side, team in (("home", home), ("away", away)):
        rows = day[day.team == league.canonical_team(team)]
        out[side] = [{
            "player_id": int(r.player_id),
            "player_name": str(r.player_name),
            "minutes": round(float(r.MIN), 1) if pd.notna(r.MIN) else None,
            "pts": float(r.PTS), "reb": float(r.REB), "ast": float(r.AST),
            "stl": float(r.STL), "blk": float(r.BLK), "tov": float(r.TOV),
        } for r in rows.itertuples()]
        out[side].sort(key=lambda p: (p["minutes"] or 0), reverse=True)
    return out


@lru_cache(maxsize=4)
def game_line_backtest(league: League = DEFAULT, metric: str = "pts") -> dict:
    """How far a projected line lands from the actual one, per player-game.

    Scored on the most recent completed season, against the baseline of using
    the player's own average for that season — the number a projection has to
    beat to be worth anything.
    """
    log = data.gamelog(league)
    hist = _player_history(league)
    if log is None or hist.empty:
        return {}
    season = int(hist.season_int.max())
    actual = log[log["season"].astype(str) == str(season)]
    column = {"pts": "PTS", "reb": "REB", "ast": "AST"}.get(metric, "PTS")
    if actual.empty or column not in actual.columns:
        return {}

    projected = {}
    for row in hist[hist.season_int == season].itertuples():
        try:
            p = project_player(league, int(row.player_id), str(season))
        except (KeyError, ValueError):
            continue
        value = p["projected"].get(metric)
        if value is not None:
            projected[int(row.player_id)] = float(value)
    if not projected:
        return {}

    # The naive rival: assume a player repeats last season's per-game average.
    previous = hist[hist.season_int == season - 1]
    last_season = {int(r.player_id): float(v)
                   for r, v in zip(previous.itertuples(),
                                   pd.to_numeric(previous.get(metric), errors="coerce"))
                   if pd.notna(v)}

    scored = actual[actual["player_id"].astype(int).isin(projected)].copy()
    ids = scored["player_id"].astype(int)
    scored["projected"] = ids.map(projected)
    scored["naive"] = ids.map(last_season)
    # Hindsight: what the player actually averaged over the season being
    # scored. No forecast can see this, so it is a floor, not a rival.
    scored["season_mean"] = scored.groupby("player_id")[column].transform("mean")
    scored = scored.dropna(subset=[column, "projected"])
    if scored.empty:
        return {}
    naive = scored.dropna(subset=["naive"])
    return {
        "metric": metric,
        "season": str(season),
        "player_games": int(len(scored)),
        "mae": round(float((scored[column] - scored["projected"]).abs().mean()), 2),
        "naive_mae": (round(float((naive[column] - naive["naive"]).abs().mean()), 2)
                      if len(naive) else None),
        "hindsight_mae": round(float((scored[column] - scored["season_mean"]).abs().mean()), 2),
    }
