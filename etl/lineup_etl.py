"""Five-man lineups, rebuilt from play-by-play.

ESPN publishes no lineup data, but it does publish substitutions, and a
substitution is a lineup change. Walking a game's subs in order reconstructs
who was on the floor for every second of it, which is the one thing the box
scores cannot say: `players_*` knows what a player did, never who he did it
alongside.

The reconstruction needs a starting point for each period, and only the first
one is given (`player_box.starter`). For the rest, a player was on the floor at
the tip of a period if his first appearance in it is anything other than being
subbed *in* — a shot, a foul, a rebound, or being subbed *out*. That resolves
essentially every team-period; the handful it doesn't are patched from whoever
finished the previous period. See `--validate`, which checks the rebuilt
minutes against the box score and prints the error.

Outputs two files per league. `data/lineup_<league>.parquet` has one row per
season, team and five-man group, with minutes, possessions and points for and
against. `data/rating_<league>.parquet` has one row per player-season: the
margin per 100 possessions the stints say he is responsible for once his
teammates and opponents are regressed out (RAPM), beside the raw on-court and
on-off numbers it adjusts.

Play-by-play is ~20 MB a season and none of it is committed — only the
aggregate, which is a few MB for two decades of both leagues.

    python etl/lineup_etl.py --league nba
    python etl/lineup_etl.py --league wnba --seasons 2020-2026
    python etl/lineup_etl.py --league nba --seasons 2025 --validate
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.leagues import LEAGUES, League  # noqa: E402

DATA = ROOT / "data"
TIMEOUT = 180

REPOS = {
    "nba": ("sportsdataverse/hoopR-nba-data", "nba"),
    "wnba": ("sportsdataverse/wehoop-wnba-data", "wnba"),
}

REGULAR_SEASON = 2
POSTSEASON = 3
SEASON_TYPES = {REGULAR_SEASON: "regular", POSTSEASON: "playoffs"}
MIN_TEAM_GAMES = 10  # drops All-Star squads, as in sdv_etl

# A postseason is 80-odd games against a regular season's 1,200, and every
# player has two coefficients to fit. Below this there is not enough play to
# regress anything out and the answer would be the prior, not the season.
MIN_PLAYOFF_STINTS = 800

# ESPN's older substitution logs are too thin to close a lineup with — about 37
# a game in 2004 against 57 today — and a missing sub is a five that never
# changes. Each league's default starts where its feed gets dense enough, which
# `--validate` measures: rebuilt minutes land within a minute of the box score
# for 92% of NBA player-games in 2015 rising to 99.5% in 2025, but the WNBA
# feed is unusable before 2020 (58% in 2015, 88% in 2019, 99.4% from 2020).
FIRST_SEASON = {"nba": 2015, "wnba": 2020}

# Every five a team used, including the ones that played one dead stretch. The
# tail is most of the rows and little of the time — a floor of five minutes
# would cut the file to a fifth — but it is what makes the with-or-without
# splits exact: each possession belongs to exactly one five, so any group of
# players can be answered by adding up the fives that contain them, and a
# missing tail would quietly bias every "without" bucket toward the starters.
MIN_LINEUP_SECONDS = 0.0


def url_for(league: League, dataset: str, filename: str, season: int) -> str:
    repo, seg = REPOS[league.key]
    return (f"https://raw.githubusercontent.com/{repo}/main/{seg}"
            f"/{dataset}/parquet/{filename}_{season}.parquet")


def fetch(league: League, dataset: str, filename: str, season: int) -> pd.DataFrame | None:
    """One season of one dataset; None if that season isn't published."""
    r = requests.get(url_for(league, dataset, filename, season), timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def real_games(pbp: pd.DataFrame) -> set:
    """game_ids where both sides are actual franchises, not All-Star squads."""
    sides = pd.concat([
        pbp[["game_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
        pbp[["game_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
    ]).drop_duplicates()
    played = sides.groupby("team_id")["game_id"].nunique()
    keep = set(played[played >= MIN_TEAM_GAMES].index)
    per_game = sides[sides["team_id"].isin(keep)].groupby("game_id")["team_id"].nunique()
    return set(per_game[per_game == 2].index)


def _period_starters(period: pd.DataFrame, team: float) -> set:
    """Who was on the floor when this period tipped, inferred from the plays.

    A player is a starter of the period if the first thing he does in it is not
    entering the game: making a play, or being the one taken off the floor.
    """
    on: set = set()
    seen: set = set()
    for r in period.itertuples():
        if r.team_id != team:
            continue
        if r.type_text == "Substitution":
            # Order matters: someone can be subbed out and back in later.
            if not pd.isna(r.athlete_id_2) and r.athlete_id_2 not in seen:
                seen.add(r.athlete_id_2)
                on.add(r.athlete_id_2)
            if not pd.isna(r.athlete_id_1):
                seen.add(r.athlete_id_1)
        elif not pd.isna(r.athlete_id_1) and r.athlete_id_1 not in seen:
            seen.add(r.athlete_id_1)
            on.add(r.athlete_id_1)
    return on


def _fix_to_five(on: set, period: pd.DataFrame, team: float,
                 previous: set, roster: set) -> set:
    """Force an inferred period-opening five to be exactly five.

    Too many means a sub the feed logged out of order, so the latest arrival is
    the one that doesn't belong. Too few means someone played the period
    without touching the ball or the bench — invisible to the plays — so fall
    back to whoever ended the previous period, then to anyone who dressed.
    """
    if len(on) > 5:
        first_seen: dict = {}
        for i, r in enumerate(period.itertuples()):
            if r.team_id == team and not pd.isna(r.athlete_id_1) and r.athlete_id_1 in on:
                first_seen.setdefault(r.athlete_id_1, i)
        on = set(sorted(on, key=lambda a: first_seen.get(a, 1 << 30))[:5])
    if len(on) < 5:
        pool = [a for a in previous if a not in on]
        pool += [a for a in roster if a not in on and a not in pool]
        on = on | set(pool[: 5 - len(on)])
    return on


# What one play contributes, in the order the stint tally keeps them.
TALLY = ["fga", "fta", "oreb", "tov", "fg_pts", "ft_pts"]


def _poss_counts(r) -> tuple[int, int, int, int, int, int]:
    """What one play contributes, for the team that owns it.

    Tuned against the team box scores: over the 2025 NBA season these rules
    reproduce field goal and free throw attempts exactly and rebounds to within
    one, and overcount turnovers by 1%. The rebound rule needs the athlete
    check — ESPN logs a ball out of bounds off a block as an offensive rebound
    by nobody, and the box score doesn't count those.

    Points are split by where they came from, since the free throw line and the
    field are separate skills and the impact model prices them separately.
    """
    tt = r.type_text or ""
    made = int(r.score_value) if (r.scoring_play and not pd.isna(r.score_value)) else 0
    if tt.startswith("Free Throw"):
        return 0, 1, 0, 0, 0, made
    if "Turnover" in tt and tt != "No Turnover":
        return 0, 0, 0, 1, 0, 0
    if tt == "Offensive Rebound" and not pd.isna(r.athlete_id_1):
        return 0, 0, 1, 0, 0, 0
    if r.shooting_play:
        return 1, 0, 0, 0, made, 0
    return 0, 0, 0, 0, 0, 0


def stints(pbp: pd.DataFrame, box: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cut every game into intervals of unchanged personnel.

    Returns one row per interval — both teams' fives, its length, and what each
    side did in it — plus per-player seconds for `--validate`.
    """
    starters = (box[box["starter"].fillna(False)]
                .groupby(["game_id", "team_id"])["athlete_id"]
                .apply(lambda s: set(s.astype(float))).to_dict())
    dressed = (box.groupby(["game_id", "team_id"])["athlete_id"]
                  .apply(lambda s: set(s.astype(float))).to_dict())

    rows: list[tuple] = []
    played_seconds: dict = defaultdict(float)

    for gid, game in pbp.groupby("game_id", sort=False):
        home = game["home_team_id"].iloc[0]
        away = game["away_team_id"].iloc[0]
        ended_previous: dict = {}

        for _, period in game.groupby("period_number", sort=False):
            floor = {}
            for team in (home, away):
                on = _period_starters(period, team)
                # Period one is the one the box score can state outright.
                if period["period_number"].iloc[0] == 1:
                    box_five = starters.get((gid, int(team)))
                    if box_five and len(box_five) == 5:
                        on = set(box_five)
                if len(on) != 5:
                    on = _fix_to_five(on, period, team, ended_previous.get(team, set()),
                                      dressed.get((gid, int(team)), set()))
                floor[team] = on

            clock = period["start_quarter_seconds_remaining"].iloc[0]
            score = {home: period["home_score"].iloc[0], away: period["away_score"].iloc[0]}
            tally = {home: [0] * len(TALLY), away: [0] * len(TALLY)}
            # When the floor has to be corrected, the player who hasn't touched
            # the game in longest is the likeliest mistake. Inferred names that
            # never appear in a play sort first, which is the point.
            last_seen: dict = {}

            def close(at: float, home_pts: float, away_pts: float) -> None:
                """Bank the interval that ends here, if it had any length."""
                nonlocal clock, score, tally
                length = clock - at
                if length > 0 and len(floor[home]) == 5 and len(floor[away]) == 5:
                    rows.append((
                        gid, home, away,
                        tuple(sorted(floor[home])), tuple(sorted(floor[away])),
                        length, home_pts - score[home], away_pts - score[away],
                        *tally[home], *tally[away],
                    ))
                    for team in (home, away):
                        for player in floor[team]:
                            played_seconds[(gid, player)] += length
                clock = at
                score = {home: home_pts, away: away_pts}
                tally = {home: [0] * len(TALLY), away: [0] * len(TALLY)}

            for i, r in enumerate(period.itertuples()):
                if not pd.isna(r.athlete_id_1):
                    last_seen[r.athlete_id_1] = i
                if r.type_text == "Substitution":
                    close(r.start_quarter_seconds_remaining, r.home_score, r.away_score)
                    if r.team_id in floor:
                        five = floor[r.team_id]
                        five.discard(r.athlete_id_2)
                        five.add(r.athlete_id_1)
                        # The outgoing player wasn't one we had on the floor, so
                        # someone else on it doesn't belong there.
                        while len(five) > 5:
                            five.discard(min(five, key=lambda a: last_seen.get(a, -1)))
                elif r.team_id in tally:
                    for j, n in enumerate(_poss_counts(r)):
                        tally[r.team_id][j] += n

            # The period's last row can carry a null or post-overtime clock, so
            # the tail runs to zero rather than to whatever it claims.
            close(0.0, period["home_score"].iloc[-1], period["away_score"].iloc[-1])
            ended_previous = {t: set(v) for t, v in floor.items()}

    cols = (["game_id", "home_id", "away_id", "home_five", "away_five", "seconds",
             "home_pts", "away_pts"]
            + [f"home_{c}" for c in TALLY] + [f"away_{c}" for c in TALLY])
    return pd.DataFrame(rows, columns=cols), played_seconds


def _one_sided(st: pd.DataFrame, side: str, other: str) -> pd.DataFrame:
    """Recast stints from one team's point of view, so both sides stack."""
    out = pd.DataFrame({
        "team_id": st[f"{side}_id"],
        "five": st[f"{side}_five"],
        "game_id": st["game_id"],
        "seconds": st["seconds"],
        "pts_for": st[f"{side}_pts"],
        "pts_against": st[f"{other}_pts"],
    })
    # Possessions the Oliver way, averaged with the opponent's count of the
    # same trips, which is how team ortg/drtg is built elsewhere in the app.
    own = (st[f"{side}_fga"] + 0.44 * st[f"{side}_fta"]
           - st[f"{side}_oreb"] + st[f"{side}_tov"])
    opp = (st[f"{other}_fga"] + 0.44 * st[f"{other}_fta"]
           - st[f"{other}_oreb"] + st[f"{other}_tov"])
    out["poss"] = 0.5 * (own + opp)
    return out


def build_lineups(st: pd.DataFrame, box: pd.DataFrame, season: int) -> pd.DataFrame:
    """Stints -> one row per five-man group."""
    both = pd.concat([_one_sided(st, "home", "away"), _one_sided(st, "away", "home")])
    agg = both.groupby(["team_id", "five"], as_index=False).agg(
        seconds=("seconds", "sum"), poss=("poss", "sum"),
        pts_for=("pts_for", "sum"), pts_against=("pts_against", "sum"),
        games=("game_id", "nunique"), stints=("game_id", "size"))
    # Every five the team used, including the ones about to be dropped: without
    # it a page of lineups can't say what share of the season it is looking at.
    team_min = (agg.groupby("team_id")["seconds"].sum() / 60).round(1)
    agg = agg[agg["seconds"] >= MIN_LINEUP_SECONDS]

    names = dict(zip(box["athlete_id"].astype(float), box["athlete_display_name"]))
    teams = (box.drop_duplicates("team_id")
                .set_index("team_id")[["team_display_name", "team_abbreviation"]])

    out = pd.DataFrame({
        "season": str(season),
        "team_id": agg["team_id"].astype("int64"),
        "team_name": agg["team_id"].map(teams["team_display_name"]),
        "team_abbr": agg["team_id"].map(teams["team_abbreviation"]),
        # Delimited rather than nested: a list column survives Parquet but not
        # every consumer of it, and these are only ever read back as a group.
        "player_ids": agg["five"].map(lambda f: "|".join(str(int(a)) for a in f)),
        "player_names": agg["five"].map(
            lambda f: "|".join(names.get(a, str(int(a))) for a in f)),
        "games": agg["games"].astype("int32"),
        "stints": agg["stints"].astype("int32"),
        "min": (agg["seconds"] / 60).round(1),
        "team_min": agg["team_id"].map(team_min),
        "poss": agg["poss"].round(1),
        "pts_for": agg["pts_for"].astype("int32"),
        "pts_against": agg["pts_against"].astype("int32"),
    })
    return out.sort_values("min", ascending=False).reset_index(drop=True)


# The pieces a possession's value breaks into. With `shots` = FGA + 0.44*FTA, a
# possession is a shot unless it was turned over or repeated after an offensive
# rebound, so points/poss = e * (1 + OREB/poss - TOV/poss) where e = points per
# shot. Splitting that against the league's own rates isolates one skill each:
#
#   shot value   (e - e_lg) * (1 + OREB/poss - TOV/poss)   how well he scores
#   rebounding    e_lg * (OREB/poss - oreb_lg)             how often he keeps it
#   turnovers    -e_lg * (TOV/poss  - tov_lg)              how rarely he loses it
#
# The three add back to points per 100 possessions short of a constant, which
# the intercept absorbs, so the fitted coefficients still sum to the whole.
# Pricing the last two at the league's efficiency rather than the stint's own is
# what keeps shooting out of them — otherwise a good shooter's efficiency leaks
# into his turnover and rebounding numbers.
VALUE_PARTS = ["field_goals", "free_throws", "second_chance", "turnovers"]

# How hard the fit is pulled toward zero: a prior on how far from average a
# player can be. It is fixed per league rather than re-chosen each season —
# cross-validating it per season optimises the prediction of individual stints,
# which is mostly noise, and leaves every season on a scale of its own that
# cannot be compared to the next. Being fixed, it also handles sample size on
# its own: a postseason carries a fraction of a season's possessions, so the
# same penalty pulls it much harder toward average, which is what a few hundred
# possessions deserve.
#
# The NBA needs a heavier hand than the WNBA. It plays five times the
# possessions, so the same penalty constrains it far less, and at the WNBA's
# setting role players on lucky stints float above the stars — Korver over
# Curry in 2015, Covington over everyone in 2018. Seasons stay comparable
# within a league, which is what matters; the two leagues never play each
# other, so a number from one was never comparable to the other anyway.
RIDGE_ALPHA = {"nba": 4000.0, "wnba": 1000.0}


def _league_rates(stints: pd.DataFrame) -> dict:
    """The season's own averages, which the pieces are measured against."""
    totals = {c: float(stints[f"home_{c}"].sum() + stints[f"away_{c}"].sum())
              for c in TALLY}
    shots = totals["fga"] + 0.44 * totals["fta"]
    poss = shots - totals["oreb"] + totals["tov"]
    return {
        "per_shot": (totals["fg_pts"] + totals["ft_pts"]) / shots if shots else 0.0,
        "fg_per_shot": totals["fg_pts"] / shots if shots else 0.0,
        "ft_per_shot": totals["ft_pts"] / shots if shots else 0.0,
        "oreb_rate": totals["oreb"] / poss if poss else 0.0,
        "tov_rate": totals["tov"] / poss if poss else 0.0,
    }


def _value_parts(pts_fg, pts_ft, fga, fta, oreb, tov, rates: dict):
    """Split points per 100 possessions into its four additive pieces."""
    # The side's own possessions, not the averaged count the lineup table uses:
    # the identity is between a team's shots and its own trips.
    shots = fga + 0.44 * fta
    poss = shots - oreb + tov
    live = shots > 0
    alive = poss > 0
    per = lambda x: np.divide(x, shots, out=np.zeros_like(shots), where=live)
    rate = lambda x: np.divide(x, poss, out=np.zeros_like(poss), where=alive)
    # Shots kept per possession: the multiplier a point of efficiency earns.
    kept = np.where(alive, 1.0 + rate(oreb) - rate(tov), 0.0)
    return {
        "field_goals": 100.0 * (per(pts_fg) - rates["fg_per_shot"]) * kept,
        "free_throws": 100.0 * (per(pts_ft) - rates["ft_per_shot"]) * kept,
        "second_chance": 100.0 * rates["per_shot"] * (rate(oreb) - rates["oreb_rate"]),
        "turnovers": -100.0 * rates["per_shot"] * (rate(tov) - rates["tov_rate"]),
    }


def _check_parts(parts: dict, pts_fg, pts_ft, fga, fta, oreb, tov) -> None:
    """The pieces rebuild points per 100 up to one constant, which the intercept
    absorbs. A drifting gap would mean the split has stopped being an identity,
    so the spread of the difference is what has to be zero — not the difference.
    """
    poss = fga + 0.44 * fta - oreb + tov
    live = poss > 0
    gap = (sum(parts.values())[live]
           - 100.0 * ((pts_fg + pts_ft)[live] / poss[live]))
    spread = float(np.max(gap) - np.min(gap)) if live.any() else 0.0
    if spread > 1e-6:
        raise AssertionError(
            f"value parts do not reconstruct points per 100: gap varies by {spread}")


def _design(stints: pd.DataFrame):
    """The regression's shape: one row per stint per direction of play.

    Returns the design matrix, the possession weights, the four value pieces as
    columns, and where each player's offensive and defensive coefficients live.
    Shared so a refit against a different centre builds exactly the same
    problem rather than a lookalike.
    """
    from scipy import sparse

    stints = stints.copy()
    # Same possession estimate the lineup table uses: Oliver's count from each
    # side, averaged, since both teams take the same trips down the floor.
    home_poss = (stints["home_fga"] + 0.44 * stints["home_fta"]
                 - stints["home_oreb"] + stints["home_tov"])
    away_poss = (stints["away_fga"] + 0.44 * stints["away_fta"]
                 - stints["away_oreb"] + stints["away_tov"])
    stints["poss"] = 0.5 * (home_poss + away_poss)
    stints = stints[stints["poss"] > 0].reset_index(drop=True)
    if stints.empty:
        return None

    players = sorted({p for five in stints["home_five"] for p in five}
                     | {p for five in stints["away_five"] for p in five})
    # Two columns per player, plus one for home court so the advantage is not
    # charged to whoever happened to be playing at home.
    offense = {p: i for i, p in enumerate(players)}
    defense = {p: i + len(players) for i, p in enumerate(players)}
    home_column = 2 * len(players)

    rows, cols, vals, weight = [], [], [], []
    order: list[tuple] = []          # (attacking side, row index)
    for i, r in enumerate(stints.itertuples()):
        for side, attacking, defending, at_home in (
            ("home", r.home_five, r.away_five, 1.0),
            ("away", r.away_five, r.home_five, 0.0),
        ):
            row = len(weight)
            for p in attacking:
                rows.append(row); cols.append(offense[p]); vals.append(1.0)
            for p in defending:
                rows.append(row); cols.append(defense[p]); vals.append(-1.0)
            if at_home:
                rows.append(row); cols.append(home_column); vals.append(1.0)
            # Weighted by the possessions behind them: a 6-point stint over 3
            # possessions is not a season of evidence.
            weight.append(stints["poss"].iloc[i])
            order.append((side, i))

    design = sparse.csr_matrix((vals, (rows, cols)),
                               shape=(len(weight), 2 * len(players) + 1))
    weight = np.asarray(weight, dtype=float)

    # Each observation's four pieces, taken from whichever side was attacking.
    sides = np.array([s for s, _ in order])
    idx = np.array([i for _, i in order])
    pick = lambda col: np.where(sides == "home",
                                stints[f"home_{col}"].to_numpy(dtype=float)[idx],
                                stints[f"away_{col}"].to_numpy(dtype=float)[idx])
    side_stats = {c: pick(c) for c in TALLY}
    rates = _league_rates(stints)
    parts = _value_parts(pts_fg=side_stats["fg_pts"], pts_ft=side_stats["ft_pts"],
                         fga=side_stats["fga"], fta=side_stats["fta"],
                         oreb=side_stats["oreb"], tov=side_stats["tov"], rates=rates)
    _check_parts(parts, side_stats["fg_pts"], side_stats["ft_pts"],
                 side_stats["fga"], side_stats["fta"],
                 side_stats["oreb"], side_stats["tov"])
    targets = np.column_stack([parts[k] for k in VALUE_PARTS])
    return design, weight, targets, players, offense, defense, home_column


def build_ratings(st: pd.DataFrame, box: pd.DataFrame, season: int,
                  league: League) -> pd.DataFrame:
    """Offensive and defensive impact per player, broken into what caused it.

    Raw plus-minus answers "what happened while he was out there", which is as
    much a statement about his teammates and opponents as about him. The fix is
    to regress it: every possession has five players trying to score and five
    trying to stop them, so each of the ten gets credit for the outcome with the
    other nine held constant.

    Each stint therefore becomes two observations, one per direction of play.
    The scoring team's five enter with a +1 in their offensive column and the
    defending five with a -1 in their defensive column. The signs mean both
    halves read the same way: positive is good, whether it came from scoring
    more or allowing less, and a player's total is the two added together.

    The target is not one number but four — the pieces of VALUE_PARTS, which sum
    to points per 100 possessions. One penalty, chosen by cross-validating the
    total, is used for all of them, which is what makes the parts add up to the
    whole exactly rather than approximately.

    That leaves ten coefficients per player over a few hundred players, so the
    fit is ridged toward zero: a player with few possessions lands near average
    rather than at an extreme.

    Raw on-court and on-off numbers come back alongside, unadjusted, because the
    gap between them and the fitted number is the useful part.
    """
    from scipy import sparse                     # ETL-only dependency
    from sklearn.linear_model import Ridge

    stints = st.copy()
    # Same possession estimate the lineup table uses: Oliver's count from each
    # side, averaged, since both teams take the same trips down the floor.
    home_poss = (stints["home_fga"] + 0.44 * stints["home_fta"]
                 - stints["home_oreb"] + stints["home_tov"])
    away_poss = (stints["away_fga"] + 0.44 * stints["away_fta"]
                 - stints["away_oreb"] + stints["away_tov"])
    stints["poss"] = 0.5 * (home_poss + away_poss)
    built = _design(stints)
    if built is None:
        return pd.DataFrame()
    design, weight, targets, players, offense, defense, home_column = built
    stints = stints.copy()
    home_poss = (stints["home_fga"] + 0.44 * stints["home_fta"]
                 - stints["home_oreb"] + stints["home_tov"])
    away_poss = (stints["away_fga"] + 0.44 * stints["away_fta"]
                 - stints["away_oreb"] + stints["away_tov"])
    stints["poss"] = 0.5 * (home_poss + away_poss)
    stints = stints[stints["poss"] > 0].reset_index(drop=True)

    # One penalty for every piece: a different alpha per target would fit four
    # unrelated models whose coefficients no longer add to the whole.
    model = Ridge(alpha=RIDGE_ALPHA[league.key])
    model.fit(design, targets, sample_weight=weight)
    coef = {part: model.coef_[i] for i, part in enumerate(VALUE_PARTS)}

    # Raw on-court and off-court totals, per player and per team.
    on: dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])   # poss, pf, pa, seconds
    team_totals: dict = defaultdict(lambda: [0.0, 0.0, 0.0])
    player_teams: dict = defaultdict(set)
    for r in stints.itertuples():
        for five, team, pf, pa in ((r.home_five, r.home_id, r.home_pts, r.away_pts),
                                   (r.away_five, r.away_id, r.away_pts, r.home_pts)):
            totals = team_totals[team]
            totals[0] += r.poss; totals[1] += pf; totals[2] += pa
            for p in five:
                slot = on[(p, team)]
                slot[0] += r.poss; slot[1] += pf; slot[2] += pa; slot[3] += r.seconds
                player_teams[p].add(team)

    names = dict(zip(box["athlete_id"].astype(float), box["athlete_display_name"]))
    teams = (box.drop_duplicates("team_id")
                .set_index("team_id")[["team_display_name", "team_abbreviation"]])
    games = (box.groupby("athlete_id")["game_id"].nunique().to_dict())

    out = []
    for p in players:
        p_poss = p_pf = p_pa = p_secs = 0.0
        off_poss = off_pf = off_pa = 0.0
        for team in player_teams[p]:
            slot = on[(p, team)]
            p_poss += slot[0]; p_pf += slot[1]; p_pa += slot[2]; p_secs += slot[3]
            total = team_totals[team]
            off_poss += total[0] - slot[0]
            off_pf += total[1] - slot[1]
            off_pa += total[2] - slot[2]
        # Team of record: where he played the most, matching the players table.
        main_team = max(player_teams[p], key=lambda t: on[(p, t)][0])
        on_net = 100.0 * (p_pf - p_pa) / p_poss if p_poss else None
        off_net = 100.0 * (off_pf - off_pa) / off_poss if off_poss else None
        out.append({
            "season": str(season),
            "player_id": int(p),
            "player_name": names.get(p, str(int(p))),
            "team_id": int(main_team),
            "team_name": teams["team_display_name"].get(main_team),
            "team_abbr": teams["team_abbreviation"].get(main_team),
            "games": int(games.get(int(p), 0)),
            "min": round(p_secs / 60, 1),
            "poss": round(p_poss, 1),
            **{f"off_{part}": round(float(coef[part][offense[p]]), 3)
               for part in VALUE_PARTS},
            **{f"def_{part}": round(float(coef[part][defense[p]]), 3)
               for part in VALUE_PARTS},
            "off_rating": round(float(sum(coef[k][offense[p]] for k in VALUE_PARTS)), 2),
            "def_rating": round(float(sum(coef[k][defense[p]] for k in VALUE_PARTS)), 2),
            "rapm": round(float(sum(coef[k][offense[p]] + coef[k][defense[p]]
                                    for k in VALUE_PARTS)), 2),
            "on_net": None if on_net is None else round(on_net, 1),
            "off_net": None if off_net is None else round(off_net, 1),
            "on_off": None if on_net is None or off_net is None else round(on_net - off_net, 1),
            "plus_minus": int(p_pf - p_pa),
        })
    frame = pd.DataFrame(out)
    print(f"    ratings: {len(frame)} players, "
          f"home court {sum(coef[k][home_column] for k in VALUE_PARTS):+.2f}")
    return frame.sort_values("rapm", ascending=False).reset_index(drop=True)


def validate(played_seconds: dict, box: pd.DataFrame) -> None:
    """Compare rebuilt minutes against the box score, and say how far off."""
    rebuilt = pd.Series(played_seconds, dtype=float) / 60
    if rebuilt.empty:
        print("  nothing to validate")
        return
    rebuilt.index = pd.MultiIndex.from_tuples(rebuilt.index, names=["game_id", "athlete_id"])
    rebuilt = rebuilt.rename("rebuilt").reset_index()
    actual = box[["game_id", "athlete_id", "minutes"]].copy()
    actual["athlete_id"] = actual["athlete_id"].astype(float)
    merged = actual.merge(rebuilt, on=["game_id", "athlete_id"], how="outer")
    err = (merged["rebuilt"].fillna(0) - merged["minutes"].fillna(0)).abs()
    print(f"  minutes vs box score: {len(err)} player-games, "
          f"mean off by {err.mean():.2f} min, median {err.median():.2f}, "
          f"{(err <= 1).mean() * 100:.1f}% within a minute")


def parse_seasons(spec: str | None, league: League) -> list[int]:
    latest = pd.Timestamp.today().year
    if not spec:
        return list(range(FIRST_SEASON[league.key], latest + 1))
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def _of_season_type(pbp: pd.DataFrame, box: pd.DataFrame, code: int
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One season type's plays and box rows, ready to be cut into stints."""
    p = pbp[pbp["season_type"] == code]
    b = box[(box["season_type"] == code)
            & (~box["did_not_play"].fillna(False))
            & (box["athlete_id"].notna())].copy()
    if p.empty:
        return p, b
    # The All-Star filter counts games per team, which a postseason fails by
    # design — a team swept in the first round plays four. Every playoff game
    # is between real franchises anyway, so it only applies to the season.
    if code == REGULAR_SEASON:
        keep = real_games(p)
        p = p[p["game_id"].isin(keep)]
        b = b[b["game_id"].isin(keep)]
    return p.sort_values(["game_id", "period_number", "game_play_number"]), b


def season_lineups(league: League, season: int, check: bool
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Rebuild one season, returning its lineups and its player ratings.

    Lineups are the regular season only — a five that played three playoff
    games is a footnote, not a rotation. Ratings are built for both, so a
    postseason can be read on its own terms.
    """
    pbp = fetch(league, "pbp", "play_by_play", season)
    box = fetch(league, "player_box", "player_box", season)
    if pbp is None or box is None:
        print(f"  {season}: not published")
        return None

    regular_pbp, regular_box = _of_season_type(pbp, box, REGULAR_SEASON)
    if regular_pbp.empty:
        print(f"  {season}: no regular-season play-by-play")
        return None

    st, played_seconds = stints(regular_pbp, regular_box)
    lineups = build_lineups(st, regular_box, season)
    print(f"  {season}: {regular_pbp['game_id'].nunique():>4} games, {len(st):>6} stints, "
          f"{len(lineups):>5} lineups")
    rated = [build_ratings(st, regular_box, season, league).assign(season_type="regular")]
    if check:
        validate(played_seconds, regular_box)

    playoff_pbp, playoff_box = _of_season_type(pbp, box, POSTSEASON)
    if not playoff_pbp.empty:
        post, _ = stints(playoff_pbp, playoff_box)
        if len(post) >= MIN_PLAYOFF_STINTS:
            print(f"        playoffs: {playoff_pbp['game_id'].nunique():>3} games, "
                  f"{len(post):>5} stints")
            rated.append(build_ratings(post, playoff_box, season, league)
                         .assign(season_type="playoffs"))
        else:
            print(f"        playoffs: only {len(post)} stints, too few to fit")

    return (lineups, pd.concat([r for r in rated if not r.empty], ignore_index=True),
            st, regular_box)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", default="nba", choices=sorted(REPOS))
    ap.add_argument("--seasons", default=None,
                    help="season or inclusive range (default: the seasons this "
                         f"league's substitution feed can be trusted for, "
                         f"{', '.join(f'{k} from {v}' for k, v in FIRST_SEASON.items())})")
    ap.add_argument("--validate", action="store_true",
                    help="check rebuilt minutes against the box score")
    args = ap.parse_args(argv)

    league = LEAGUES[args.league]
    seasons = parse_seasons(args.seasons, league)
    print(f"[{league.label}] lineups from play-by-play, {seasons[0]}-{seasons[-1]}")

    kept: dict = {}
    built = []
    for yr in seasons:
        result = season_lineups(league, yr, args.validate)
        if result is None:
            continue
        lineups, ratings, stints, box = result
        built.append((lineups, ratings))
        kept[yr] = (stints, box)
    if not built:
        raise SystemExit(f"No {league.label} play-by-play downloaded.")

    ratings = pd.concat([b[1] for b in built], ignore_index=True)
    ratings = add_window_ratings(ratings, kept, league)
    ratings = add_box_prior_ratings(ratings, kept, league)

    write_merged(pd.concat([b[0] for b in built], ignore_index=True),
                 f"lineup{league.suffix}", ["season", "team_name", "min"],
                 [True, True, False])
    write_merged(ratings, f"rating{league.suffix}",
                 ["season", "season_type", "rapm"], [True, True, False])


# How many seasons a rolling fit pools. One season of stints is a weak estimate
# — it repeats season to season at about r=0.4 — and three is the window most
# public work settles on: enough to steady the number, not so much that a
# player's development is averaged away.
RAPM_WINDOW = 3

# Box-score rates the prior is built from, per 100 possessions of playing time.
PRIOR_FEATURES = ["pts", "ast", "reb", "stl", "blk", "tov", "ts_pct", "usg_pct"]


def add_window_ratings(ratings: pd.DataFrame, kept: dict, league: League) -> pd.DataFrame:
    """A rolling multi-season RAPM, beside the single-season one.

    Fitting the same model over a longer window trades currency for stability:
    the one-season number moves as much with who a player happened to share the
    floor with as with the player.
    """
    seasons = sorted(kept)
    out = []
    for i, yr in enumerate(seasons):
        window = seasons[max(0, i - RAPM_WINDOW + 1):i + 1]
        stints = pd.concat([kept[w][0] for w in window], ignore_index=True)
        box = pd.concat([kept[w][1] for w in window], ignore_index=True)
        fit = build_ratings(stints, box, yr, league)
        if fit.empty:
            continue
        print(f"    {yr} {RAPM_WINDOW}-season window ({window[0]}-{window[-1]}): "
              f"{len(fit)} players over {len(stints)} stints")
        out.append(fit[["player_id", "rapm"]].assign(season=str(yr)))
    if not out:
        return ratings
    window_col = pd.concat(out, ignore_index=True).rename(columns={"rapm": "rapm_window"})
    # Only the regular season has a window; a postseason keeps its own number.
    return ratings.merge(window_col, on=["season", "player_id"], how="left")


def add_box_prior_ratings(ratings: pd.DataFrame, kept: dict, league: League
                          ) -> pd.DataFrame:
    """RAPM refit against a box-score prior instead of against zero.

    Plain ridge pulls a thinly-played player toward average, which is the right
    instinct but the wrong target: the box score already says something about
    him. Fitting a model from box-score rates to the plain RAPM gives a per
    player expectation, and refitting with that as the centre keeps the stints
    as evidence while starting each player somewhere defensible.

    Ridge around a prior needs no new solver: with beta = mu + b, minimising
    ||y - X(mu + b)||^2 + lambda||b||^2 is the ordinary problem on the residual
    y - X*mu, so the prior is subtracted out, fit as usual, and added back.
    """
    from scipy import sparse
    from sklearn.linear_model import Ridge

    box = _load_optional_players(league)
    if box is None:
        print("  no player table on disk — skipping the box-score prior")
        return ratings

    plain = ratings[ratings["season_type"] == "regular"]
    frame = plain.merge(box, on=["season", "player_id"], how="inner")
    frame = frame[frame["poss"] >= 1000].dropna(subset=PRIOR_FEATURES + ["rapm"])
    if len(frame) < 200:
        print("  too few player-seasons to fit a prior — skipping")
        return ratings

    # What the box score says impact should be, learned from the plain fit.
    model = Ridge(alpha=1.0)
    model.fit(frame[PRIOR_FEATURES].to_numpy(dtype=float),
              frame["rapm"].to_numpy(dtype=float))
    fitted = model.predict(frame[PRIOR_FEATURES].to_numpy(dtype=float))
    print(f"  box-score prior fit on {len(frame)} player-seasons, "
          f"R2 {model.score(frame[PRIOR_FEATURES].to_numpy(dtype=float), frame['rapm']):.3f}")

    priors = box.dropna(subset=PRIOR_FEATURES).copy()
    priors["prior"] = model.predict(priors[PRIOR_FEATURES].to_numpy(dtype=float))
    lookup = priors.set_index(["season", "player_id"])["prior"].to_dict()

    out = []
    for yr, (stints, season_box) in sorted(kept.items()):
        fit = _fit_with_prior(stints, season_box, yr, league, lookup)
        if fit is not None:
            out.append(fit)
    if not out:
        return ratings
    prior_col = pd.concat(out, ignore_index=True)
    return ratings.merge(prior_col, on=["season", "player_id"], how="left")


def _load_optional_players(league: League) -> pd.DataFrame | None:
    """The player-season box table, if the other ETL has run."""
    path = DATA / f"players{league.suffix}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["season"] = df["season"].astype(str)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    keep = ["season", "player_id"] + [c for c in PRIOR_FEATURES if c in df.columns]
    return df[keep] if len(keep) > 2 else None


def _fit_with_prior(stints: pd.DataFrame, box: pd.DataFrame, season: int,
                    league: League, lookup: dict) -> pd.DataFrame | None:
    """One season's RAPM, centred on the box-score prior rather than zero."""
    from scipy import sparse
    from sklearn.linear_model import Ridge

    built = _design(stints)
    if built is None:
        return None
    design, weight, targets, players, offense, defense, _ = built
    # The prior speaks to total impact, so the refit works on the total.
    target = targets.sum(axis=1)
    # A player with no box row starts at average, which is what zero means here.
    mu = np.zeros(design.shape[1])
    for p in players:
        prior = lookup.get((str(season), int(p)))
        if prior is not None and np.isfinite(prior):
            # Split evenly across the two ends: the box model predicts a total
            # and has nothing to say about which half it came from.
            mu[offense[p]] = prior / 2
            mu[defense[p]] = prior / 2
    model = Ridge(alpha=RIDGE_ALPHA[league.key])
    model.fit(design, target - design @ mu, sample_weight=weight)
    coef = model.coef_ + mu
    return pd.DataFrame({
        "season": str(season),
        "player_id": [int(p) for p in players],
        "rapm_prior": [round(float(coef[offense[p]] + coef[defense[p]]), 2)
                       for p in players],
    })


def write_merged(out: pd.DataFrame, stem: str, sort_by: list[str],
                 ascending: list[bool]) -> None:
    """Write one output, keeping seasons this run didn't rebuild.

    Refreshing the current season is a nightly job; re-downloading a decade of
    play-by-play to do it is not.
    """
    path = DATA / f"{stem}.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        # A file from an older version of this script describes the same rows
        # differently; merging the two would leave half of them with holes.
        if list(old.columns) != list(out.columns):
            print(f"{path.name} has an older schema — rebuilding it from this run only")
            old = old.iloc[0:0]
        kept = old[~old["season"].astype(str).isin(set(out["season"]))]
        if not kept.empty:
            print(f"keeping {kept['season'].nunique()} season(s) already in {path.name}")
            out = pd.concat([kept, out], ignore_index=True)

    out = out.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path.name:<26} {len(out):>7} rows  {out['season'].nunique()} seasons")


if __name__ == "__main__":
    main()
