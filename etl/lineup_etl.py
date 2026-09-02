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

Outputs `data/lineup_<league>.parquet`: one row per season, team and five-man
group, with minutes, possessions, points for and against.

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
MIN_TEAM_GAMES = 10  # drops All-Star squads, as in sdv_etl

# ESPN's older substitution logs are too thin to close a lineup with — about 37
# a game in 2004 against 57 today — and a missing sub is a five that never
# changes. Each league's default starts where its feed gets dense enough, which
# `--validate` measures: rebuilt minutes land within a minute of the box score
# for 92% of NBA player-games in 2015 rising to 99.5% in 2025, but the WNBA
# feed is unusable before 2020 (58% in 2015, 88% in 2019, 99.4% from 2020).
FIRST_SEASON = {"nba": 2015, "wnba": 2020}

# A team uses about 375 different fives a season, most of them for one dead
# stretch each: net rating over two possessions is noise, and storing the tail
# would double the repo's committed data. Five minutes keeps 178 lineups a team
# and 83% of the minutes played. `team_min` records what was dropped.
MIN_LINEUP_SECONDS = 300.0


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


def _poss_counts(r) -> tuple[int, int, int, int]:
    """(fga, fta, oreb, tov) contributed by one play, for the team that owns it.

    Tuned against the team box scores: over the 2025 NBA season these rules
    reproduce field goal and free throw attempts exactly and rebounds to within
    one, and overcount turnovers by 1%. The rebound rule needs the athlete
    check — ESPN logs a ball out of bounds off a block as an offensive rebound
    by nobody, and the box score doesn't count those.
    """
    tt = r.type_text or ""
    if tt.startswith("Free Throw"):
        return 0, 1, 0, 0
    if "Turnover" in tt and tt != "No Turnover":
        return 0, 0, 0, 1
    if tt == "Offensive Rebound" and not pd.isna(r.athlete_id_1):
        return 0, 0, 1, 0
    if r.shooting_play:
        return 1, 0, 0, 0
    return 0, 0, 0, 0


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
            tally = {home: [0, 0, 0, 0], away: [0, 0, 0, 0]}  # fga, fta, oreb, tov
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
                tally = {home: [0, 0, 0, 0], away: [0, 0, 0, 0]}

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

    cols = ["game_id", "home_id", "away_id", "home_five", "away_five", "seconds",
            "home_pts", "away_pts",
            "home_fga", "home_fta", "home_oreb", "home_tov",
            "away_fga", "away_fta", "away_oreb", "away_tov"]
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


def season_lineups(league: League, season: int, check: bool) -> pd.DataFrame | None:
    pbp = fetch(league, "pbp", "play_by_play", season)
    box = fetch(league, "player_box", "player_box", season)
    if pbp is None or box is None:
        print(f"  {season}: not published")
        return None

    pbp = pbp[pbp["season_type"] == REGULAR_SEASON]
    box = box[(box["season_type"] == REGULAR_SEASON)
              & (~box["did_not_play"].fillna(False))
              & (box["athlete_id"].notna())].copy()
    if pbp.empty:
        print(f"  {season}: no regular-season play-by-play")
        return None

    keep = real_games(pbp)
    pbp = pbp[pbp["game_id"].isin(keep)]
    box = box[box["game_id"].isin(keep)]
    pbp = pbp.sort_values(["game_id", "period_number", "game_play_number"])

    st, played_seconds = stints(pbp, box)
    lineups = build_lineups(st, box, season)
    print(f"  {season}: {pbp['game_id'].nunique():>4} games, {len(st):>6} stints, "
          f"{len(lineups):>5} lineups")
    if check:
        validate(played_seconds, box)
    return lineups


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

    frames = [f for f in (season_lineups(league, yr, args.validate) for yr in seasons)
              if f is not None]
    if not frames:
        raise SystemExit(f"No {league.label} play-by-play downloaded.")

    out = pd.concat(frames, ignore_index=True)
    path = DATA / f"lineup{league.suffix}.parquet"

    # Seasons already on disk that this run didn't rebuild are kept: refreshing
    # the current season is a nightly job, re-downloading a decade of
    # play-by-play to do it is not.
    if path.exists():
        old = pd.read_parquet(path)
        # A file from an older version of this script describes the same
        # lineups differently; merging the two would leave half the rows with
        # holes in them. Rebuild instead.
        if list(old.columns) != list(out.columns):
            print(f"{path.name} has an older schema — rebuilding it from this run only")
            old = old.iloc[0:0]
        kept = old[~old["season"].astype(str).isin(set(out["season"]))]
        if not kept.empty:
            print(f"keeping {kept['season'].nunique()} season(s) already built")
            out = pd.concat([kept, out], ignore_index=True)

    out = out.sort_values(["season", "team_name", "min"],
                          ascending=[True, True, False]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path.name:<26} {len(out):>7} rows  "
          f"{out['season'].nunique()} seasons, {out['team_name'].nunique()} teams")


if __name__ == "__main__":
    main()
