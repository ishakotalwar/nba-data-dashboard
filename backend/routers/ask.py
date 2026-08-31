"""Natural-language front door to the analytics already in this app.

The rule this module exists to enforce: a model may decide *what to compute*,
never *what the answer is*. A question becomes a validated `AskQuery`, the
query is executed by the same router functions the UI calls, and the numbers
come back from Parquet. Nothing here recomputes analytics of its own.
"""
from __future__ import annotations

import difflib
import re

import pandas as pd
from fastapi import APIRouter

from .. import analytics, data, leagues
from ..ask_parse import llm_available, parse_llm, parse_rules
from ..ask_schema import (
    METRIC_ALIASES,
    METRIC_LABELS,
    TEAM_METRIC_ALIASES,
    AskQuery,
    AskRequest,
    PlayerRef,
)
from . import compare as compare_router
from . import explorer as explorer_router
from . import shots as shots_router
from . import similarity as similarity_router
from . import teams as teams_router

router = APIRouter(prefix="/api", tags=["ask"])

# Nicknames the data will never contain but people always type.
NICKNAMES = {
    "sga": "Shai Gilgeous-Alexander",
    "the beard": "James Harden",
    "cp3": "Chris Paul",
    "kd": "Kevin Durant",
    "steph": "Stephen Curry",
    "giannis": "Giannis Antetokounmpo",
    "luka": "Luka Doncic",
    "ad": "Anthony Davis",
    "kat": "Karl-Anthony Towns",
    "dame": "Damian Lillard",
    "pg13": "Paul George",
    "jokic": "Nikola Jokic",
    "joker": "Nikola Jokic",
    "bron": "LeBron James",
    "lebron": "LeBron James",
    "mj": "Michael Jordan",
}

# Topics people reasonably ask about that this dataset simply does not contain.
# Answering these from general knowledge is exactly what the feature must not do.
OUT_OF_SCOPE: list[tuple[str, str]] = [
    (r"\bsalar(?:y|ies)\b|\bcontract\b|\bpaid\b|\bearn(?:ed|ings)?\b|\bcap hit\b",
     "I don't have salary or contract data in Full Court — only on-court statistics."),
    (r"\binjur(?:y|ies|ed)\b|\bhealth\b|\bmissed games? due\b",
     "I don't have injury data in Full Court."),
    (r"\bmvp\b|\ball[- ]star\b|\baward\b|\ball[- ]nba\b|\brookie of the year\b",
     "I don't have awards or voting data — only the per-season statistics themselves."),
    (r"\bplayoffs?\b|\bpostseason\b|\bfinals\b|\bchampionships?\b|\btitles?\b|\brings?\b",
     "Full Court covers regular-season data only, so I can't answer playoff questions."),
    (r"\bdraft(?:ed)?\b|\bcollege\b|\bncaa\b",
     "I don't have draft or college data in Full Court."),
    (r"\bcoach(?:es|ing)?\b|\battendance\b|\btrade[ds]?\b",
     "That isn't in the dataset — Full Court holds player- and team-season statistics."),
]

# Shots are only pulled for recent seasons; say so rather than return nothing.
SHOT_SEASON_HINT = "2022"


def _fail(message: str, **extra) -> dict:
    return {"status": "unsupported", "summary": message, "results": [], **extra}


def _clarify(message: str, options: list[dict]) -> dict:
    return {"status": "needs_clarification", "summary": message,
            "options": options, "results": []}


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def _expand_nickname(name: str) -> str:
    """"steph curry" and "KD" both name a real player; "MJ Walker" does not.

    A whole-string nickname always expands. A single token inside a longer name
    expands only when the nickname's own surname is also present, so a shared
    first name or initials can't hijack a different player.
    """
    low = name.strip().lower()
    if low in NICKNAMES:
        return NICKNAMES[low]
    tokens = low.split()
    for token in tokens:
        full = NICKNAMES.get(token)
        if not full:
            continue
        surname = full.split()[-1].lower()
        if len(tokens) == 1 or surname in tokens:
            return full
    return name.strip()


def _fuzzy_names(df: pd.DataFrame, query: str) -> list[str]:
    """Closest real names to a misspelling.

    Compared against the whole name and against each part of it, because a
    misspelling can target either half: "Jokich" is close to the surname
    "Jokic", "Yannis" to the first name "Giannis", and neither is close to the
    full "Nikola Jokic" / "Giannis Antetokounmpo".
    """
    names = df["player_name"].dropna().unique().tolist()
    low = query.lower()
    scored: list[tuple[float, str]] = []
    for n in names:
        parts = [n.lower()] + n.lower().split()
        best = max(difflib.SequenceMatcher(None, low, part).ratio() for part in parts)
        if best >= 0.72:
            scored.append((best, n))
    scored.sort(reverse=True)
    return [n for _, n in scored[:6]]


def _candidates(lg, name: str) -> pd.DataFrame:
    df = data.players(lg)
    query = _expand_nickname(name)

    exact = df[df["player_name"].str.lower() == query.lower()]
    if not exact.empty:
        return exact

    contains = df[df["player_name"].str.contains(re.escape(query), case=False, na=False)]
    if not contains.empty:
        return contains

    # Nothing literal matched — fall back to spelling distance.
    close = _fuzzy_names(df, query)
    return df[df["player_name"].isin(close)] if close else df.iloc[0:0]


def _resolve_player(lg, ref: PlayerRef) -> tuple[dict | None, dict | None]:
    """(resolved, error). Resolution is deterministic: exact name, else the
    substring match that is clearly the most prominent player in that season."""
    sub = _candidates(lg, ref.player)
    if sub.empty:
        return None, _fail(
            f"I don't have a {lg.label} player matching “{ref.player}” in the data."
        )

    if ref.season:
        in_season = sub[sub["season"] == str(ref.season)]
        if in_season.empty:
            names = sorted(sub["player_name"].unique())[:6]
            who = names[0] if len(names) == 1 else f"anyone matching “{ref.player}”"
            return None, _fail(
                f"I don't have a {ref.season} {lg.label} season for {who}."
            )
        sub = in_season

    names = sub["player_name"].unique()
    if len(names) > 1:
        # Rank by production so "Curry" in 2016 resolves to Stephen, not Seth,
        # but only when the gap is decisive.
        scored = []
        for n in names:
            rows = sub[sub["player_name"] == n]
            pts = pd.to_numeric(rows.get("pts"), errors="coerce").fillna(0).max()
            gp = pd.to_numeric(rows.get("gp"), errors="coerce").fillna(0).max()
            scored.append((float(pts) * float(gp), n))
        scored.sort(reverse=True)
        top, second = scored[0], scored[1]
        if second[0] <= 0 or top[0] < 2 * second[0]:
            return None, _clarify(
                f"“{ref.player}” matches more than one {lg.label} player.",
                [{"player_name": n, "seasons": sorted(
                    sub[sub["player_name"] == n]["season"].astype(str).unique())[-6:]}
                 for _, n in scored[:6]],
            )
        sub = sub[sub["player_name"] == top[1]]

    row = sub.sort_values("season").iloc[-1]
    return {
        "player_id": int(row["player_id"]),
        "player_name": str(row["player_name"]),
        "season": str(ref.season or row["season"]),
    }, None


def _infer_league(question: str, requested: str | None, ui_league: str | None):
    """Explicit league wins; otherwise the UI's league. Never search both."""
    return leagues.get(requested or ui_league)


def _validate(query: AskQuery, lg) -> dict | None:
    """Reject anything the dataset cannot answer, with a reason."""
    available = set(data.available_metrics(lg))
    for f in query.filters:
        if f.metric not in METRIC_ALIASES.values():
            return _fail(f"“{f.metric}” isn't a metric I know about.")
        if f.metric not in available:
            return _fail(
                f"{f.metric} isn't available for individual {lg.label} players "
                "in this dataset."
            )
    if query.intent == "team_explorer" and query.metric:
        if query.metric not in TEAM_METRIC_ALIASES.values():
            return _fail(f"“{query.metric}” isn't a team metric I can rank on.")
    seasons = set(data.seasons(lg))
    for bound in (query.season_from, query.season_to):
        if bound and seasons and bound > max(seasons):
            return _fail(
                f"I only have {lg.label} seasons through {max(seasons)}."
            )
    return None


# --------------------------------------------------------------------------
# Execution — every branch calls the existing router function.
# --------------------------------------------------------------------------

def _run_explorer(query: AskQuery, lg) -> dict:
    sort_key = query.sort or (query.filters[0].metric if query.filters else "pts")
    req = explorer_router.ExplorerRequest(
        league=lg.key,
        season_from=query.season_from,
        season_to=query.season_to,
        min_gp=query.min_gp,
        filters=[explorer_router.Filter(metric=f.metric, op=f.op, value=f.value,
                                        value2=f.value2) for f in query.filters],
        sort=sort_key,
        dir=query.dir,
        page=1,
        page_size=min(query.limit, 50),
    )
    out = explorer_router.explorer(req)
    total = out.get("total", 0)
    shown = len(out.get("rows", []))

    if query.filters:
        summary = (f"{total:,} {lg.label} player-season{'s' if total != 1 else ''} match"
                   + (f" — showing the top {shown}." if total > shown else "."))
        columns = ["player_name", "season", "team_abbr"] + [f.metric for f in query.filters]
    else:
        # A ranking question: say what it ranked on, and on what pool.
        direction = "Lowest" if query.dir == "asc" else "Top"
        summary = (f"{direction} {shown} {lg.label} player-seasons by "
                   f"{METRIC_LABELS.get(sort_key, sort_key)}"
                   + (f", minimum {query.min_gp} games." if query.min_gp else "."))
        if query.note:
            summary += f" {query.note}"
        columns = ["player_name", "season", "team_abbr", sort_key]

    return {
        "status": "ok",
        "summary": summary,
        "columns": columns,
        "results": out.get("rows", []),
        "total": total,
        "target_page": "explorer",
        "navigate": {"page": "explorer", "state": req.model_dump()},
    }


def _run_similarity(query: AskQuery, lg, anchor: dict) -> dict:
    req = similarity_router.SimilarityRequest(
        player_id=anchor["player_id"], season=anchor["season"], league=lg.key,
        preset=query.preset or "Overall", k=min(query.limit, 25),
    )
    out = similarity_router.similarity(req)
    matches = out.get("matches", [])
    preset = (query.preset or "Overall").lower()
    lens = "" if preset == "overall" else f" through a {preset} lens"
    return {
        "status": "ok",
        "summary": (f"Seasons most like {lg.display_season(anchor['season'])} "
                    f"{anchor['player_name']}{lens}, from "
                    f"{out.get('pool_size', 0):,} qualifying player-seasons."),
        "results": matches,
        "anchor": out.get("anchor"),
        "target_page": "similarity",
        "navigate": {"page": "similarity",
                     "state": {**req.model_dump(),
                               "player_name": anchor["player_name"]}},
    }


def _run_compare(query: AskQuery, lg, resolved: list[dict]) -> dict:
    req = compare_router.CompareRequest(
        league=lg.key, mode="season",
        selections=[compare_router.Selection(player_id=r["player_id"], season=r["season"])
                    for r in resolved],
        metrics=[m for m in ("pts", "reb", "ast", "ts_pct") if m in data.available_metrics(lg)],
    )
    out = compare_router.compare(req)
    names = " vs ".join(r["key"] for r in out.get("rows", []))
    return {
        "status": "ok",
        "summary": names or "Nothing to compare.",
        "results": out.get("rows", []),
        "metrics": out.get("metrics", []),
        "target_page": "compare",
        "navigate": {"page": "compare",
                     "state": {**req.model_dump(),
                               "players": resolved}},
    }


def _run_shots(query: AskQuery, lg, who: dict) -> dict:
    try:
        out = shots_router.shot_zones(who["player_id"], who["season"], lg.key)
    except Exception:
        return _fail(
            f"I don't have shot locations for {lg.display_season(who['season'])} "
            f"{who['player_name']} — shot data starts at "
            f"{lg.display_season(SHOT_SEASON_HINT)}."
        )
    zones = out.get("zones", [])
    best = max(
        (z for z in zones if (z.get("fga") or 0) >= 20 and z.get("diff") is not None),
        key=lambda z: z["diff"], default=None,
    )
    where = (f" Most efficient zone relative to the league: {best['zone']} "
             f"({best['diff']:+.1%} vs league).") if best else ""
    return {
        "status": "ok",
        "summary": (f"{lg.display_season(out['season'])} {out['player_name']} — "
                    f"{out['total_fga']:,} attempts, {out['fg_pct']:.1%} overall.{where}"),
        "results": zones,
        "target_page": "shots",
        "navigate": {"page": "shots",
                     "state": {"player_id": who["player_id"], "season": who["season"],
                               "player_name": who["player_name"], "league": lg.key}},
    }


# Every team-season fits comfortably in one call (718 NBA, 302 WNBA), and the
# ranking endpoint has no direction or season parameter — so ask for all of it,
# then narrow. Slicing before filtering would silently drop qualifying teams.
_ALL_TEAM_SEASONS = 10_000


def _run_team_explorer(query: AskQuery, lg) -> dict:
    metric = query.metric or "net"
    # `teams_rankings` already orders best-first for this metric (it knows that
    # a low drtg is good), so "worst" means the far end of that list rather
    # than a reversed sort.
    want_worst = query.dir == "asc"

    out = teams_router.teams_rankings(metric=metric, league=lg.key,
                                      limit=_ALL_TEAM_SEASONS)
    rows = out.get("rows", [])
    if query.season_from or query.season_to:
        lo, hi = query.season_from or "0000", query.season_to or "9999"
        rows = [r for r in rows if lo <= str(r.get("season", "")) <= hi]

    n = min(query.limit, 25)
    rows = list(reversed(rows[-n:])) if want_worst else rows[:n]

    label = "Worst" if want_worst else "Best"
    span = ""
    if query.season_from or query.season_to:
        lo = lg.display_season(query.season_from) if query.season_from else "the start"
        hi = lg.display_season(query.season_to) if query.season_to else "now"
        span = f", {lo} to {hi}"
    return {
        "status": "ok",
        "summary": f"{label} {len(rows)} {lg.label} team-seasons by {metric}{span}.",
        "results": rows,
        "metric": metric,
        "target_page": "teams",
        "navigate": {"page": "teams", "state": {"metric": metric, "league": lg.key}},
    }


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------

@router.get("/ask/capabilities")
def capabilities(league: str | None = None):
    """What the natural-language layer can currently answer."""
    lg = leagues.get(league)
    return {
        "league": lg.key,
        "metrics": data.available_metrics(lg),
        "team_metrics": sorted(set(TEAM_METRIC_ALIASES.values())),
        "presets": list(similarity_router.PRESETS),
        "llm_parser": llm_available(),
        "examples": [
            "Which NBA players since 2010 averaged at least 25 points per game?",
            "Which players since 2003 averaged 20+ PPG and 8+ APG?",
            "Who had seasons most similar to 2025 SGA?",
            "Find seasons similar to 2016 Stephen Curry for shooting.",
            "Compare 2016 Curry and 2024 Luka.",
            "Where was Stephen Curry most efficient in 2022?",
            "Which NBA teams had the best eFG% since 2003?",
            "Show WNBA players since 2020 who shot at least 40% from three.",
            "Best WNBA defensive players",
            "Who are the best rim protectors?",
        ],
    }


@router.post("/ask")
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        return _fail("Ask me something about the NBA or WNBA data.")

    for pattern, message in OUT_OF_SCOPE:
        if re.search(pattern, question, re.I):
            return _fail(message)

    lg_guess = leagues.get(req.league)
    query = parse_rules(question, req.league)
    parser = "rules"
    if query is None:
        query = parse_llm(question, req.league,
                          data.available_metrics(lg_guess),
                          sorted(set(TEAM_METRIC_ALIASES.values())))
        parser = "llm"
    if query is None:
        return _fail(
            "I couldn't turn that into a query I can run. Try naming a stat and a "
            "number — for example “players since 2015 averaging 25+ points”.",
            parser=parser,
        )

    lg = _infer_league(question, query.league, req.league)
    invalid = _validate(query, lg)
    if invalid:
        return {**invalid, "parser": parser, "query": query.model_dump()}

    # Resolve any named players before executing.
    resolved: list[dict] = []
    for ref in query.players:
        who, err = _resolve_player(lg, ref)
        if err:
            return {**err, "parser": parser, "query": query.model_dump()}
        resolved.append(who)

    if query.intent == "explorer":
        result = _run_explorer(query, lg)
    elif query.intent == "similarity":
        if not resolved:
            return _fail("Tell me which player-season to match against.")
        result = _run_similarity(query, lg, resolved[0])
    elif query.intent == "compare":
        if len(resolved) < 2:
            return _fail("Name two player-seasons to compare.")
        result = _run_compare(query, lg, resolved)
    elif query.intent == "shot_analysis":
        if not resolved:
            return _fail("Tell me whose shots to look at.")
        result = _run_shots(query, lg, resolved[0])
    else:
        result = _run_team_explorer(query, lg)

    return analytics.json_safe({
        **result,
        "intent": query.intent,
        "league": lg.key,
        # The answer's league can differ from the one the UI is showing, so it
        # carries its own season format — a WNBA row must not render as 2014-15.
        "season_format": lg.season_format,
        "query": query.model_dump(),
        "parser": parser,
    })
