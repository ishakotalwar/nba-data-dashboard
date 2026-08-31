"""Turning an English question into an AskQuery.

Two parsers, tried in order:

1. `parse_rules` — regex over the shapes basketball questions actually take.
   Free, instant, offline, and deterministic, which means the common questions
   behave identically on every run and the feature works with no API key.
2. `parse_llm` — Claude, with the schema enforced by structured output, for
   anything the rules miss. Only ever used to *understand* the question; it
   never sees or produces a statistic.

Both return an AskQuery or None. The caller decides what to do with None.
"""
from __future__ import annotations

import os
import re

from .ask_schema import (
    METRIC_ALIASES,
    PRESET_ALIASES,
    SUPERLATIVE_CATEGORIES,
    TEAM_METRIC_ALIASES,
    AskFilter,
    AskQuery,
    PlayerRef,
)

# A leaderboard question: "best defensive players", "top scorers since 2015",
# "worst three-point shooters". No numbers, just a ranking.
_SUPERLATIVE = re.compile(
    r"\b(best|top|greatest|leading|leaders?|most|worst|lowest|fewest)\b", re.I
)
_ASCENDING = re.compile(r"\b(worst|lowest|fewest)\b", re.I)
# Rankings need a games floor or a three-game call-up tops every list.
RANKING_MIN_GP = 15

# Metrics stored as a fraction (0.454) but spoken as a percentage ("45%").
FRACTION_METRICS = {"fg_pct", "three_pct", "ft_pct", "ts_pct", "usg_pct"}

_TEAM_WORDS = re.compile(r"\bteams?\b|\bfranchises?\b", re.I)
_SIMILAR_WORDS = re.compile(r"\bsimilar\b|\blike\b|\bcomparable\b|\breminiscent\b", re.I)
_COMPARE_WORDS = re.compile(r"\bcompare\b|\bversus\b|\bvs\.?\b|\bagainst\b|\bside by side\b", re.I)
_SHOT_WORDS = re.compile(
    r"\bshot chart\b|\bshots?\b|\bzones?\b|\bfrom where\b|\bwhere (?:was|is|did)\b"
    r"|\bmost efficient\b|\bhot spots?\b",
    re.I,
)

# "since 2010", "from 2015", "after 2003"
_SINCE = re.compile(r"\b(?:since|from|after)\s+(\d{4})\b", re.I)
# "in 2022", "during 2016", "for 2024"
_IN_YEAR = re.compile(r"\b(?:in|during|for)\s+(\d{4})\b", re.I)
# "between 2010 and 2015"
_BETWEEN_YEARS = re.compile(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", re.I)
# a bare year immediately before a capitalised name: "2016 Stephen Curry"
_YEAR_NAME = re.compile(r"\b(\d{4})\s+([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*)")

# "25+ points", "at least 25 points", "40% from three", "under 3 turnovers"
_AT_LEAST = r"(?:at least|minimum(?: of)?|no fewer than|>=?)"
_MORE_THAN = r"(?:more than|over|above|greater than|>)"
_AT_MOST = r"(?:at most|no more than|<=?)"
_LESS_THAN = r"(?:less than|under|below|fewer than|<)"

_ALIAS_RE = "|".join(
    sorted((re.escape(a) for a in METRIC_ALIASES), key=len, reverse=True)
)


def _canonical_value(metric: str, value: float, saw_percent: bool) -> float:
    """'40% from three' -> 0.40; '25 points' -> 25.0."""
    if metric in FRACTION_METRICS and (saw_percent or value > 1.5):
        return value / 100.0
    return value


def _find_conditions(q: str) -> list[AskFilter]:
    """Every 'metric OP value' phrase in the question, in either word order."""
    out: list[AskFilter] = []
    seen: set[tuple[str, str, float]] = set()

    # number-first: "25+ points", "at least 25 points per game", "40% from three"
    number_first = re.compile(
        rf"(?:{_AT_LEAST}\s+|{_MORE_THAN}\s+|{_AT_MOST}\s+|{_LESS_THAN}\s+)?"
        rf"(\d+(?:\.\d+)?)\s*(%?)\s*(\+)?\s*"
        rf"(?:per game\s+)?(?:or more\s+)?"
        rf"(?:{_ALIAS_RE})",
        re.I,
    )
    for m in number_first.finditer(q):
        phrase = m.group(0)
        alias = _match_alias(phrase)
        if not alias:
            continue
        op = _operator_for(phrase, plus=bool(m.group(3)))
        value = _canonical_value(alias, float(m.group(1)), m.group(2) == "%")
        key = (alias, op, value)
        if key not in seen:
            seen.add(key)
            out.append(AskFilter(metric=alias, op=op, value=value))

    # metric-first: "points above 25", "three point percentage over 40%"
    metric_first = re.compile(
        rf"({_ALIAS_RE})\s+(?:of\s+)?"
        rf"({_AT_LEAST}|{_MORE_THAN}|{_AT_MOST}|{_LESS_THAN})\s+"
        rf"(\d+(?:\.\d+)?)\s*(%?)",
        re.I,
    )
    for m in metric_first.finditer(q):
        alias = METRIC_ALIASES.get(m.group(1).lower())
        if not alias:
            continue
        op = _operator_for(m.group(2), plus=False)
        value = _canonical_value(alias, float(m.group(3)), m.group(4) == "%")
        key = (alias, op, value)
        if key not in seen:
            seen.add(key)
            out.append(AskFilter(metric=alias, op=op, value=value))
    return out


def _match_alias(phrase: str) -> str | None:
    """Longest alias appearing in the phrase wins, so 'three point percentage'
    beats 'points'."""
    low = phrase.lower()
    best: tuple[int, str] | None = None
    for alias, canonical in METRIC_ALIASES.items():
        if alias in low and (best is None or len(alias) > best[0]):
            best = (len(alias), canonical)
    return best[1] if best else None


def _operator_for(phrase: str, plus: bool) -> str:
    low = phrase.lower()
    if plus or re.search(_AT_LEAST, low):
        return ">="
    if re.search(_MORE_THAN, low):
        return ">"
    if re.search(_AT_MOST, low):
        return "<="
    if re.search(_LESS_THAN, low):
        return "<"
    return ">="  # "averaged 25 points" reads as a floor, not an exact match


def _league(q: str, fallback: str | None) -> str | None:
    if re.search(r"\bwnba\b", q, re.I):
        return "wnba"
    if re.search(r"\bnba\b", q, re.I):
        return "nba"
    return fallback


def _seasons(q: str) -> tuple[str | None, str | None]:
    m = _BETWEEN_YEARS.search(q)
    if m:
        lo, hi = sorted((m.group(1), m.group(2)))
        return lo, hi
    m = _SINCE.search(q)
    if m:
        return m.group(1), None
    m = _IN_YEAR.search(q)
    if m:
        return m.group(1), m.group(1)
    return None, None


def _named_player_seasons(q: str) -> list[PlayerRef]:
    """'2016 Stephen Curry and 2024 Luka' -> two refs, season attached."""
    refs: list[PlayerRef] = []
    for m in _YEAR_NAME.finditer(q):
        name = _clean_name(m.group(2))
        if name:
            refs.append(PlayerRef(player=name, season=m.group(1)))
    return refs


def _clean_name(raw: str) -> str:
    """Strip sentence punctuation and connectives a capitalisation run picks up
    — "Luka." and "Curry and" are the same player as "Luka" and "Curry"."""
    name = raw.strip().strip(".,;:!?")
    name = re.sub(r"\s+(And|Vs|Versus|Or)$", "", name, flags=re.I)
    return name.strip().strip(".,;:!?").strip()


def _preset(q: str) -> str | None:
    for alias, preset in PRESET_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q, re.I):
            return preset
    return None


def _ranking_metric(q: str) -> tuple[str | None, str | None]:
    """What a leaderboard question wants ranked, and any caveat about it.

    A named metric ("top scorers", "most blocks") wins over a broad category
    ("best defensive players"), because it is the less ambiguous reading.
    """
    low = q.lower()
    named = None
    best_len = 0
    for alias, canonical in METRIC_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low) and len(alias) > best_len:
            named, best_len = canonical, len(alias)
    if named:
        return named, None

    for word, (metric, note) in SUPERLATIVE_CATEGORIES.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            return metric, note
    return None, None


def _team_metric(q: str) -> str | None:
    low = q.lower()
    best: tuple[int, str] | None = None
    for alias, canonical in TEAM_METRIC_ALIASES.items():
        if alias in low and (best is None or len(alias) > best[0]):
            best = (len(alias), canonical)
    return best[1] if best else None


def parse_rules(question: str, league: str | None = None) -> AskQuery | None:
    """Best-effort parse with no model in the loop. None if nothing matched."""
    q = question.strip()
    if not q:
        return None
    lg = _league(q, league)
    season_from, season_to = _seasons(q)

    # Team questions: "which teams had the best eFG% since 2003"
    if _TEAM_WORDS.search(q):
        metric = _team_metric(q)
        if metric:
            return AskQuery(
                intent="team_explorer", league=lg, season_from=season_from,
                season_to=season_to, metric=metric,
                dir="asc" if _ASCENDING.search(q) else "desc",
                limit=10,
            )

    refs = _named_player_seasons(q)

    # Shot questions: "where was Stephen Curry most efficient in 2022"
    if _SHOT_WORDS.search(q) and not _SIMILAR_WORDS.search(q):
        who = refs or _bare_names(q, season_from or season_to)
        if who:
            ref = who[0]
            season = ref.season or season_from or season_to
            return AskQuery(intent="shot_analysis", league=lg,
                            players=[PlayerRef(player=ref.player, season=season)])

    # Similarity: "seasons most similar to 2025 SGA"
    if _SIMILAR_WORDS.search(q):
        who = refs or _bare_names(q, season_from or season_to)
        if who:
            ref = who[0]
            return AskQuery(
                intent="similarity", league=lg,
                players=[PlayerRef(player=ref.player,
                                   season=ref.season or season_from or season_to)],
                preset=_preset(q) or "Overall", limit=10,
            )

    # Comparison: "compare 2016 Curry and 2024 Luka"
    if _COMPARE_WORDS.search(q) and len(refs) >= 2:
        return AskQuery(intent="compare", league=lg, players=refs[:5])

    # Leaderboards: "best WNBA defensive players", "top scorers since 2015".
    # Checked before conditions so "most points" doesn't fall through as noise,
    # but after the intents above so "most similar to X" still means similarity.
    if _SUPERLATIVE.search(q) and not _SIMILAR_WORDS.search(q):
        metric, note = _ranking_metric(q)
        if metric:
            return AskQuery(
                intent="explorer", league=lg, season_from=season_from,
                season_to=season_to, sort=metric,
                dir="asc" if _ASCENDING.search(q) else "desc",
                min_gp=RANKING_MIN_GP, limit=25, note=note,
            )

    # Everything else with a numeric condition is an explorer query.
    filters = _find_conditions(q)
    if filters:
        sort = filters[0].metric
        return AskQuery(
            intent="explorer", league=lg, season_from=season_from, season_to=season_to,
            filters=filters, sort=sort,
            dir="asc" if re.search(r"\bworst\b|\blowest\b|\bfewest\b", q, re.I) else "desc",
            limit=25,
        )
    return None


def _bare_names(q: str, season: str | None) -> list[PlayerRef]:
    """Capitalised runs that look like a name, for questions with no inline year."""
    stop = {
        "Which", "Who", "What", "Where", "Show", "Find", "Compare", "NBA", "WNBA",
        "The", "Most", "Best", "Season", "Seasons", "Player", "Players", "And",
    }
    out: list[PlayerRef] = []
    for m in re.finditer(r"\b([A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+)*)", q):
        name = _clean_name(m.group(1))
        words = [w for w in name.split() if w not in stop]
        if not words:
            continue
        cleaned = " ".join(words)
        # A single all-caps token (SGA, MJ) is a plausible nickname; otherwise
        # require it to look like a name rather than a sentence opener.
        if len(cleaned) < 2 or cleaned in stop:
            continue
        out.append(PlayerRef(player=cleaned, season=season))
    return out


# --------------------------------------------------------------------------
# LLM parser — understanding only, never statistics.
# --------------------------------------------------------------------------

_SYSTEM = """You convert basketball questions into a structured query for an \
NBA/WNBA analytics app. You never answer the question and never state any \
statistic — a separate engine runs your query against a local database.

Choose exactly one intent:
- explorer: filter player-seasons by numeric conditions
- similarity: find seasons similar to one player-season
- compare: put named player-seasons side by side
- shot_analysis: where a player shot from, versus league average
- team_explorer: rank team-seasons by one team metric

Rules:
- metric must be one of: {metrics}
- team metric (team_explorer only) must be one of: {team_metrics}
- preset (similarity only) must be one of: Overall, Scoring, Shooting, Playmaking, Defense
- percentages are stored as fractions: 40% -> 0.4
- seasons are a plain year string, e.g. "2016"
- league is "nba" or "wnba"; leave it null if the question does not say
- put player names exactly as written by the user; do not correct or expand them
"""


def parse_llm(question: str, league: str | None, metrics: list[str],
              team_metrics: list[str]) -> AskQuery | None:
    """Parse with Claude. Returns None if unavailable or the call fails —
    the feature degrades to the rule parser rather than erroring."""
    try:
        import anthropic
    except ImportError:
        return None
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None

    try:
        client = anthropic.Anthropic()
        response = client.messages.parse(
            model="claude-opus-5",
            max_tokens=2000,
            output_config={"effort": "low"},
            system=_SYSTEM.format(metrics=", ".join(metrics),
                                  team_metrics=", ".join(team_metrics)),
            messages=[{
                "role": "user",
                "content": f"Current league in the UI: {league or 'nba'}\nQuestion: {question}",
            }],
            output_format=AskQuery,
        )
        return response.parsed_output
    except Exception:
        # Malformed output, no credit, network down — all the same to the caller.
        return None


def llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
