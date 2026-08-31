"""The structured query language behind /api/ask.

A natural-language question is turned into one of these objects and nothing
else — every statistic in the answer comes from executing the object against
local Parquet, never from a model. Keeping the schema in its own module lets
both parsers (deterministic and LLM) and the executor share one definition.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

INTENTS = ("explorer", "similarity", "compare", "shot_analysis", "team_explorer")
OPERATORS = (">", ">=", "<", "<=", "=", "between")

# Natural language -> the canonical column names in backend/data.py. The values
# here must always be a subset of data.CANDIDATE_METRICS; ask.py asserts that.
METRIC_ALIASES: dict[str, str] = {
    "points": "pts", "ppg": "pts", "points per game": "pts",
    "scoring": "pts", "pts": "pts",
    "rebounds": "reb", "rebound": "reb", "rpg": "reb", "boards": "reb",
    "rebounds per game": "reb", "reb": "reb",
    "assists": "ast", "assist": "ast", "apg": "ast", "dimes": "ast",
    "assists per game": "ast", "ast": "ast",
    "steals": "stl", "steal": "stl", "spg": "stl", "stl": "stl",
    "blocks": "blk", "block": "blk", "bpg": "blk", "blk": "blk",
    "turnovers": "tov", "turnover": "tov", "tov": "tov",
    "field goal percentage": "fg_pct", "field goal %": "fg_pct", "fg%": "fg_pct",
    "fg pct": "fg_pct", "shooting percentage": "fg_pct", "fg_pct": "fg_pct",
    "three point percentage": "three_pct", "three-point percentage": "three_pct",
    "3 point percentage": "three_pct", "3-point percentage": "three_pct",
    "three point %": "three_pct", "3p%": "three_pct", "3pt%": "three_pct",
    "from three": "three_pct", "from deep": "three_pct", "three pct": "three_pct",
    "three point shooting": "three_pct", "three-point shooting": "three_pct",
    "3 point shooting": "three_pct", "three_pct": "three_pct",
    "free throw percentage": "ft_pct", "free throw %": "ft_pct", "ft%": "ft_pct",
    "from the line": "ft_pct", "ft_pct": "ft_pct",
    "true shooting": "ts_pct", "true shooting percentage": "ts_pct",
    "ts%": "ts_pct", "efficiency": "ts_pct", "ts_pct": "ts_pct",
    "usage": "usg_pct", "usage rate": "usg_pct", "usg%": "usg_pct",
    "usg_pct": "usg_pct",
    "offensive rating": "ortg", "ortg": "ortg",
    "defensive rating": "drtg", "drtg": "drtg",
    # How people describe a role rather than name a column.
    "rim protector": "blk", "rim protectors": "blk", "rim protection": "blk",
    "shot blocker": "blk", "shot blockers": "blk", "shot blocking": "blk",
    "floor general": "ast", "floor generals": "ast", "playmaker": "ast",
    "playmakers": "ast", "passer": "ast", "passers": "ast",
    "distributor": "ast", "distributors": "ast",
    "sharpshooter": "three_pct", "sharpshooters": "three_pct",
    "three point specialist": "three_pct", "3 point specialist": "three_pct",
    "three point shooter": "three_pct", "three point shooters": "three_pct",
    "3 point shooter": "three_pct", "3 point shooters": "three_pct",
    "three-point shooter": "three_pct", "three-point shooters": "three_pct",
    "scorer": "pts", "scorers": "pts", "bucket getter": "pts",
    "rebounder": "reb", "rebounders": "reb", "glass cleaner": "reb",
    "ball hawk": "stl", "ball hawks": "stl", "thief": "stl",
    "free throw shooter": "ft_pct", "free throw shooters": "ft_pct",
}

# Categories a superlative can name ("best defensive players"). These are
# deliberately kept out of METRIC_ALIASES: they are ambiguous, so they only
# resolve when the question is explicitly asking for a ranking, and the answer
# always says which single metric it ranked on.
SUPERLATIVE_CATEGORIES: dict[str, tuple[str, str | None]] = {
    "defensive": ("blk", "Full Court has no per-player defensive rating, so this "
                         "ranks on blocks — ask for steals for perimeter defence."),
    "defense": ("blk", "Full Court has no per-player defensive rating, so this "
                       "ranks on blocks — ask for steals for perimeter defence."),
    "defenders": ("blk", "Full Court has no per-player defensive rating, so this "
                         "ranks on blocks — ask for steals for perimeter defence."),
    "offensive": ("pts", None),
    "scoring": ("pts", None),
    "shooting": ("ts_pct", None),
    "playmaking": ("ast", None),
    "passing": ("ast", None),
    "rebounding": ("reb", None),
    "efficient": ("ts_pct", None),
    "efficiency": ("ts_pct", None),
}

# Short prose names for summaries written server-side.
METRIC_LABELS: dict[str, str] = {
    "pts": "points per game", "reb": "rebounds", "ast": "assists",
    "stl": "steals", "blk": "blocks", "tov": "turnovers",
    "fg_pct": "field goal %", "three_pct": "three-point %", "ft_pct": "free throw %",
    "ts_pct": "true shooting %", "usg_pct": "usage rate",
    "ortg": "offensive rating", "drtg": "defensive rating",
}

# Team metrics live in teams.py's RANKABLE set, which uses display-style names.
TEAM_METRIC_ALIASES: dict[str, str] = {
    "efg": "eFG%", "efg%": "eFG%", "effective field goal": "eFG%",
    "effective field goal percentage": "eFG%", "efg pct": "eFG%",
    "turnover rate": "TOV%", "tov%": "TOV%", "turnover percentage": "TOV%",
    "offensive rebound rate": "ORB%", "orb%": "ORB%", "offensive rebounding": "ORB%",
    "free throw rate": "FT rate", "ft rate": "FT rate",
    "net rating": "net", "net": "net", "point differential": "net",
    "offensive rating": "ortg", "ortg": "ortg",
    "defensive rating": "drtg", "drtg": "drtg",
    "pace": "pace",
    "wins": "wins", "win percentage": "win_pct", "win pct": "win_pct",
    "record": "win_pct",
}

# The similarity router keys its presets with capitals; users won't type them.
PRESET_ALIASES: dict[str, str] = {
    "overall": "Overall", "scoring": "Scoring", "shooting": "Shooting",
    "playmaking": "Playmaking", "passing": "Playmaking", "defense": "Defense",
    "defensive": "Defense",
}


class AskFilter(BaseModel):
    """One metric condition, e.g. pts >= 25."""
    metric: str = Field(description="Canonical metric key, e.g. 'pts' or 'three_pct'")
    op: Literal[">", ">=", "<", "<=", "=", "between"]
    value: float
    value2: float | None = Field(default=None, description="Upper bound, 'between' only")


class PlayerRef(BaseModel):
    """A player the user named, before resolution to an id."""
    player: str
    season: str | None = None


class AskQuery(BaseModel):
    """The validated request an /api/ask answer is computed from."""
    intent: Literal["explorer", "similarity", "compare", "shot_analysis", "team_explorer"]
    league: str | None = None
    season_from: str | None = None
    season_to: str | None = None
    filters: list[AskFilter] = []
    players: list[PlayerRef] = []
    preset: str | None = None
    metric: str | None = Field(default=None, description="Metric to sort/rank on")
    sort: str | None = None
    dir: Literal["asc", "desc"] = "desc"
    limit: int = 20
    min_gp: int = Field(default=0, description="Games played floor, used by rankings")
    note: str | None = Field(default=None, description="Caveat to show with the answer")


class AskRequest(BaseModel):
    question: str
    league: str | None = Field(default=None, description="The UI's current league")
