"""League registry.

Everything that differs between the NBA and WNBA lives here so the rest of the
backend can stay league-agnostic and just pass a league key around.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException


@dataclass(frozen=True)
class League:
    key: str            # url/query value, e.g. "wnba"
    league_id: str      # stats.nba.com LeagueID, e.g. "10"
    label: str          # display name
    season_format: str  # "range" -> 2024-25, "year" -> 2024
    suffix: str         # parquet filename suffix, e.g. "_wnba"
    season_start_month: int  # used to age players at season start
    three_point_arc: float   # radius, tenths of a foot from the hoop
    three_point_corner: float  # corner-3 line |x|, same units
    team_minutes: int   # five players times the length of a game

    def season(self, start_year: int) -> str:
        """Canonical season string for a starting calendar year."""
        if self.season_format == "year":
            return str(start_year)
        return f"{start_year}-{str(start_year + 1)[-2:]}"

    def season_start_year(self, season: str) -> int | None:
        """Calendar year in which `season` tipped off.

        A league whose season begins in the second half of the year is
        labelled by the year it finishes in, so it started the year before.
        """
        try:
            year = int(str(season)[:4])
        except (TypeError, ValueError):
            return None
        return year - 1 if self.season_start_month >= 7 else year

    def canonical_team(self, abbr: str) -> str:
        """Today's abbreviation for a franchise that used to use another."""
        return TEAM_ALIASES.get(self.key, {}).get(str(abbr), str(abbr))

    def display_season(self, season: str) -> str:
        """Stored season label -> label for humans.

        Stored labels are a plain year: the year the season *ends* for a
        "range" league (NBA `2026` ran Oct 2025 - Apr 2026, shown as 2025-26),
        and the season's own year for a "year" league (WNBA `2026`).
        """
        if self.season_format == "year":
            return str(season)
        try:
            end_year = int(str(season)[:4])
        except (TypeError, ValueError):
            return str(season)
        return f"{end_year - 1}-{str(end_year)[-2:]}"


NBA = League(
    key="nba", league_id="00", label="NBA", season_format="range", suffix="_nba",
    season_start_month=10, three_point_arc=237.5, three_point_corner=220.0,
    team_minutes=240,  # 48-minute game
)
# WNBA arc is 22' 1.75"; corner line 21' 7.75" (FIBA-derived, adopted 2013).
WNBA = League(
    key="wnba", league_id="10", label="WNBA", season_format="year", suffix="_wnba",
    season_start_month=5, three_point_arc=221.5, three_point_corner=216.5,
    team_minutes=200,  # 40-minute game
)

# A franchise whose abbreviation changed in the source data — through
# relocation, a rebrand, or simple inconsistency across eras — is still one
# franchise. Without this its history splits in two and anything cumulative
# (an Elo rating, say) starts over. Keyed to the abbreviation used today.
TEAM_ALIASES: dict[str, dict[str, str]] = {
    "nba": {
        "SEA": "OKC",   # SuperSonics -> Thunder, 2008
        "NJ": "BKN",    # New Jersey -> Brooklyn Nets, 2012
    },
    "wnba": {
        "CT": "CON", "CONN": "CON",  # Connecticut Sun, three spellings
        "LOS": "LA",                 # Los Angeles Sparks
        "NYL": "NY",                 # New York Liberty
        "PHO": "PHX",                # Phoenix Mercury
        "WAS": "WSH",                # Washington Mystics
        "SAS": "LV", "SA": "LV",     # San Antonio Stars -> Las Vegas Aces, 2018
        "TUL": "DAL",                # Tulsa Shock -> Dallas Wings, 2016
    },
}

LEAGUES: dict[str, League] = {lg.key: lg for lg in (NBA, WNBA)}
DEFAULT = NBA


def get(key: str | None) -> League:
    """Resolve a league key, defaulting to NBA. Raises 400 on an unknown key."""
    if not key:
        return DEFAULT
    lg = LEAGUES.get(key.strip().lower())
    if lg is None:
        raise HTTPException(
            400, f"Unknown league {key!r}. Expected one of: {', '.join(sorted(LEAGUES))}."
        )
    return lg
