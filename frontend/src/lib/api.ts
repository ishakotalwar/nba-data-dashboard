const BASE = "/api";

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) msg = body.detail;
    } catch {}
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export type LeagueKey = string;

export type PlayerBio = {
  height?: string | null;
  weight?: string | null;
  position?: string | null;
  birthdate?: string | null;
  birthplace?: string | null;
  age?: number | null;
};

export type PlayerInfo = {
  player_id: number;
  name: string;
  seasons: string[];
  teams: { season: string; team_abbr: string | null }[];
  bio: PlayerBio;
};

/** The app's selection entity: a player in a specific season. */
export type PlayerSeason = {
  playerId?: number;
  playerName: string;
  season: string;
};

export type LeagueInfo = {
  key: LeagueKey;
  label: string;
  available: boolean;
};

export type Meta = {
  league: LeagueKey;
  league_label: string;
  /** How to render a stored season label; see lib/season.ts. */
  season_format?: "range" | "year";
  players: string[];
  /** player_name -> ESPN athlete id, used to build headshot URLs. */
  player_ids: Record<string, number>;
  teams: string[];
  seasons: string[];
  /** Seasons with rebuilt five-man lineups — a shorter list than `seasons`. */
  lineup_seasons: string[];
  metrics: string[];
  /** Basis key -> label, e.g. { game: "Per game", per100: "Per 100 possessions" }. */
  rate_bases: Record<string, string>;
  /** Which games a shot chart can draw from: regular, playoffs, both. */
  shot_season_types: Record<string, string>;
  /** Seasons with player impact ratings — the same range as `lineup_seasons`. */
  rating_seasons: string[];
  invert_metrics: string[];
  // Three-point geometry for this league, in tenths of a foot from the hoop.
  court: { arc: number; corner: number };
};

/** Query string with `league` appended, so every request is league-scoped. */
function q_(league: LeagueKey, extra: Record<string, string> = {}) {
  return new URLSearchParams({ ...extra, league });
}

export const api = {
  leagues: () =>
    fetch(`${BASE}/leagues`).then((r) =>
      j<{ leagues: LeagueInfo[]; default: LeagueKey }>(r)
    ),

  meta: (league: LeagueKey) =>
    fetch(`${BASE}/meta?${q_(league)}`).then((r) => j<Meta>(r)),

  players: (league: LeagueKey, q = "", limit = 50) =>
    fetch(`${BASE}/players?${q_(league, { q, limit: String(limit) })}`).then((r) => j<any>(r)),

  player: (playerId: number, league: LeagueKey) =>
    fetch(`${BASE}/player/${playerId}?${q_(league)}`).then((r) => j<PlayerInfo>(r)),

  /** Natural-language question -> structured query -> grounded result. */
  ask: (question: string, league: LeagueKey) =>
    fetch(`${BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, league }),
    }).then((r) => j<any>(r)),

  /** Dates that have scheduled games, for the calendar grid. */
  predictCalendar: (league: LeagueKey) =>
    fetch(`${BASE}/predictions/calendar?${q_(league)}`).then((r) => j<any>(r)),

  /** One date's games, each with the model's read on it. */
  predictSchedule: (league: LeagueKey, date: string) =>
    fetch(`${BASE}/predictions/schedule?${q_(league, { date })}`).then((r) => j<any>(r)),

  /** One game: the Elo read, plus a projected line for every rotation player. */
  predictGame: (league: LeagueKey, gameId: string) =>
    fetch(`${BASE}/predictions/game/${gameId}?${q_(league)}`).then((r) => j<any>(r)),

  /** Team power ratings, the Elo backtest and its calibration. */
  predictTeams: (league: LeagueKey) =>
    fetch(`${BASE}/predictions/teams?${q_(league)}`).then((r) => j<any>(r)),

  predictMatchup: (home: string, away: string, league: LeagueKey) =>
    fetch(`${BASE}/predictions/matchup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ home, away, league }),
    }).then((r) => j<any>(r)),

  /** One player's projection, the seasons behind it and the model's own error. */
  predictPlayer: (playerId: number, league: LeagueKey) =>
    fetch(`${BASE}/predictions/player/${playerId}?${q_(league)}`).then((r) => j<any>(r)),

  askCapabilities: (league: LeagueKey) =>
    fetch(`${BASE}/ask/capabilities?${q_(league)}`).then((r) => j<any>(r)),

  playerSeason: (playerId: number, season: string, league: LeagueKey, per = "game") =>
    fetch(
      `${BASE}/player/${playerId}/season/${encodeURIComponent(season)}?${q_(league, { per })}`
    ).then((r) => j<any>(r)),

  playerCareer: (playerId: number, league: LeagueKey, recent = 10, per = "game") =>
    fetch(`${BASE}/player/${playerId}/career?${q_(league, { recent: String(recent), per })}`)
      .then((r) => j<any>(r)),

  compare: (body: {
    selections: { player_id: number; season?: string }[];
    metrics: string[];
    league: LeagueKey;
    mode: "season" | "career";
    per?: string;
  }) =>
    fetch(`${BASE}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j<any>(r)),

  similarityPresets: () =>
    fetch(`${BASE}/similarity/presets`).then((r) => j<any>(r)),

  similarity: (body: {
    player_id: number;
    season: string;
    league: LeagueKey;
    preset?: string;
    weights?: Record<string, number>;
    k?: number;
    min_gp?: number;
    same_season_only?: boolean;
  }) =>
    fetch(`${BASE}/similarity`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j<any>(r)),

  teamSeries: (team: string, league: LeagueKey) =>
    fetch(`${BASE}/teams/series?${q_(league, { team })}`).then((r) => j<any>(r)),

  explorerFields: (league: LeagueKey) =>
    fetch(`${BASE}/explorer/fields?${q_(league)}`).then((r) => j<any>(r)),

  explorer: (body: any) =>
    fetch(`${BASE}/explorer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j<any>(r)),

  teamsLeague: (season: string, league: LeagueKey) =>
    fetch(`${BASE}/teams/league?${q_(league, { season })}`).then((r) => j<any>(r)),

  teamsCompare: (
    a: { team: string; season: string },
    b: { team: string; season: string },
    league: LeagueKey
  ) =>
    fetch(`${BASE}/teams/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a, b, league }),
    }).then((r) => j<any>(r)),

  teamsRankings: (metric: string, league: LeagueKey, limit = 15) =>
    fetch(`${BASE}/teams/rankings?${q_(league, { metric, limit: String(limit) })}`)
      .then((r) => j<any>(r)),

  /** Five-man lineups for a season, league-wide or for one team. */
  teamLineups: (
    season: string,
    league: LeagueKey,
    opts: { team?: string; minMinutes?: number; limit?: number } = {}
  ) =>
    fetch(
      `${BASE}/teams/lineups?${q_(league, {
        season,
        ...(opts.team ? { team: opts.team } : {}),
        min_minutes: String(opts.minMinutes ?? 50),
        limit: String(opts.limit ?? 250),
      })}`
    ).then((r) => j<any>(r)),

  teamFactors: (team: string, season: string, league: LeagueKey) =>
    fetch(`${BASE}/teams/factors?${q_(league, { team, season })}`).then((r) => j<any>(r)),

  shots: (
    playerId: number,
    season: string,
    mode: "scatter" | "hex",
    league: LeagueKey,
    seasonType = "regular"
  ) =>
    fetch(
      `${BASE}/shots?${q_(league, {
        player_id: String(playerId), season, mode, season_type: seasonType,
      })}`
    ).then((r) => j<any>(r)),

  shotZones: (playerId: number, season: string, league: LeagueKey, seasonType = "regular") =>
    fetch(
      `${BASE}/shots/zones?${q_(league, {
        player_id: String(playerId), season, season_type: seasonType,
      })}`
    ).then((r) => j<any>(r)),

  shotCompare: (
    a: { player_id: number; season: string },
    b: { player_id: number; season: string },
    league: LeagueKey,
    seasonType = "regular"
  ) =>
    fetch(`${BASE}/shots/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a, b, league, season_type: seasonType }),
    }).then((r) => j<any>(r)),

  /** Player impact ratings (RAPM) for one season. */
  playerRatings: (
    season: string,
    league: LeagueKey,
    opts: { minPoss?: number; limit?: number; team?: string } = {}
  ) =>
    fetch(
      `${BASE}/players/ratings?${q_(league, {
        season,
        min_poss: String(opts.minPoss ?? 500),
        limit: String(opts.limit ?? 100),
        ...(opts.team ? { team: opts.team } : {}),
      })}`
    ).then((r) => j<any>(r)),
};
