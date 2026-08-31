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
  players: string[];
  /** player_name -> ESPN athlete id, used to build headshot URLs. */
  player_ids: Record<string, number>;
  teams: string[];
  seasons: string[];
  metrics: string[];
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

  playerSeason: (playerId: number, season: string, league: LeagueKey) =>
    fetch(`${BASE}/player/${playerId}/season/${encodeURIComponent(season)}?${q_(league)}`)
      .then((r) => j<any>(r)),

  playerCareer: (playerId: number, league: LeagueKey, recent = 10) =>
    fetch(`${BASE}/player/${playerId}/career?${q_(league, { recent: String(recent) })}`)
      .then((r) => j<any>(r)),

  compare: (body: {
    selections: { player_id: number; season?: string }[];
    metrics: string[];
    league: LeagueKey;
    mode: "season" | "career";
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

  teamFactors: (team: string, season: string, league: LeagueKey) =>
    fetch(`${BASE}/teams/factors?${q_(league, { team, season })}`).then((r) => j<any>(r)),

  shots: (playerId: number, season: string, mode: "scatter" | "hex", league: LeagueKey) =>
    fetch(`${BASE}/shots?${q_(league, { player_id: String(playerId), season, mode })}`)
      .then((r) => j<any>(r)),

  shotZones: (playerId: number, season: string, league: LeagueKey) =>
    fetch(`${BASE}/shots/zones?${q_(league, { player_id: String(playerId), season })}`)
      .then((r) => j<any>(r)),

  shotCompare: (
    a: { player_id: number; season: string },
    b: { player_id: number; season: string },
    league: LeagueKey
  ) =>
    fetch(`${BASE}/shots/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a, b, league }),
    }).then((r) => j<any>(r)),
};
