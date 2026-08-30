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
function q(league: LeagueKey, extra: Record<string, string> = {}) {
  return new URLSearchParams({ ...extra, league });
}

export const api = {
  leagues: () =>
    fetch(`${BASE}/leagues`).then((r) =>
      j<{ leagues: LeagueInfo[]; default: LeagueKey }>(r)
    ),

  meta: (league: LeagueKey) =>
    fetch(`${BASE}/meta?${q(league)}`).then((r) => j<Meta>(r)),

  compare: (params: {
    players: string[];
    metrics: string[];
    seasonLo?: string;
    seasonHi?: string;
    league: LeagueKey;
  }) => {
    const p = q(params.league, {
      players: params.players.join(","),
      metrics: params.metrics.join(","),
    });
    if (params.seasonLo) p.set("seasonLo", params.seasonLo);
    if (params.seasonHi) p.set("seasonHi", params.seasonHi);
    return fetch(`${BASE}/compare?${p}`).then((r) => j<any>(r));
  },

  trends: (params: {
    player: string;
    metrics: string[];
    leagueAvg?: boolean;
    league: LeagueKey;
  }) =>
    fetch(
      `${BASE}/trends?${q(params.league, {
        player: params.player,
        metrics: params.metrics.join(","),
        leagueAvg: String(params.leagueAvg ?? true),
      })}`
    ).then((r) => j<any>(r)),

  percentiles: (player: string, season: string, league: LeagueKey) =>
    fetch(`${BASE}/percentiles?${q(league, { player, season })}`).then((r) => j<any>(r)),

  similar: (body: {
    anchor: string;
    k: number;
    weights: Record<string, number>;
    seasonLo?: string;
    seasonHi?: string;
    league: LeagueKey;
  }) =>
    fetch(`${BASE}/similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j<any>(r)),

  gamelog: (
    player: string,
    season: string,
    stat: string,
    window: number,
    league: LeagueKey
  ) =>
    fetch(
      `${BASE}/gamelog?${q(league, {
        player,
        season,
        stat,
        window: String(window),
      })}`
    ).then((r) => j<any>(r)),

  ageCurves: (body: { players: string[]; metric: string; league: LeagueKey }) =>
    fetch(`${BASE}/age-curves`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j<any>(r)),

  teamSeries: (team: string, league: LeagueKey) =>
    fetch(`${BASE}/teams/series?${q(league, { team })}`).then((r) => j<any>(r)),

  teamsLeague: (season: string, league: LeagueKey) =>
    fetch(`${BASE}/teams/league?${q(league, { season })}`).then((r) => j<any>(r)),

  teamFactors: (team: string, season: string, league: LeagueKey) =>
    fetch(`${BASE}/teams/factors?${q(league, { team, season })}`).then((r) => j<any>(r)),

  shots: (player: string, season: string, mode: "scatter" | "hex", league: LeagueKey) =>
    fetch(`${BASE}/shots?${q(league, { player, season, mode })}`).then((r) => j<any>(r)),
};
