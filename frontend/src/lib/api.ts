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

export type Meta = {
  players: string[];
  teams: string[];
  seasons: string[];
  metrics: string[];
  invert_metrics: string[];
};

export const api = {
  meta: () => fetch(`${BASE}/meta`).then((r) => j<Meta>(r)),

  compare: (params: {
    players: string[];
    metrics: string[];
    seasonLo?: string;
    seasonHi?: string;
  }) => {
    const q = new URLSearchParams({
      players: params.players.join(","),
      metrics: params.metrics.join(","),
    });
    if (params.seasonLo) q.set("seasonLo", params.seasonLo);
    if (params.seasonHi) q.set("seasonHi", params.seasonHi);
    return fetch(`${BASE}/compare?${q}`).then((r) => j<any>(r));
  },

  trends: (params: { player: string; metrics: string[]; league?: boolean }) => {
    const q = new URLSearchParams({
      player: params.player,
      metrics: params.metrics.join(","),
      league: String(params.league ?? true),
    });
    return fetch(`${BASE}/trends?${q}`).then((r) => j<any>(r));
  },

  percentiles: (player: string, season: string) =>
    fetch(`${BASE}/percentiles?player=${encodeURIComponent(player)}&season=${encodeURIComponent(season)}`)
      .then((r) => j<any>(r)),

  similar: (body: {
    anchor: string;
    k: number;
    weights: Record<string, number>;
    seasonLo?: string;
    seasonHi?: string;
  }) =>
    fetch(`${BASE}/similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j<any>(r)),

  gamelog: (player: string, season: string, stat: string, window: number) =>
    fetch(
      `${BASE}/gamelog?player=${encodeURIComponent(player)}&season=${encodeURIComponent(
        season
      )}&stat=${stat}&window=${window}`
    ).then((r) => j<any>(r)),

  ageCurves: (body: { players: string[]; metric: string }) =>
    fetch(`${BASE}/age-curves`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j<any>(r)),

  teamSeries: (team: string) =>
    fetch(`${BASE}/teams/series?team=${encodeURIComponent(team)}`).then((r) => j<any>(r)),

  teamFactors: (team: string, season: string) =>
    fetch(
      `${BASE}/teams/factors?team=${encodeURIComponent(team)}&season=${encodeURIComponent(season)}`
    ).then((r) => j<any>(r)),

  shots: (player: string, season: string, mode: "scatter" | "hex") =>
    fetch(
      `${BASE}/shots?player=${encodeURIComponent(player)}&season=${encodeURIComponent(
        season
      )}&mode=${mode}`
    ).then((r) => j<any>(r)),
};
