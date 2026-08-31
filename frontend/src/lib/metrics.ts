/**
 * One source of truth for metric labels, formatting and semantics.
 * Panels must not hardcode these — Compare, Percentiles, Similarity, Explorer
 * and the player overview all read from here.
 */
export type MetricCategory = "scoring" | "shooting" | "playmaking" | "rebounding" | "defense" | "efficiency" | "usage";

export type MetricFormat = "decimal" | "percent" | "rating" | "integer";

export type MetricDef = {
  key: string;
  label: string;
  shortLabel: string;
  format: MetricFormat;
  category: MetricCategory;
  higherIsBetter: boolean;
  /** Longer explanation, for tooltips and table headers. */
  description?: string;
};

export const METRICS: MetricDef[] = [
  { key: "pts", label: "Points", shortLabel: "PTS", format: "decimal", category: "scoring", higherIsBetter: true, description: "Points per game" },
  { key: "reb", label: "Rebounds", shortLabel: "REB", format: "decimal", category: "rebounding", higherIsBetter: true, description: "Rebounds per game" },
  { key: "ast", label: "Assists", shortLabel: "AST", format: "decimal", category: "playmaking", higherIsBetter: true, description: "Assists per game" },
  { key: "stl", label: "Steals", shortLabel: "STL", format: "decimal", category: "defense", higherIsBetter: true, description: "Steals per game" },
  { key: "blk", label: "Blocks", shortLabel: "BLK", format: "decimal", category: "defense", higherIsBetter: true, description: "Blocks per game" },
  { key: "tov", label: "Turnovers", shortLabel: "TOV", format: "decimal", category: "playmaking", higherIsBetter: false, description: "Turnovers per game — fewer is better" },
  { key: "min", label: "Minutes", shortLabel: "MIN", format: "decimal", category: "usage", higherIsBetter: true, description: "Minutes per game" },
  { key: "gp", label: "Games played", shortLabel: "GP", format: "integer", category: "usage", higherIsBetter: true },
  { key: "fg_pct", label: "Field goal %", shortLabel: "FG%", format: "percent", category: "shooting", higherIsBetter: true },
  { key: "three_pct", label: "Three-point %", shortLabel: "3P%", format: "percent", category: "shooting", higherIsBetter: true },
  { key: "ft_pct", label: "Free throw %", shortLabel: "FT%", format: "percent", category: "shooting", higherIsBetter: true },
  { key: "ts_pct", label: "True shooting %", shortLabel: "TS%", format: "percent", category: "efficiency", higherIsBetter: true, description: "Points per shooting possession, counting threes and free throws" },
  { key: "usg_pct", label: "Usage rate", shortLabel: "USG%", format: "percent", category: "usage", higherIsBetter: true, description: "Share of team plays a player uses while on the floor" },
  { key: "ortg", label: "Offensive rating", shortLabel: "ORtg", format: "rating", category: "efficiency", higherIsBetter: true, description: "Points produced per 100 possessions" },
  { key: "drtg", label: "Defensive rating", shortLabel: "DRtg", format: "rating", category: "defense", higherIsBetter: false, description: "Points allowed per 100 possessions — lower is better" },
  { key: "pace", label: "Pace", shortLabel: "Pace", format: "rating", category: "usage", higherIsBetter: true, description: "Possessions per game" },
];

const BY_KEY = new Map(METRICS.map((m) => [m.key, m]));

/** Falls back to a generated definition so an unknown column still renders. */
export function metric(key: string): MetricDef {
  return (
    BY_KEY.get(key) ?? {
      key,
      label: key,
      shortLabel: key.toUpperCase(),
      format: "decimal",
      category: "scoring",
      higherIsBetter: true,
    }
  );
}

export const label = (key: string) => metric(key).label;
export const shortLabel = (key: string) => metric(key).shortLabel;
export const higherIsBetter = (key: string) => metric(key).higherIsBetter;

export function formatValue(key: string, v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  switch (metric(key).format) {
    case "percent":
      // Stored as a 0–1 rate; shown as a percentage.
      return `${(v * 100).toFixed(1)}%`;
    case "rating":
      return v.toFixed(1);
    case "integer":
      return String(Math.round(v));
    default:
      return v.toFixed(1);
  }
}

/** Signed difference, formatted like the metric itself.
 *  A percent metric reports the gap in percentage points. */
export function formatDelta(key: string, v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "";
  const sign = v > 0 ? "+" : v < 0 ? "−" : "";
  const mag = Math.abs(v);
  switch (metric(key).format) {
    case "percent":
      return `${sign}${(mag * 100).toFixed(1)}pp`;
    case "integer":
      return `${sign}${Math.round(mag)}`;
    default:
      return `${sign}${mag.toFixed(1)}`;
  }
}

/** "92nd", "1st", "23rd" — for league ranks and percentiles. */
export function ordinal(n: number): string {
  const v = Math.round(n);
  const s = ["th", "st", "nd", "rd"];
  const k = v % 100;
  return v + (s[(k - 20) % 10] || s[k] || s[0]);
}

/** Order metrics for display: scoring first, then the rest by category. */
const CATEGORY_ORDER: MetricCategory[] = [
  "scoring", "shooting", "efficiency", "playmaking", "rebounding", "defense", "usage",
];

export function sortMetrics(keys: string[]): string[] {
  return [...keys].sort((a, b) => {
    const ca = CATEGORY_ORDER.indexOf(metric(a).category);
    const cb = CATEGORY_ORDER.indexOf(metric(b).category);
    if (ca !== cb) return ca - cb;
    return METRICS.findIndex((m) => m.key === a) - METRICS.findIndex((m) => m.key === b);
  });
}

/** Weight presets for similarity (agents.md §4). */
export const SIMILARITY_PRESETS: Record<string, Record<string, number>> = {
  Overall: {},
  Scoring: { pts: 2.5, usg_pct: 1.8, ts_pct: 1.5, ast: 0.5, reb: 0.5, tov: 0.5, ortg: 1, drtg: 0.5 },
  Shooting: { ts_pct: 2.5, pts: 1.2, usg_pct: 0.8, ortg: 1.5, ast: 0.4, reb: 0.4, tov: 0.4, drtg: 0.4 },
  Playmaking: { ast: 2.5, tov: 1.8, usg_pct: 1.2, pts: 0.8, ortg: 1, reb: 0.4, ts_pct: 0.6, drtg: 0.4 },
  Defense: { drtg: 2.5, reb: 1.5, pts: 0.4, ast: 0.4, tov: 0.6, ts_pct: 0.3, usg_pct: 0.3, ortg: 0.4 },
};
