/** Season labels for display.
 *
 * The API stores and accepts a plain year ("2026"). What that year *means*
 * differs by league, so the backend reports `season_format` in /api/meta and
 * this is the only place the frontend turns a stored label into a shown one.
 *
 * Only ever format at the point of rendering — every value sent back to the
 * API must stay the raw stored label.
 */
export type SeasonFormat = "range" | "year";

export function formatSeason(season: string | number | null | undefined, fmt?: SeasonFormat): string {
  if (season == null || season === "") return "";
  const raw = String(season);
  if (fmt !== "range") return raw;
  const end = Number(raw.slice(0, 4));
  if (!Number.isFinite(end)) return raw;
  return `${end - 1}-${String(end).slice(-2)}`;
}

/** "2003–2026" -> "2002-03 – 2025-26" for a range of stored labels. */
export function formatSeasonRange(from: string, to: string, fmt?: SeasonFormat): string {
  return from === to ? formatSeason(from, fmt) : `${formatSeason(from, fmt)}–${formatSeason(to, fmt)}`;
}
