import type { LeagueKey } from "./api";

/**
 * ESPN's headshot CDN, keyed by the same athlete id sportsdataverse stores as
 * `player_id`. The `combiner` endpoint resizes server-side — a full-size PNG is
 * ~270 KB, the same image at 120px is ~16 KB.
 *
 * Not every player has one: coverage is ~100% from 2015 on but only ~50% in the
 * mid-2000s, and missing ids return 404. Callers must handle that (see Avatar).
 */
export function headshotUrl(
  id: number | undefined,
  league: LeagueKey,
  size = 64
): string | null {
  if (id == null) return null;
  const px = Math.round(size * 2); // 2x for retina
  return (
    `https://a.espncdn.com/combiner/i?img=/i/headshots/${league}/players/full/${id}.png` +
    `&w=${px}&h=${px}&scale=crop`
  );
}

/** "A'ja Wilson" -> "AW". Fallback when there's no headshot. */
export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}
