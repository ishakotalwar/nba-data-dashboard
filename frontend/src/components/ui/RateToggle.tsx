import { cn } from "@/lib/cn";
import type { Meta } from "@/lib/api";

/** Shorter labels than the API's, for a control that sits inline in a header. */
const SHORT: Record<string, string> = {
  game: "Per game",
  per36: "Per 36",
  per75: "Per 75",
  per100: "Per 100",
};

const TITLE: Record<string, string> = {
  game: "Counting stats as a per-game average",
  per36: "Counting stats scaled to 36 minutes of playing time",
  per75:
    "Counting stats per 75 possessions — roughly what a starter uses in a " +
    "game, so the numbers stay close to the per-game ones they replace",
  per100:
    "Counting stats per 100 possessions the player's team used while they " +
    "were on the floor — their minutes times their team's pace, since ESPN " +
    "publishes no possession data",
};

/**
 * Picks the basis counting stats are expressed on. It changes percentiles,
 * ranks and filters too, not just the number printed, because a player's
 * standing depends on what everyone else is measured by.
 */
export function RateToggle({
  meta,
  value,
  onChange,
  className,
}: {
  meta: Meta;
  value: string;
  onChange: (v: string) => void;
  className?: string;
}) {
  const bases = Object.keys(meta.rate_bases ?? { game: "Per game" });
  if (bases.length < 2) return null;
  return (
    <div className={cn("flex items-center gap-4", className)}>
      {bases.map((b) => (
        <button
          key={b}
          type="button"
          onClick={() => onChange(b)}
          title={TITLE[b] ?? meta.rate_bases[b]}
          className={cn(
            "border-b-2 pb-0.5 text-sm transition",
            b === value
              ? "border-accent text-ink"
              : "border-transparent text-mute hover:text-ink"
          )}
        >
          {SHORT[b] ?? meta.rate_bases[b]}
        </button>
      ))}
    </div>
  );
}
