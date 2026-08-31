import type { Meta } from "@/lib/api";
import { formatSeason } from "@/lib/season";

/**
 * The first thing the app shows: which of the two halves you want.
 *
 * Both are always one click away afterwards — the header keeps a switch — so
 * this is a starting point rather than a gate.
 */
export function Landing({
  meta,
  onPick,
}: {
  meta: Meta;
  onPick: (mode: "stats" | "predictions") => void;
}) {
  const span =
    meta.seasons.length > 1
      ? `${formatSeason(meta.seasons[0], meta.season_format)}–${formatSeason(
          meta.seasons[meta.seasons.length - 1],
          meta.season_format,
        )}`
      : formatSeason(meta.seasons[0], meta.season_format);

  return (
    <div className="mx-auto max-w-4xl px-6 py-16">
      <div className="mb-10 text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-ink">Full Court</h1>
        <p className="mt-2 text-sm text-mute">
          NBA and WNBA analytics · {span} · {meta.players.length.toLocaleString()} players
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Choice
          title="Explore stats"
          blurb="What already happened."
          points={[
            "Player-seasons, percentiles and league ranks",
            "Historical similarity and side-by-side comparison",
            "Shot charts by zone against the league",
            "A filter stack over every season on record",
          ]}
          cta="Explore stats"
          onClick={() => onPick("stats")}
        />
        <Choice
          title="Explore predictions"
          blurb="What happens next."
          points={[
            "Team power ratings from an Elo model",
            "Win probability for any matchup",
            "Projected player lines for next season",
            "Each model shown against its own backtest",
          ]}
          cta="Explore predictions"
          onClick={() => onPick("predictions")}
          accent
        />
      </div>

      <p className="mt-8 text-center text-xs text-mute">
        You can switch between the two at any time from the header.
      </p>
    </div>
  );
}

function Choice({
  title,
  blurb,
  points,
  cta,
  onClick,
  accent = false,
}: {
  title: string;
  blurb: string;
  points: string[];
  cta: string;
  onClick: () => void;
  accent?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="card group flex flex-col items-start p-6 text-left transition hover:border-accent"
    >
      <div className="text-lg font-semibold text-ink">{title}</div>
      <div className="mt-0.5 text-sm text-mute">{blurb}</div>
      <ul className="mt-4 flex-1 space-y-1.5 text-sm text-mute">
        {points.map((p) => (
          <li key={p} className="flex gap-2">
            <span aria-hidden className="text-accent">
              ·
            </span>
            {p}
          </li>
        ))}
      </ul>
      <span
        className={
          accent
            ? "btn btn-primary mt-6 w-full"
            : "btn btn-ghost mt-6 w-full group-hover:border-accent"
        }
      >
        {cta}
      </span>
    </button>
  );
}
