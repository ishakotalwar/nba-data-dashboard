import { LinePlayer } from "@/components/LinePlayer";
import { cn } from "@/lib/cn";

/**
 * The first thing the app shows: which of the two halves you want.
 *
 * Both are always one click away afterwards — the header keeps a switch — so
 * this is a starting point rather than a gate.
 */
export function Landing({ onPick }: { onPick: (mode: "stats" | "predictions") => void }) {
  return (
    <div className="mx-auto max-w-4xl px-6 py-16">
      <div className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-ink">Full Court</h1>
        <p className="mt-2 text-sm text-mute">
          NBA and WNBA analytics, 2003 to now
        </p>
      </div>

      <LinePlayer />

      <div className="grid items-start gap-4 md:grid-cols-2">
        <Choice
          title="Explore stats"
          blurb="What already happened."
          points={[
            "Player pages with percentiles, per 36, 75 or 100 possessions",
            "Five ways to rank impact, rebuilt from every substitution",
            "Five-man lineups, and how a team played without any of them",
            "Shot charts by zone, regular season or playoffs",
            "Season similarity and side-by-side comparison",
            "A filter stack over every player-season on record",
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
        />
      </div>
    </div>
  );
}

function Choice({
  title,
  blurb,
  points,
  cta,
  onClick,
}: {
  title: string;
  blurb: string;
  points: string[];
  cta: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "card group relative z-0 flex w-full flex-col items-start p-6 text-left",
        "transition-all duration-200 ease-out will-change-transform",
        // The card lifts off the page on hover, and the same treatment is
        // mirrored on keyboard focus so both routes in feel identical.
        "hover:z-10 hover:-translate-y-1.5 hover:scale-[1.1] hover:border-accent hover:shadow-2xl hover:shadow-black/40",
        "focus-visible:z-10 focus-visible:-translate-y-1.5 focus-visible:scale-[1.1] focus-visible:border-accent focus-visible:shadow-2xl",
      )}
    >
      <div className="text-lg font-semibold text-ink">{title}</div>
      <div className="mt-0.5 text-sm text-mute">{blurb}</div>

      {/* The detail is held back until the card is hovered or focused. The
          0fr -> 1fr grid row animates to the content's own height, so the card
          grows to fit rather than to a guessed pixel value. Focus is included
          so the list is still reachable from the keyboard. */}
      <div
        className="grid w-full grid-rows-[0fr] transition-[grid-template-rows] duration-300 ease-out group-hover:grid-rows-[1fr] group-focus-visible:grid-rows-[1fr]"
      >
        <ul className="min-h-0 overflow-hidden text-sm text-mute">
          {points.map((p, i) => (
            <li key={p} className={cn("flex gap-2", i === 0 ? "pt-4" : "pt-1.5")}>
              <span aria-hidden className="text-accent">
                ·
              </span>
              {p}
            </li>
          ))}
        </ul>
      </div>
      <span className="btn btn-primary mt-6 w-full">{cta}</span>
    </button>
  );
}


