import type { Meta } from "@/lib/api";
import { cn } from "@/lib/cn";
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
      <div className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-ink">Full Court</h1>
        <p className="mt-2 text-sm text-mute">
          NBA and WNBA analytics · {span} · {meta.players.length.toLocaleString()} players
        </p>
      </div>

      <Dribbler />

      <div className="grid items-start gap-4 md:grid-cols-2">
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

/**
 * A player mid-dribble, drawn as a flat silhouette: one tapered outline per
 * bone, unioned by sharing a fill. The far arm and leg sit behind a knockout
 * pass in the page colour, which is what keeps them reading as depth instead
 * of merging into the torso. Decorative only.
 */
function Dribbler() {
  return (
    <svg
      viewBox="96 56 250 424"
      role="presentation"
      aria-hidden
      className="mx-auto my-8 h-56 w-auto text-ink"
      fill="none"
    >
      <defs>
        <g id="fc-dribbler">
          <path d="M 262.3 145.3 L 288.9 128.1 L 271.9 103.4 L 246.4 122.2 Z M 287.0 102.7 L 285.5 70.3 L 244.5 86.7 L 265.8 111.2 Z M 268.0 135.1 L 300.9 98.2 L 251.9 75.1 L 244.3 123.9 Z M 276.5 189.9 L 265.6 141.9 L 241.0 139.3 L 220.4 184.0 Z M 225.3 151.6 L 205.5 200.7 L 246.3 224.8 L 279.8 183.8 Z M 201.6 207.5 L 179.8 242.0 L 229.7 266.9 L 244.3 228.8 Z M 186.7 272.3 L 199.4 273.4 L 216.0 232.0 L 206.0 224.0 Z M 264.1 200.7 L 277.7 171.7 L 247.3 144.4 L 220.0 161.0 Z M 281.7 176.1 L 258.8 150.4 L 228.5 163.1 L 230.6 197.4 Z M 226.4 232.8 L 210.0 232.9 L 197.8 269.7 L 210.8 279.6 Z M 247.6 160.3 L 286.1 217.7 L 312.4 203.6 L 286.2 139.7 Z M 285.0 211.7 L 294.0 267.8 L 313.9 266.4 L 314.8 209.6 Z M 294.2 269.8 L 292.2 292.4 L 317.9 291.2 L 314.0 268.9 Z M 196.0 269.2 L 256.1 342.4 L 286.7 319.9 L 234.5 240.8 Z M 253.4 327.9 L 252.2 417.9 L 272.0 420.1 L 291.0 332.2 Z M 256.6 428.4 L 298.2 447.9 L 305.0 435.7 L 266.2 410.9 Z M 254.8 413.1 L 243.5 427.8 L 257.1 439.5 L 269.9 426.1 Z" />
          <circle cx="272" cy="96" r="29" />
          <circle cx="280" cy="116" r="15" />
          <circle cx="254" cy="134" r="14" />
          <circle cx="250" cy="172" r="32" />
          <circle cx="224" cy="216" r="24" />
          <circle cx="206" cy="252" r="28" />
          <circle cx="268" cy="152" r="22" />
          <circle cx="216" cy="256" r="24" />
          <circle cx="300" cy="212" r="15" />
          <circle cx="304" cy="268" r="10" />
          <circle cx="305" cy="290" r="13" />
          <circle cx="272" cy="332" r="19" />
          <circle cx="262" cy="420" r="10" />
          <circle cx="302" cy="442" r="7" />
          <circle cx="250" cy="434" r="9" />
        </g>
      </defs>

      <ellipse cx="210" cy="462" rx="112" ry="10" fill="currentColor" opacity={0.14} />

      <g fill="currentColor">
        {/* far side of the body */}
        <g opacity={0.92}>
          <path d="M 228.2 133.1 L 191.3 175.0 L 210.5 195.2 L 254.3 160.5 Z M 194.4 173.2 L 158.4 199.7 L 167.4 215.2 L 208.3 197.2 Z M 156.3 201.1 L 142.4 204.7 L 152.8 225.7 L 164.1 216.7 Z M 176.8 234.3 L 125.9 310.8 L 155.0 332.0 L 212.3 260.2 Z M 122.0 322.4 L 140.0 410.2 L 159.8 408.0 L 157.6 318.4 Z M 143.7 402.3 L 113.6 434.6 L 123.1 444.8 L 157.3 416.8 Z" />
          <circle cx="240" cy="148" r="19" />
          <circle cx="200" cy="186" r="14" />
          <circle cx="162" cy="208" r="9" />
          <circle cx="150" cy="214" r="12" />
          <circle cx="194" cy="248" r="22" />
          <circle cx="140" cy="322" r="18" />
          <circle cx="150" cy="410" r="10" />
          <circle cx="118" cy="440" r="7" />
        </g>
        {/* knockout, then the near side over it */}
        <use
          href="#fc-dribbler"
          fill="rgb(var(--c-bg))"
          stroke="rgb(var(--c-bg))"
          strokeWidth={11}
          strokeLinejoin="round"
        />
        <use href="#fc-dribbler" fill="currentColor" />
      </g>

      <g className="text-accent" transform="translate(308,344)">
        <circle r="30" fill="currentColor" />
        <g stroke="rgb(var(--c-bg))" strokeWidth={2.4} fill="none">
          <path d="M -30 0 H 30 M 0 -30 V 30" />
          <path d="M 0 -30 C -17 -17 -17 17 0 30 M 0 -30 C 17 -17 17 17 0 30" />
        </g>
      </g>
    </svg>
  );
}
