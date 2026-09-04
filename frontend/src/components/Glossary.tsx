import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/cn";

type Term = { term: string; points: string[] };
type Section = { heading: string; terms: Term[] };

/**
 * What every number on the site means, and — where it matters — how it was
 * arrived at. Several of these are this app's own construction rather than a
 * published metric, and the entries say so: a reader deserves to know when a
 * number is Hollinger's and when it is ours.
 */
const GLOSSARY: Section[] = [
  {
    heading: "Rates and efficiency",
    terms: [
      {
        term: "Possession",
        points: [
          "A team's trip down the floor.",
          "Nobody publishes a count, so it is estimated: field goal attempts, plus 0.44 free throw attempts, less offensive rebounds, plus turnovers.",
          "Averaged with the opponent's count of the same trips.",
        ],
      },
      { term: "ORtg · Offensive rating", points: ["Points scored per 100 possessions."] },
      {
        term: "DRtg · Defensive rating",
        points: ["Points allowed per 100 possessions.", "Lower is better."],
      },
      {
        term: "Net rating",
        points: ["Offensive rating minus defensive rating.", "The margin per 100 possessions."],
      },
      { term: "Pace", points: ["Possessions a team uses per game."] },
      {
        term: "TS% · True shooting",
        points: [
          "Points per shooting possession used, with free throws counted at 0.44 of one.",
          "Credits three-pointers and free throws, which field goal percentage does not.",
        ],
      },
      {
        term: "USG% · Usage",
        points: [
          "The share of a team's plays a player finishes while on the floor.",
          "Shots, trips to the line, and turnovers.",
        ],
      },
      {
        term: "eFG%",
        points: ["Field goal percentage with a three counted as one and a half twos."],
      },
      {
        term: "TOV% · ORB% · FT rate",
        points: [
          "Turnovers per possession.",
          "The share of available offensive rebounds taken.",
          "Free throws earned per field goal attempt.",
        ],
      },
    ],
  },
  {
    heading: "Impact metrics",
    terms: [
      {
        term: "RAPM",
        points: [
          "Regularized adjusted plus-minus.",
          "Every stint becomes two observations, one per direction of play.",
          "The five attacking carry +1 in their offensive column, the five defending −1 in their defensive column, against points scored per 100 possessions.",
          "Ridged toward zero, so a player with few possessions lands near average.",
          "The result is the margin per 100 a player is responsible for with the other nine held constant.",
        ],
      },
      {
        term: "RAPM · 3yr",
        points: [
          "The same fit over a rolling three seasons.",
          "One season of stints repeats year to year at about r = 0.40, which is weak.",
          "Steadier, at the cost of describing this season in particular.",
        ],
      },
      {
        term: "Box-prior RAPM",
        points: [
          "The same fit shrunk toward what the box score predicts instead of toward average.",
          "The box-score model is itself fitted against plain RAPM.",
          "A thin sample starts somewhere defensible rather than at zero.",
        ],
      },
      {
        term: "Offense · Defense",
        points: [
          "The two halves a RAPM fit splits each possession into.",
          "Both read the same way: positive is good, whether from scoring more or allowing less.",
          "They add to the total exactly.",
        ],
      },
      {
        term: "Shot value · Turnover value · Second chance",
        points: [
          "What an offensive or defensive number is made of.",
          "A possession is a shot unless it was turned over or repeated after an offensive rebound.",
          "Shot value splits again into field goals and free throws.",
          "Every level adds to the one above it exactly.",
        ],
      },
      {
        term: "PER",
        points: [
          "Hollinger's Player Efficiency Rating.",
          "A per-minute box-score rating, corrected for team pace.",
          "Scaled so the league average is 15.",
          "Sees no possessions and no defence beyond steals and blocks.",
        ],
      },
      {
        term: "On/off",
        points: [
          "Team net rating with a player on the floor minus the same without them.",
          "Unadjusted, so it carries their teammates and opponents with it.",
          "A team that collapses without a player may only be telling you about their backup.",
        ],
      },
      {
        term: "Load",
        points: [
          "The share of a team's possessions a player was on the floor for.",
          "Five players are out there at once, so a player who never sits reaches 100%.",
        ],
      },
      {
        term: "Per 100 · Total",
        points: [
          "Whether an impact number is a rate, or that rate over the possessions actually played.",
          "Total rewards the durability the rate deliberately ignores.",
        ],
      },
    ],
  },
  {
    heading: "Lineups and WOWY",
    terms: [
      {
        term: "Stint",
        points: [
          "A stretch of play with no substitutions.",
          "ESPN publishes no lineup data, but it does publish substitutions, so who was on the floor is rebuilt by walking them.",
        ],
      },
      {
        term: "Lineup",
        points: [
          "A five-player unit.",
          "Every five a team used is stored, including the ones that played a single dead stretch.",
          "Each possession belongs to exactly one of them.",
        ],
      },
      {
        term: "WOWY",
        points: [
          "With or without you.",
          "What a team did with a group of players on the floor, and what it did without them.",
          "Split into every combination, with no adjustment for teammates or opponents.",
        ],
      },
      {
        term: "Share",
        points: ["The portion of a team's floor time a given five was out there for."],
      },
    ],
  },
  {
    heading: "Shooting",
    terms: [
      {
        term: "Shot zones",
        points: [
          "Rim (within four feet), paint, short midrange and long midrange.",
          "Threes split into corners, wings and the top of the arc.",
          "Boundaries follow each league's own three-point geometry, so the WNBA's shorter line is handled properly.",
        ],
      },
      {
        term: "League comparison",
        points: [
          "A player's percentage in a zone against every shot taken there that season.",
          "Drawn from the same kind of game, so playoff shooting is measured against playoff defence.",
        ],
      },
      {
        term: "Regular season · Playoffs · Both",
        points: [
          "Which games a shot chart draws from.",
          "Everything else on the site is regular season only.",
        ],
      },
    ],
  },
  {
    heading: "Predictions",
    terms: [
      {
        term: "Elo",
        points: [
          "A team power rating that moves after each game by how surprising the result was.",
          "Drives the win probabilities on the game calendar.",
        ],
      },
      {
        term: "Projected lines",
        points: [
          "Per-player projections refit to a real game.",
          "240 team minutes, 200 in the WNBA, split by what each player earned.",
          "Totals pulled toward what teams actually score, with anyone ruled out left off.",
        ],
      },
      {
        term: "Similarity",
        points: [
          "How close two player-seasons are across a weighted set of stats.",
          "Scores cluster in the high 90s, so trust the order rather than the number.",
        ],
      },
    ],
  },
  {
    heading: "Conventions",
    terms: [
      {
        term: "Season labels",
        points: [
          "NBA seasons are labelled by the year they end, WNBA seasons by their own year.",
          "2024 means 2023-24 in one and 2024 in the other.",
        ],
      },
      {
        term: "Percentile",
        points: [
          "Where a player sits among everyone who cleared the same qualifying bar on the same page.",
          "Change the bar and the percentiles move with it.",
        ],
      },
      {
        term: "Data source",
        points: [
          "ESPN, via hoopR for the NBA and wehoop for the WNBA, from 2003 on.",
          "Impact metrics and lineups start later, where the substitution feed gets dense enough to rebuild a five from.",
        ],
      },
    ],
  },
];

export function Glossary() {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<string[]>([]);

  // Escape closes, and the page behind should not scroll under the panel.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [open]);

  const toggle = (key: string) =>
    setExpanded((current) =>
      current.includes(key) ? current.filter((k) => k !== key) : [...current, key]
    );

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="What the numbers mean"
        className="border border-border px-2 py-0.5 text-xs text-mute transition hover:border-accent hover:text-ink"
      >
        Glossary
      </button>

      {/* Through a portal: the header this button sits in uses backdrop-blur,
          which makes it a containing block for fixed positioning, so an overlay
          rendered in place would anchor to the header rather than the page. */}
      {open &&
        createPortal(
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 sm:p-8"
            onClick={() => setOpen(false)}
          >
            {/* The panel is a fixed height with its own scrolling body, rather
                than a tall card scrolled by the backdrop: a header pinned over a
                scrolling page clips whatever passes beneath it. */}
            <div
              className="card flex max-h-[85vh] w-full max-w-2xl flex-col"
              // The panel swallows its own clicks so only the backdrop closes.
              onClick={(e) => e.stopPropagation()}
            >
              <div className="card-header shrink-0">
                <div className="text-xl font-semibold tracking-tight text-ink">Glossary</div>
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  aria-label="Close"
                  className="btn btn-ghost px-2 py-1 text-sm"
                >
                  ✕
                </button>
              </div>
              <div className="card-body min-h-0 flex-1 space-y-5 overflow-y-auto">
                {GLOSSARY.map((section) => (
                  <section key={section.heading}>
                    <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-accent">
                      {section.heading}
                    </h3>
                    <div className="divide-y divide-border/70 border-y border-border/70">
                      {section.terms.map((t) => {
                        const key = `${section.heading}:${t.term}`;
                        const isOpen = expanded.includes(key);
                        return (
                          <div key={key}>
                            <button
                              type="button"
                              onClick={() => toggle(key)}
                              aria-expanded={isOpen}
                              className="flex w-full items-center gap-2.5 py-2 text-left transition hover:text-accent"
                            >
                              <span
                                aria-hidden
                                className={cn(
                                  "grid h-4 w-4 shrink-0 place-items-center border text-[11px] leading-none transition",
                                  isOpen
                                    ? "border-accent bg-accent text-onAccent"
                                    : "border-border text-mute"
                                )}
                              >
                                {isOpen ? "−" : "+"}
                              </span>
                              <span className={cn("text-sm", isOpen && "font-medium")}>
                                {t.term}
                              </span>
                            </button>
                            {isOpen && (
                              <ul className="space-y-1 pb-3 pl-[26px] pr-2">
                                {t.points.map((point) => (
                                  <li key={point} className="flex gap-2 text-sm text-mute">
                                    <span aria-hidden className="text-accent">
                                      ·
                                    </span>
                                    <span className="leading-relaxed">{point}</span>
                                  </li>
                                ))}
                              </ul>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </section>
                ))}
              </div>
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
