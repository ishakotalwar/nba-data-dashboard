import { useState } from "react";
import type { Meta } from "@/lib/api";
import { TeamCompare } from "./TeamCompare";
import { Teams } from "./Teams";
import { TeamMatchup } from "./TeamMatchup";
import { TeamRankings } from "./TeamRankings";
import { cn } from "@/lib/cn";

const VIEWS = [
  { v: "league", label: "League table" },
  { v: "team", label: "Team profile" },
  { v: "matchup", label: "Matchup" },
  { v: "leaders", label: "All-time leaders" },
] as const;

/**
 * One Teams destination. The league-wide table and the single-team profile are
 * two views of the same subject, so they switch inside the page rather than
 * costing another top-level tab.
 */
export function TeamsSection({ meta }: { meta: Meta }) {
  const [view, setView] = useState<string>("league");
  return (
    <div className="space-y-4">
      <div className="flex gap-5 border-b border-border">
        {VIEWS.map((v) => (
          <button
            key={v.v}
            onClick={() => setView(v.v)}
            className={cn(
              "-mb-px border-b-2 pb-2 text-sm transition",
              v.v === view
                ? "border-accent text-ink"
                : "border-transparent text-mute hover:text-ink"
            )}
          >
            {v.label}
          </button>
        ))}
      </div>
      {view === "league" && <TeamCompare meta={meta} />}
      {view === "team" && <Teams meta={meta} />}
      {view === "matchup" && <TeamMatchup meta={meta} />}
      {view === "leaders" && <TeamRankings meta={meta} />}
    </div>
  );
}
