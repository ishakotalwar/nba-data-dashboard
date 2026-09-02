import { useEffect, useState } from "react";
import type { Meta } from "@/lib/api";
import { ViewTabs } from "@/components/ui/ViewTabs";
import { TeamCompare } from "./TeamCompare";
import { Teams } from "./Teams";
import { Lineups } from "./Lineups";
import { TeamRankings } from "./TeamRankings";

const VIEWS = [
  { v: "league", label: "League table" },
  { v: "team", label: "Team profile" },
  { v: "lineups", label: "Lineups" },
  { v: "leaders", label: "All-time leaders" },
] as const;

/**
 * One Teams destination: the league-wide table, one team's history, the fives
 * that team put on the floor, and the all-time leaderboards. Four views of the
 * same subject, so they switch inside the page rather than costing four tabs.
 */
export function TeamsSection({ meta, view }: { meta: Meta; view?: string }) {
  const [active, setActive] = useState<string>(view ?? "league");
  useEffect(() => {
    if (view) setActive(view);
  }, [view]);

  return (
    <div className="space-y-4">
      <ViewTabs views={VIEWS} value={active} onChange={setActive} />
      {active === "league" && <TeamCompare meta={meta} />}
      {active === "team" && <Teams meta={meta} />}
      {active === "lineups" && <Lineups meta={meta} />}
      {active === "leaders" && <TeamRankings meta={meta} />}
    </div>
  );
}
