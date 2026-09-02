import { useEffect, useState } from "react";
import type { Meta } from "@/lib/api";
import { ViewTabs } from "@/components/ui/ViewTabs";
import { Compare } from "./Compare";
import { TeamMatchup } from "./TeamMatchup";

const VIEWS = [
  { v: "players", label: "Players" },
  { v: "teams", label: "Teams" },
] as const;

/** Head-to-head, for either subject. Comparing two players and comparing two
 *  team-seasons are the same question asked of different rows, so they live
 *  together rather than one on each half of the app. */
export function CompareSection({
  meta,
  view,
  seedFor,
}: {
  meta: Meta;
  view?: string;
  seedFor: (page: string) => any;
}) {
  const [active, setActive] = useState<string>(view ?? "players");
  useEffect(() => {
    if (view) setActive(view);
  }, [view]);

  return (
    <div className="space-y-4">
      <ViewTabs views={VIEWS} value={active} onChange={setActive} />
      {active === "players" && <Compare meta={meta} seed={seedFor("compare")} />}
      {active === "teams" && <TeamMatchup meta={meta} />}
    </div>
  );
}
