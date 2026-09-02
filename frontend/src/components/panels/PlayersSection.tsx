import { useEffect, useState } from "react";
import type { Meta } from "@/lib/api";
import { ViewTabs } from "@/components/ui/ViewTabs";
import { Players } from "./Players";
import { Similar } from "./Similar";
import { ShotAnalysis } from "./ShotAnalysis";

const VIEWS = [
  { v: "overview", label: "Overview" },
  { v: "similar", label: "Similarity" },
  { v: "shots", label: "Shot Analysis" },
] as const;

/** Everything about one player: the profile, who else played like him, and
 *  where he shot from. */
export function PlayersSection({
  meta,
  view,
  seedFor,
}: {
  meta: Meta;
  /** Which view to open on, when Ask Full Court points at one of them. */
  view?: string;
  seedFor: (page: string) => any;
}) {
  const [active, setActive] = useState<string>(view ?? "overview");
  useEffect(() => {
    if (view) setActive(view);
  }, [view]);

  return (
    <div className="space-y-4">
      <ViewTabs views={VIEWS} value={active} onChange={setActive} />
      {active === "overview" && <Players meta={meta} />}
      {active === "similar" && <Similar meta={meta} seed={seedFor("similarity")} />}
      {active === "shots" && <ShotAnalysis meta={meta} seed={seedFor("shots")} />}
    </div>
  );
}
