import type { ReactNode } from "react";
import { traceColor } from "./Plot";

type Props = {
  /** Player names in the same order as the chart's traces. */
  names: string[];
  renderAvatar: (name: string, size: number) => ReactNode;
  size?: number;
  className?: string;
};

/**
 * Legend showing each player's face beside their trace color. Replaces
 * Plotly's own legend on charts whose traces map one-to-one to players, so the
 * two never disagree — colors come from the same `traceColor` the chart uses.
 */
export function PlayerLegend({ names, renderAvatar, size = 30, className }: Props) {
  if (names.length === 0) return null;
  return (
    <div className={"flex flex-wrap items-center gap-x-5 gap-y-2 " + (className ?? "")}>
      {names.map((name, i) => (
        <span key={name} className="flex items-center gap-2 text-sm text-ink">
          <span className="relative">
            {renderAvatar(name, size)}
            <span
              aria-hidden
              className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 ring-2 ring-panel"
              style={{ background: traceColor(i) }}
            />
          </span>
          <span className="truncate">{name}</span>
        </span>
      ))}
    </div>
  );
}
