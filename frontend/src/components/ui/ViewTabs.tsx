import { cn } from "@/lib/cn";

export type View = { v: string; label: string };

/**
 * The second row of navigation, inside a top-level tab. Views of one subject
 * switch here rather than costing another tab up top, so the header stays four
 * destinations wide however many pages hang off each one.
 */
export function ViewTabs({
  views,
  value,
  onChange,
}: {
  views: readonly View[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="no-scrollbar flex gap-5 overflow-x-auto border-b border-border">
      {views.map((view) => (
        <button
          key={view.v}
          onClick={() => onChange(view.v)}
          className={cn(
            "-mb-px whitespace-nowrap border-b-2 pb-2 text-sm transition",
            view.v === value
              ? "border-accent text-ink"
              : "border-transparent text-mute hover:text-ink"
          )}
        >
          {view.label}
        </button>
      ))}
    </div>
  );
}
