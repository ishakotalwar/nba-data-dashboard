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
    <div className="no-scrollbar flex gap-2 overflow-x-auto pb-1">
      {views.map((view) => (
        <button
          key={view.v}
          onClick={() => onChange(view.v)}
          className={cn(
            "whitespace-nowrap border px-2.5 py-1 text-xs font-medium uppercase tracking-wider transition",
            view.v === value
              ? "border-accent2 bg-accent2/10 text-ink"
              : "border-transparent text-mute hover:border-border hover:text-ink"
          )}
        >
          {view.label}
        </button>
      ))}
    </div>
  );
}
