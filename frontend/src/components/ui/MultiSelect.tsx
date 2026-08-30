import { useMemo, useRef, useState } from "react";
import * as Popover from "@radix-ui/react-popover";
import { cn } from "@/lib/cn";
import { VirtualList } from "./VirtualList";

const ROW_H = 40;

type Props = {
  options: string[];
  value: string[];
  onChange: (v: string[]) => void;
  placeholder?: string;
  max?: number;
  className?: string;
  /** Render a leading avatar for each option. Omit for non-player lists. */
  renderAvatar?: (name: string, size: number) => React.ReactNode;
};

export function MultiSelect({
  options,
  value,
  onChange,
  placeholder = "Select",
  max,
  className,
  renderAvatar,
}: Props) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // No cap: the list is windowed, so every match stays reachable by scrolling.
  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    return qq ? options.filter((o) => o.toLowerCase().includes(qq)) : options;
  }, [options, q]);

  const toggle = (v: string) => {
    if (value.includes(v)) onChange(value.filter((x) => x !== v));
    else {
      if (max && value.length >= max) return;
      onChange([...value, v]);
    }
  };

  const remove = (v: string) => onChange(value.filter((x) => x !== v));

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          className={cn(
            "flex min-h-[42px] w-full flex-wrap items-center gap-1.5 rounded-lg border border-border bg-bg/60 px-2.5 py-1.5 text-left text-sm",
            "hover:border-accent/40 focus:outline-none focus:ring-2 focus:ring-accent/30",
            className
          )}
        >
          {value.length === 0 && <span className="text-mute">{placeholder}</span>}
          {value.map((v) => (
            <span key={v} className={cn("chip", renderAvatar && "!py-0.5 !pl-0.5")}>
              {renderAvatar?.(v, 22)}
              {v}
              <button
                type="button"
                className="ml-1 text-mute hover:text-ink"
                onClick={(e) => {
                  e.stopPropagation();
                  remove(v);
                }}
              >
                ×
              </button>
            </span>
          ))}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="start"
          sideOffset={6}
          className="z-50 w-[--radix-popover-trigger-width] rounded-lg border border-border bg-panel p-2 shadow-card"
          onOpenAutoFocus={(e) => {
            e.preventDefault();
            inputRef.current?.focus();
          }}
        >
          <input
            ref={inputRef}
            className="input mb-2"
            placeholder="Type to filter…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          <VirtualList
            items={filtered}
            rowHeight={ROW_H}
            maxHeight={288}
            empty={<div className="p-2 text-sm text-mute">No matches</div>}
            renderRow={(o) => {
              const on = value.includes(o);
              const disabled = !on && max != null && value.length >= max;
              return (
                <button
                  key={o}
                  disabled={disabled}
                  onClick={() => toggle(o)}
                  style={{ height: ROW_H }}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2 text-left text-sm",
                    on ? "bg-accent/15 text-ink" : "hover:bg-border/60",
                    disabled && "opacity-40"
                  )}
                >
                  {renderAvatar?.(o, 28)}
                  <span className="truncate">{o}</span>
                  {on && <span className="ml-auto text-accent">✓</span>}
                </button>
              );
            }}
          />
          <div className="mt-1.5 border-t border-border pt-1.5 text-[11px] text-mute">
            {filtered.length.toLocaleString()}
            {filtered.length === 1 ? " match" : " matches"}
            {max ? ` · pick up to ${max}` : ""}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
