import { useMemo, useRef, useState } from "react";
import * as Popover from "@radix-ui/react-popover";
import { cn } from "@/lib/cn";

type Props = {
  options: string[];
  value: string[];
  onChange: (v: string[]) => void;
  placeholder?: string;
  max?: number;
  className?: string;
};

export function MultiSelect({ options, value, onChange, placeholder = "Select…", max, className }: Props) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    const pool = qq ? options.filter((o) => o.toLowerCase().includes(qq)) : options;
    return pool.slice(0, 80);
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
            <span key={v} className="chip">
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
          <div className="max-h-64 overflow-auto">
            {filtered.length === 0 && <div className="p-2 text-sm text-mute">No matches</div>}
            {filtered.map((o) => {
              const on = value.includes(o);
              const disabled = !on && max != null && value.length >= max;
              return (
                <button
                  key={o}
                  disabled={disabled}
                  onClick={() => toggle(o)}
                  className={cn(
                    "flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-left text-sm",
                    on ? "bg-accent/15 text-ink" : "hover:bg-border/60",
                    disabled && "opacity-40"
                  )}
                >
                  <span>{o}</span>
                  {on && <span className="text-accent">✓</span>}
                </button>
              );
            })}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
