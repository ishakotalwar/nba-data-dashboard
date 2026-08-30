import { useMemo, useRef, useState } from "react";
import * as Popover from "@radix-ui/react-popover";
import { cn } from "@/lib/cn";

type Props = {
  options: string[];
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
};

export function PlayerCombobox({ options, value, onChange, placeholder = "Search player…", className }: Props) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    if (!qq) return options.slice(0, 50);
    return options.filter((o) => o.toLowerCase().includes(qq)).slice(0, 50);
  }, [options, q]);

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          className={cn(
            "flex h-[42px] w-full items-center rounded-lg border border-border bg-bg/60 px-3 text-left text-sm",
            "hover:border-accent/40 focus:outline-none focus:ring-2 focus:ring-accent/30",
            className
          )}
        >
          <span className={cn("truncate", !value && "text-mute")}>{value || placeholder}</span>
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
            placeholder="Type a player name…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          <div className="max-h-64 overflow-auto">
            {filtered.length === 0 && <div className="p-2 text-sm text-mute">No matches</div>}
            {filtered.map((o) => (
              <button
                key={o}
                onClick={() => {
                  onChange(o);
                  setOpen(false);
                  setQ("");
                }}
                className={cn(
                  "flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-left text-sm",
                  value === o ? "bg-accent/15 text-ink" : "hover:bg-border/60"
                )}
              >
                <span>{o}</span>
                {value === o && <span className="text-accent">✓</span>}
              </button>
            ))}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
