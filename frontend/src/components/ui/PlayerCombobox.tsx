import { useMemo, useRef, useState } from "react";
import * as Popover from "@radix-ui/react-popover";
import { cn } from "@/lib/cn";
import { VirtualList } from "./VirtualList";

const ROW_H = 42;

type Props = {
  options: string[];
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
  /** Render a leading avatar for each option. */
  renderAvatar?: (name: string, size: number) => React.ReactNode;
};

export function PlayerCombobox({
  options,
  value,
  onChange,
  placeholder = "Select",
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

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          className={cn(
           "flex h-[42px] w-full items-center gap-2 border border-border bg-bg px-3 text-left text-sm",
           "hover:border-mute focus:border-accent focus:outline-none",
            className
          )}
        >
          {value && renderAvatar?.(value, 30)}
          <span className={cn("truncate", !value && "text-mute")}>{value || placeholder}</span>
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="start"
          sideOffset={6}
          className="z-50 w-[--radix-popover-trigger-width] border border-border bg-panel p-2"
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
          <VirtualList
            items={filtered}
            rowHeight={ROW_H}
            maxHeight={288}
            empty={<div className="p-2 text-sm text-mute">No matches</div>}
            renderRow={(o) => (
              <button
                key={o}
                onClick={() => {
                  onChange(o);
                  setOpen(false);
                  setQ("");
                }}
                style={{ height: ROW_H }}
                className={cn(
                 "flex w-full items-center gap-2 px-2 text-left text-sm",
                  value === o ? "bg-border text-ink" : "hover:bg-border/60"
                )}
              >
                {renderAvatar?.(o, 30)}
                <span className="truncate">{o}</span>
                {value === o && <span className="ml-auto text-accent">✓</span>}
              </button>
            )}
          />
          <div className="mt-1.5 border-t border-border pt-1.5 text-[11px] text-mute">
            {filtered.length.toLocaleString()}
            {filtered.length === 1 ? " player" : " players"}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
