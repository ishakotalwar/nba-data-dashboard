import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

type Props<T> = {
  items: T[];
  rowHeight: number;
  maxHeight: number;
  overscan?: number;
  renderRow: (item: T, index: number) => ReactNode;
  empty?: ReactNode;
};

/**
 * Minimal windowed list: renders only the rows in view (plus a little overscan)
 * so a picker can offer every player without mounting thousands of nodes — and
 * so only visible headshots are ever requested.
 */
export function VirtualList<T>({
  items,
  rowHeight,
  maxHeight,
  overscan = 6,
  renderRow,
  empty,
}: Props<T>) {
  const ref = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewport, setViewport] = useState(maxHeight);

  // The popover animates open, so measure after layout rather than trusting maxHeight.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setViewport(el.clientHeight || maxHeight);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [maxHeight]);

  // Filtering rebuilds the list, so return to the top rather than stranding the
  // viewport in the middle of a now-shorter list.
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = 0;
    setScrollTop(0);
  }, [items]);

  if (items.length === 0) return <>{empty}</>;

  const total = items.length * rowHeight;
  // Clamp: a shrinking list (e.g. the filter narrowed) leaves scrollTop past the
  // new end, and the browser's own clamp fires no scroll event to correct it.
  const maxStart = Math.max(0, Math.ceil((total - viewport) / rowHeight));
  const start = Math.min(
    maxStart,
    Math.max(0, Math.floor(scrollTop / rowHeight) - overscan)
  );
  const visible = Math.ceil(viewport / rowHeight) + overscan * 2;
  const end = Math.min(items.length, start + visible);

  return (
    <div
      ref={ref}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
      style={{ maxHeight }}
      className="overflow-auto"
    >
      <div style={{ height: total, position: "relative" }}>
        <div style={{ transform: `translateY(${start * rowHeight}px)` }}>
          {items.slice(start, end).map((item, i) => renderRow(item, start + i))}
        </div>
      </div>
    </div>
  );
}
