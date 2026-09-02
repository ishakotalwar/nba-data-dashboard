import { useCallback, useEffect, useRef, useState } from "react";

import { Avatar } from "@/components/ui/Avatar";
import { api } from "@/lib/api";
import type { LeagueKey, Meta } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatValue, label as metricLabel } from "@/lib/metrics";
import { formatSeason } from "@/lib/season";

type Size = "sm" | "md" | "lg";

const SIZES: Record<Size, string> = {
  sm: "w-[360px] h-[420px]",
  md: "w-[520px] h-[560px]",
  lg: "w-[760px] h-[calc(100vh-6rem)]",
};

const NEXT_SIZE: Record<Size, Size> = { sm: "md", md: "lg", lg: "sm" };
const SIZE_KEY = "full-court-ask-size";

type AskResult = {
  status: "ok" | "needs_clarification" | "unsupported";
  summary?: string;
  intent?: string;
  results?: any[];
  columns?: string[];
  options?: { player_name: string; seasons: string[] }[];
  navigate?: { page: string; state: any };
  target_page?: string;
  total?: number;
  metric?: string;
  parser?: string;
  /** The answer's own league format — an NBA question can return WNBA rows. */
  season_format?: "range" | "year";
};

/**
 * The natural-language entry point, available on every page.
 *
 * It never renders a statistic it was handed by a model: everything shown here
 * comes from /api/ask, which parses the question into a structured query and
 * executes it against local Parquet.
 */
export function AskFullCourt({
  meta,
  onNavigate,
}: {
  meta: Meta;
  /** Called with the API's page name; App.tsx routes it to a tab and view. */
  onNavigate?: (page: string, navigate?: { page: string; state: any }) => void;
}) {
  const [open, setOpen] = useState(false);
  const [size, setSize] = useState<Size>(() => {
    try {
      const saved = localStorage.getItem(SIZE_KEY);
      if (saved === "sm" || saved === "md" || saved === "lg") return saved;
    } catch {
      /* storage blocked — the default is fine */
    }
    return "md";
  });
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [examples, setExamples] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(SIZE_KEY, size);
    } catch {
      /* not worth surfacing */
    }
  }, [size]);

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
    if (examples.length === 0) {
      api.askCapabilities(meta.league).then((d) => setExamples(d.examples ?? [])).catch(() => {});
    }
  }, [open, meta.league, examples.length]);

  // Cmd/Ctrl-K from anywhere, Escape to dismiss.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const run = useCallback(
    (q: string) => {
      const text = q.trim();
      if (!text) return;
      setLoading(true);
      setErr(null);
      api
        .ask(text, meta.league)
        .then((d) => setResult(d))
        .catch((e) => setErr(e?.message ?? "Something went wrong."))
        .finally(() => setLoading(false));
    },
    [meta.league],
  );

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="Ask Full Court  (⌘K)"
        className="btn btn-primary fixed bottom-5 right-5 z-40 rounded-full px-4 py-3 shadow-lg"
      >
        Ask
      </button>
    );
  }

  return (
    <div
      className={cn(
        "card fixed bottom-5 right-5 z-40 flex flex-col overflow-hidden shadow-2xl",
        "max-w-[calc(100vw-2.5rem)] max-h-[calc(100vh-2.5rem)]",
        SIZES[size],
      )}
      role="dialog"
      aria-label="Ask Full Court"
    >
      <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <div className="text-sm font-semibold">Ask Full Court</div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setSize(NEXT_SIZE[size])}
            className="btn btn-ghost px-2 py-1 text-xs"
            title={`Resize (now ${size})`}
          >
            {size === "lg" ? "⤡ Small" : "⤢ Bigger"}
          </button>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="btn btn-ghost px-2 py-1 text-xs"
            title="Close (Esc)"
          >
            ✕
          </button>
        </div>
      </header>

      <form
        className="flex gap-2 border-b border-border p-3"
        onSubmit={(e) => {
          e.preventDefault();
          run(question);
        }}
      >
        <input
          ref={inputRef}
          className="input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={`Ask about ${meta.league_label} data…`}
        />
        <button type="submit" className="btn btn-primary whitespace-nowrap" disabled={loading}>
          {loading ? "…" : "Ask"}
        </button>
      </form>

      <div className="min-h-0 flex-1 overflow-auto p-3 text-sm">
        {err && <div className="text-bad">{err}</div>}

        {!result && !err && (
          <div className="space-y-2">
            <div className="label">Try one of these</div>
            {examples.map((ex) => (
              <button
                key={ex}
                type="button"
                className="block w-full border border-border bg-bg px-3 py-2 text-left text-xs text-mute hover:border-accent hover:text-ink"
                onClick={() => {
                  setQuestion(ex);
                  run(ex);
                }}
              >
                {ex}
              </button>
            ))}
          </div>
        )}

        {result && <Answer result={result} meta={meta} onNavigate={onNavigate} />}
      </div>
    </div>
  );
}

function Answer({
  result,
  meta,
  onNavigate,
}: {
  result: AskResult;
  meta: Meta;
  /** Called with the API's page name; App.tsx routes it to a tab and view. */
  onNavigate?: (page: string, navigate?: { page: string; state: any }) => void;
}) {
  // Format by the answer's league, not the one the UI happens to be showing.
  const fmt = (s: string) => formatSeason(s, result.season_format ?? meta.season_format);

  if (result.status === "needs_clarification") {
    return (
      <div className="space-y-2">
        <div>{result.summary}</div>
        <div className="label">Did you mean</div>
        {(result.options ?? []).map((o) => (
          <div key={o.player_name} className="border border-border bg-bg px-3 py-2 text-xs">
            <div className="text-ink">{o.player_name}</div>
            <div className="text-mute">{o.seasons.map(fmt).join(", ")}</div>
          </div>
        ))}
      </div>
    );
  }

  if (result.status === "unsupported") {
    return <div className="text-mute">{result.summary}</div>;
  }

  const rows = result.results ?? [];
  return (
    <div className="space-y-3">
      <div className="text-ink">{result.summary}</div>

      {result.intent === "explorer" && <PlayerRows rows={rows} columns={result.columns} fmt={fmt} />}
      {result.intent === "similarity" && <SimilarRows rows={rows} fmt={fmt} meta={meta} />}
      {result.intent === "compare" && <CompareRows rows={rows} />}
      {result.intent === "shot_analysis" && <ZoneRows rows={rows} />}
      {result.intent === "team_explorer" && (
        <TeamRows rows={rows} metric={result.metric} fmt={fmt} />
      )}

      {result.target_page && onNavigate && (
        <button
          type="button"
          className="btn btn-ghost w-full text-xs"
          onClick={() =>
            onNavigate(result.target_page!, result.navigate)
          }
        >
          Open in {PAGE_LABELS[result.target_page] ?? result.target_page}
        </button>
      )}
    </div>
  );
}

const PAGE_LABELS: Record<string, string> = {
  explorer: "Explorer",
  similarity: "Similarity",
  compare: "Compare",
  shots: "Shot Analysis",
  teams: "Teams",
};

function Cell({ children }: { children: React.ReactNode }) {
  return <td className="whitespace-nowrap px-2 py-1.5">{children}</td>;
}

function PlayerRows({
  rows,
  columns,
  fmt,
}: {
  rows: any[];
  columns?: string[];
  fmt: (s: string) => string;
}) {
  const metrics = (columns ?? []).filter(
    (c) => !["player_name", "season", "team_abbr"].includes(c),
  );
  if (rows.length === 0) return <div className="text-mute">No player-seasons matched.</div>;
  return (
    <div className="overflow-x-auto border border-border">
      <table className="w-full text-xs">
        <thead className="text-mute">
          <tr className="border-b border-border">
            <th className="px-2 py-1.5 text-left font-medium">Player</th>
            <th className="px-2 py-1.5 text-left font-medium">Season</th>
            {metrics.map((m) => (
              <th key={m} className="px-2 py-1.5 text-right font-medium">
                {metricLabel(m)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.player_name}-${r.season}-${i}`} className="border-t border-border/60">
              <Cell>{r.player_name}</Cell>
              <Cell>{fmt(r.season)}</Cell>
              {metrics.map((m) => (
                <td key={m} className="px-2 py-1.5 text-right tabular-nums">
                  {formatValue(m, r[m])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SimilarRows({
  rows,
  fmt,
  meta,
}: {
  rows: any[];
  fmt: (s: string) => string;
  meta: Meta;
}) {
  return (
    <ol className="space-y-1">
      {rows.map((m, i) => (
        <li
          key={`${m.player_id}-${m.season}`}
          className="flex items-center gap-2 border border-border bg-bg px-2 py-1.5 text-xs"
        >
          <span className="w-4 text-mute">{i + 1}</span>
          <Avatar name={m.player_name} id={m.player_id} league={meta.league} size={22} />
          <span className="flex-1 text-ink">
            {fmt(m.season)} {m.player_name}
          </span>
          <span className="tabular-nums text-mute">{(m.similarity * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ol>
  );
}

function CompareRows({ rows }: { rows: any[] }) {
  const metrics = Object.keys(rows[0]?.values ?? {});
  return (
    <div className="overflow-x-auto border border-border">
      <table className="w-full text-xs">
        <thead className="text-mute">
          <tr className="border-b border-border">
            <th className="px-2 py-1.5 text-left font-medium">Season</th>
            {metrics.map((m) => (
              <th key={m} className="px-2 py-1.5 text-right font-medium">
                {metricLabel(m)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="border-t border-border/60">
              <Cell>{r.key}</Cell>
              {metrics.map((m) => (
                <td key={m} className="px-2 py-1.5 text-right tabular-nums">
                  {formatValue(m, r.values?.[m])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ZoneRows({ rows }: { rows: any[] }) {
  return (
    <div className="overflow-x-auto border border-border">
      <table className="w-full text-xs">
        <thead className="text-mute">
          <tr className="border-b border-border">
            <th className="px-2 py-1.5 text-left font-medium">Zone</th>
            <th className="px-2 py-1.5 text-right font-medium">FGA</th>
            <th className="px-2 py-1.5 text-right font-medium">FG%</th>
            <th className="px-2 py-1.5 text-right font-medium">vs league</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((z) => (
            <tr key={z.zone} className="border-t border-border/60">
              <Cell>{z.zone}</Cell>
              <td className="px-2 py-1.5 text-right tabular-nums">{z.fga}</td>
              <td className="px-2 py-1.5 text-right tabular-nums">
                {z.fg_pct == null ? "—" : `${(z.fg_pct * 100).toFixed(1)}%`}
              </td>
              <td
                className={cn(
                  "px-2 py-1.5 text-right tabular-nums",
                  z.diff > 0 ? "text-good" : z.diff < 0 ? "text-bad" : "text-mute",
                )}
              >
                {z.diff == null ? "—" : `${z.diff > 0 ? "+" : ""}${(z.diff * 100).toFixed(1)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TeamRows({
  rows,
  metric,
  fmt,
}: {
  rows: any[];
  metric?: string;
  fmt: (s: string) => string;
}) {
  if (rows.length === 0) return <div className="text-mute">No team-seasons in that range.</div>;
  return (
    <div className="overflow-x-auto border border-border">
      <table className="w-full text-xs">
        <thead className="text-mute">
          <tr className="border-b border-border">
            <th className="px-2 py-1.5 text-left font-medium">Team-season</th>
            <th className="px-2 py-1.5 text-right font-medium">{metric ?? "value"}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.team}-${r.season}-${i}`} className="border-t border-border/60">
              <Cell>
                {fmt(String(r.season))} {r.team}
              </Cell>
              <td className="px-2 py-1.5 text-right tabular-nums">
                {typeof r[metric ?? ""] === "number"
                  ? Number(r[metric ?? ""]).toFixed(3)
                  : (r[metric ?? ""] ?? "—")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
