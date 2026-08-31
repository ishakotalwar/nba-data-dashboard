import { useEffect, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { playerAvatar } from "@/components/ui/Avatar";
import { formatValue, label, shortLabel, sortMetrics } from "@/lib/metrics";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

type Filt = { metric: string; op: string; value: number; value2?: number };

const OPS = [
  { value: ">=", label: "≥" },
  { value: ">", label: ">" },
  { value: "<=", label: "≤" },
  { value: "<", label: "<" },
  { value: "=", label: "=" },
  { value: "between", label: "between" },
];

/** `seed` is the ExplorerRequest that Ask Full Court just executed. */
export function Explorer({ meta, seed }: { meta: Meta; seed?: any }) {
  const avatar = playerAvatar(meta);
  const seasons = meta.seasons;
  const seasonOptions = seasons.map((s) => ({ value: s, label: formatSeason(s, meta.season_format) }));
  const metricKeys = sortMetrics(meta.metrics);

  const [from, setFrom] = useState(seasons[0] ?? "");
  const [to, setTo] = useState(seasons.at(-1) ?? "");
  const [minGp, setMinGp] = useState("0");
  const [minMin, setMinMin] = useState("0");
  const [team, setTeam] = useState("");
  const [player, setPlayer] = useState("");
  const [filters, setFilters] = useState<Filt[]>([
    { metric: "pts", op: ">=", value: 25 },
  ]);
  const [sort, setSort] = useState("pts");
  const [dir, setDir] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(1);
  const [teams, setTeams] = useState<string[]>([]);
  useEffect(() => {
    if (!seed) return;
    if (seed.season_from) setFrom(String(seed.season_from));
    if (seed.season_to) setTo(String(seed.season_to));
    if (Array.isArray(seed.filters) && seed.filters.length) {
      setFilters(
        seed.filters.map((f: any) => ({
          metric: f.metric,
          op: f.op,
          value: f.value,
          value2: f.value2 ?? undefined,
        })),
      );
    }
    if (seed.sort) setSort(seed.sort);
    if (seed.dir) setDir(seed.dir);
    setPage(1);
    setSeedPending(true);
  }, [seed]);

  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  // Set when a seed lands, cleared once the query it implies has been run.
  const [seedPending, setSeedPending] = useState(false);

  useEffect(() => {
    api.explorerFields(meta.league).then((f) => setTeams(f.teams ?? [])).catch(() => setTeams([]));
  }, [meta.league]);

  // Runs on the render *after* the seed setters land, so `run` reads the
  // seeded values rather than the previous ones.
  useEffect(() => {
    if (!seedPending) return;
    setSeedPending(false);
    run(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seedPending, from, to, filters, sort, dir]);

  const run = (toPage = 1) => {
    setErr(null);
    setPage(toPage);
    api
      .explorer({
        league: meta.league,
        season_from: from || undefined,
        season_to: to || undefined,
        min_gp: Number(minGp) || 0,
        min_min: Number(minMin) || 0,
        team: team || undefined,
        player: player || undefined,
        filters: filters.filter((f) => f.metric && !Number.isNaN(f.value)),
        sort,
        dir,
        page: toPage,
        page_size: 25,
      })
      .then(setData)
      .catch((e) => {
        setErr(e.message);
        setData(null);
      });
  };

  // Re-run on sort changes once results exist.
  useEffect(() => {
    if (data) run(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sort, dir]);

  const setFilter = (i: number, patch: Partial<Filt>) =>
    setFilters((fs) => fs.map((f, idx) => (idx === i ? { ...f, ...patch } : f)));

  const columns: string[] = data?.columns ?? [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Stat explorer"
        />
        <CardBody className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-6">
            <div>
              <div className="label mb-1.5">Season from</div>
              <Select value={from} onChange={setFrom} options={seasonOptions} />
            </div>
            <div>
              <div className="label mb-1.5">to</div>
              <Select value={to} onChange={setTo} options={seasonOptions} />
            </div>
            <div>
              <div className="label mb-1.5">Min games</div>
              <input className="input" value={minGp} onChange={(e) => setMinGp(e.target.value)} inputMode="numeric" />
            </div>
            <div>
              <div className="label mb-1.5">Min minutes</div>
              <input className="input" value={minMin} onChange={(e) => setMinMin(e.target.value)} inputMode="numeric" />
            </div>
            <div>
              <div className="label mb-1.5">Team</div>
              <Select value={team} onChange={setTeam} options={teams} placeholder="Any" />
            </div>
            <div>
              <div className="label mb-1.5">Player contains</div>
              <input className="input" value={player} onChange={(e) => setPlayer(e.target.value)} placeholder="Any" />
            </div>
          </div>

          <div className="space-y-2">
            <div className="label">Conditions</div>
            {filters.map((f, i) => (
              <div key={i} className="flex flex-wrap items-center gap-2">
                <div className="w-52">
                  <Select
                    value={f.metric}
                    onChange={(v) => setFilter(i, { metric: v })}
                    options={metricKeys.map((k) => ({ value: k, label: label(k) }))}
                  />
                </div>
                <div className="w-32">
                  <Select value={f.op} onChange={(v) => setFilter(i, { op: v })} options={OPS} />
                </div>
                <input
                  className="input w-28"
                  value={String(f.value)}
                  onChange={(e) => setFilter(i, { value: Number(e.target.value) })}
                  inputMode="decimal"
                />
                {f.op === "between" && (
                  <input
                    className="input w-28"
                    value={String(f.value2 ?? "")}
                    onChange={(e) => setFilter(i, { value2: Number(e.target.value) })}
                    inputMode="decimal"
                    placeholder="upper"
                  />
                )}
                <button
                  onClick={() => setFilters((fs) => fs.filter((_, idx) => idx !== i))}
                  className="border border-border px-3 py-2 text-mute transition hover:text-ink"
                >
                  ×
                </button>
              </div>
            ))}
            <div className="flex items-center gap-3">
              <button
                className="btn btn-ghost"
                onClick={() => setFilters((fs) => [...fs, { metric: "ts_pct", op: ">=", value: 0.55 }])}
              >
                + Add condition
              </button>
              <button className="btn btn-primary" onClick={() => run(1)}>
                Run query
              </button>
              <span className="text-xs text-mute">
                Percentages are rates: 38% is 0.38.
              </span>
            </div>
          </div>
          {err && <div className="text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={data ? `${data.total.toLocaleString()} player-seasons` : "Results"}
          subtitle={data ? `Page ${data.page} of ${data.pages}` : undefined}
          right={
            data && data.pages > 1 ? (
              <div className="flex items-center gap-2 text-sm">
                <button
                  className="btn btn-ghost disabled:opacity-40"
                  disabled={page <= 1}
                  onClick={() => run(page - 1)}
                >
                  ‹ Prev
                </button>
                <button
                  className="btn btn-ghost disabled:opacity-40"
                  disabled={page >= data.pages}
                  onClick={() => run(page + 1)}
                >
                  Next ›
                </button>
              </div>
            ) : undefined
          }
        />
        <CardBody className="p-0">
          {!data ? (
            <div className="px-5 py-8 text-sm text-mute">No results yet.</div>
          ) : data.rows.length === 0 ? (
            <div className="px-5 py-8 text-sm text-mute">Nothing matched those conditions.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-mute">
                    {columns.map((c) => (
                      <th
                        key={c}
                        className={cn(
                          "px-3 py-2 font-medium",
                          ["player_name", "season", "team_abbr"].includes(c) ? "text-left" : "text-right"
                        )}
                      >
                        <button
                          onClick={() => {
                            if (sort === c) setDir((d) => (d === "desc" ? "asc" : "desc"));
                            else {
                              setSort(c);
                              setDir("desc");
                            }
                          }}
                          className={cn("hover:text-ink", sort === c && "text-accent")}
                        >
                          {c === "player_name" ? "Player" : c === "team_abbr" ? "Team" : c === "season" ? "Season" : shortLabel(c)}
                          {sort === c ? (dir === "desc" ? " ↓" : " ↑") : ""}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r: any) => (
                    <tr key={`${r.player_name}-${r.season}`} className="border-t border-border/60">
                      {columns.map((c) => (
                        <td
                          key={c}
                          className={cn(
                            "px-3 py-2",
                            ["player_name", "season", "team_abbr"].includes(c)
                              ? "whitespace-nowrap"
                              : "text-right tabular-nums"
                          )}
                        >
                          {c === "player_name" ? (
                            <span className="flex items-center gap-2">
                              {avatar(r.player_name, 24)}
                              {r.player_name}
                            </span>
                          ) : c === "season" ? (
                            formatSeason(r[c], meta.season_format) || "—"
                          ) : c === "team_abbr" ? (
                            r[c] ?? "—"
                          ) : (
                            formatValue(c, r[c])
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
