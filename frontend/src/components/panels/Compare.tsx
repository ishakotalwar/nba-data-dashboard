import { useEffect, useMemo, useState } from "react";
import { api, type Meta, type PlayerSeason } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { MultiSelect } from "@/components/ui/MultiSelect";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";
import { PlayerSeasonSelector, emptySelection } from "@/components/ui/PlayerSeasonSelector";
import { PlayerLegend } from "@/components/ui/PlayerLegend";
import { playerAvatar } from "@/components/ui/Avatar";
import { formatDelta, formatValue, label, ordinal, shortLabel, sortMetrics } from "@/lib/metrics";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

const MODES = [
  { v: "season", label: "Single season" },
  { v: "career", label: "Career" },
] as const;

const MAX = 5;

/** `seed` carries the player-seasons Ask Full Court just compared. */
export function Compare({ meta, seed }: { meta: Meta; seed?: any }) {
  const avatar = playerAvatar(meta);
  const [mode, setMode] = useState<"season" | "career">("season");
  const [picks, setPicks] = useState<PlayerSeason[]>([{ ...emptySelection }]);

  useEffect(() => {
    if (!Array.isArray(seed?.players) || seed.players.length === 0) return;
    setPicks(
      seed.players.map((p: any) => ({
        playerId: p.player_id,
        playerName: p.player_name ?? "",
        season: String(p.season),
      })),
    );
  }, [seed]);
  const [metrics, setMetrics] = useState<string[]>([]);
  const [view, setView] = useState<"radar" | "bar">("radar");
  const [careerMetric, setCareerMetric] = useState("pts");
  const [data, setData] = useState<any>(null);
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const ready = picks.filter((p) => p.playerId && (mode === "career" || p.season));

  useEffect(() => {
    if (ready.length === 0 || (mode === "season" && metrics.length === 0)) {
      setData(null);
      return;
    }
    setErr(null);
    api
      .compare({
        selections: ready.map((p) => ({ player_id: p.playerId!, season: p.season || undefined })),
        metrics: mode === "career" ? [careerMetric] : metrics,
        league: meta.league,
        mode,
      })
      .then(setData)
      .catch((e) => {
        setErr(e.message);
        setData(null);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(ready), JSON.stringify(metrics), mode, careerMetric, meta.league]);

  const rows = data?.mode === "season" ? data.rows ?? [] : [];
  const curves = data?.mode === "career" ? data.curves ?? [] : [];

  const setPick = (i: number, v: PlayerSeason) =>
    setPicks((ps) => ps.map((p, idx) => (idx === i ? v : p)));
  const addPick = () => setPicks((ps) => (ps.length >= MAX ? ps : [...ps, { ...emptySelection }]));
  const removePick = (i: number) =>
    setPicks((ps) => (ps.length === 1 ? [{ ...emptySelection }] : ps.filter((_, idx) => idx !== i)));

  // --- single-season charts -------------------------------------------------
  const seasonTraces = useMemo(() => {
    if (!rows.length) return [];
    if (view === "radar" && data?.radar) {
      const feats: string[] = data.radar.features;
      return Object.entries(data.radar.values as Record<string, number[]>).map(([key, vals]) => ({
        type: "scatterpolar",
        name: key,
        r: [...vals, vals[0]],
        theta: [...feats.map(shortLabel), shortLabel(feats[0])],
        fill: "toself",
        opacity: 0.5,
        hovertemplate: `%{theta}: %{r:.0%} percentile<extra>${key}</extra>`,
      }));
    }
    return (data?.metrics ?? []).map((m: string) => ({
      type: "bar",
      name: shortLabel(m),
      x: rows.map((r: any) => r.key),
      y: rows.map((r: any) => r.values[m]),
      hovertemplate: `%{x}<br>${label(m)}: %{y}<extra></extra>`,
    }));
  }, [rows, view, data]);

  const careerTraces = useMemo(() => {
    if (!curves.length) return [];
    const useAge = curves.every((c: any) => c.has_age);
    return curves.map((c: any) => ({
      type: "scatter",
      mode: "lines+markers",
      name: c.player_name,
      x: c.points.map((p: any) => (useAge ? p.age : formatSeason(p.season, meta.season_format))),
      y: c.points.map((p: any) => p[careerMetric]),
      hovertemplate: `${useAge ? "age %{x}" : "%{x}"}: %{y}<extra>${"%{fullData.name}"}</extra>`,
    }));
  }, [curves, careerMetric]);

  const useAge = curves.length > 0 && curves.every((c: any) => c.has_age);

  const sortedRows = useMemo(() => {
    if (!sort) return rows;
    return [...rows].sort((a: any, b: any) => {
      const x = a.values[sort.key], y = b.values[sort.key];
      if (x == null) return 1;
      if (y == null) return -1;
      return x === y ? 0 : (x < y ? -1 : 1) * sort.dir;
    });
  }, [rows, sort]);

  const metricOptions = sortMetrics(meta.metrics);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Compare player-seasons"
          right={
            <div className="flex gap-4 text-sm">
              {MODES.map((m) => (
                <button
                  key={m.v}
                  onClick={() => setMode(m.v)}
                  className={cn(
                    "border-b-2 pb-0.5 transition",
                    mode === m.v ? "border-accent text-ink" : "border-transparent text-mute hover:text-ink"
                  )}
                >
                  {m.label}
                </button>
              ))}
            </div>
          }
        />
        <CardBody className="space-y-3">
          {picks.map((p, i) => (
            <div key={i} className="flex items-end gap-2">
              <PlayerSeasonSelector
                meta={meta}
                value={p}
                onChange={(v) => setPick(i, v)}
                seasonless={mode === "career"}
                playerLabel={i === 0 ? "Player" : ""}
                className="grid flex-1 gap-3 md:grid-cols-2"
              />
              <button
                onClick={() => removePick(i)}
                title="Remove"
                className="h-[42px] border border-border px-3 text-mute transition hover:text-ink"
              >
                ×
              </button>
            </div>
          ))}
          <div className="flex items-center gap-4">
            <button
              onClick={addPick}
              disabled={picks.length >= MAX}
              className="btn btn-ghost disabled:opacity-40"
            >
              + Add player
            </button>
            {mode === "season" ? (
              <div className="flex-1">
                <div className="label mb-1.5">Metrics</div>
                <MultiSelect
                  options={metricOptions}
                  value={metrics}
                  onChange={setMetrics}
                  renderLabel={label}
                />
              </div>
            ) : (
              <div className="w-56">
                <div className="label mb-1.5">Metric</div>
                <Select
                  value={careerMetric}
                  onChange={setCareerMetric}
                  options={metricOptions.map((k) => ({ value: k, label: label(k) }))}
                />
              </div>
            )}
          </div>
          {err && <div className="text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      {mode === "season" ? (
        <>
          <Card>
            <CardHeader
              title={view === "radar" ? "Percentile profile" : "Raw values"}
              right={
                <div className="flex gap-4 text-sm">
                  {(["radar", "bar"] as const).map((v) => (
                    <button
                      key={v}
                      onClick={() => setView(v)}
                      className={cn(
                        "border-b-2 pb-0.5 transition",
                        view === v ? "border-accent text-ink" : "border-transparent text-mute hover:text-ink"
                      )}
                    >
                      {v === "radar" ? "Radar" : "Bars"}
                    </button>
                  ))}
                </div>
              }
            />
            <CardBody>
              {view === "radar" && rows.length > 0 && (
                <PlayerLegend
                  names={rows.map((r: any) => r.key)}
                  renderAvatar={(name, size) =>
                    avatar(rows.find((r: any) => r.key === name)?.player_name ?? name, size)
                  }
                  className="mb-2 px-1"
                />
              )}
              <Plot
                data={seasonTraces as any}
                layout={
                  view === "radar"
                    ? {
                        showlegend: false,
                        polar: {
                          bgcolor: "rgba(0,0,0,0)",
                          radialaxis: { range: [0, 1], tickformat: ".0%", gridcolor: "#1f2630", color: "#8a94a2" },
                          angularaxis: { gridcolor: "#1f2630", color: "#cbd3de" },
                        },
                        margin: { t: 30, l: 40, r: 40, b: 30 },
                      }
                    : { barmode: "group", margin: { t: 20 }, xaxis: { type: "category" }, yaxis: { title: "Value" } }
                }
                height={460}
                placeholder="Add player-seasons and pick metrics"
              />
            </CardBody>
          </Card>

          <Card>
            <CardHeader
              title="Stat table"
            />
            <CardBody className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wider text-mute">
                      <th className="px-4 py-2 font-medium">Player-season</th>
                      {(data?.metrics ?? []).map((m: string) => (
                        <th key={m} className="px-3 py-2 text-right font-medium">
                          <button
                            onClick={() =>
                              setSort((s) =>
                                s && s.key === m ? { key: m, dir: (s.dir * -1) as 1 | -1 } : { key: m, dir: -1 }
                              )
                            }
                            className={cn("hover:text-ink", sort?.key === m && "text-accent")}
                          >
                            {shortLabel(m)}
                            {sort?.key === m ? (sort.dir === -1 ? " ↓" : " ↑") : ""}
                          </button>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedRows.length === 0 && (
                      <tr>
                        <td className="px-4 py-6 text-mute" colSpan={99}>
                          Add player-seasons and pick metrics.
                        </td>
                      </tr>
                    )}
                    {sortedRows.map((r: any) => (
                      <tr key={r.key} className="border-t border-border/60">
                        <td className="whitespace-nowrap px-4 py-2">
                          <span className="flex items-center gap-2">
                            {avatar(r.player_name, 26)}
                            <span>
                              {r.key}
                              <span className="ml-2 text-xs text-mute">{r.team}</span>
                            </span>
                          </span>
                        </td>
                        {(data?.metrics ?? []).map((m: string) => (
                          <td key={m} className="px-3 py-2 text-right tabular-nums">
                            <div>{formatValue(m, r.values[m])}</div>
                            <div className="text-[11px] text-mute">
                              {r.percentiles[m] != null ? ordinal(r.percentiles[m]) : "—"}
                              {r.vs_league[m] != null && (
                                <span
                                  title="vs. that season's league average"
                                  className={r.vs_league[m] >= 0 ? " text-accent2" : " text-bad"}
                                >
                                  {" "}
                                  {formatDelta(m, r.vs_league[m])}
                                </span>
                              )}
                            </div>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardBody>
          </Card>
        </>
      ) : (
        <Card>
          <CardHeader
            title="Career trajectories"
          />
          <CardBody>
            {curves.length > 0 && (
              <PlayerLegend
                names={curves.map((c: any) => c.player_name)}
                renderAvatar={avatar}
                className="mb-2 px-1"
              />
            )}
            <Plot
              data={careerTraces as any}
              layout={{
                showlegend: false,
                margin: { t: 20 },
                xaxis: useAge
                  ? { title: "Age at season start" }
                  : { title: "Season", type: "category", nticks: 12, tickangle: 0 },
                yaxis: { title: label(careerMetric) },
                hovermode: "closest",
              }}
              height={480}
              placeholder="Add players to plot their careers"
            />
          </CardBody>
        </Card>
      )}
    </div>
  );
}
