import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { MultiSelect } from "@/components/ui/MultiSelect";
import { Plot } from "@/components/ui/Plot";
import { cn } from "@/lib/cn";

export function Compare({ meta }: { meta: Meta }) {
  const [players, setPlayers] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<string[]>(
    ["ts_pct", "ortg", "drtg"].filter((m) => meta.metrics.includes(m))
  );
  const [view, setView] = useState<"radar" | "bar">("radar");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (players.length === 0 || metrics.length === 0) {
      setData(null);
      return;
    }
    setLoading(true);
    api
      .compare({
        players,
        metrics,
        seasonLo: meta.seasons[0],
        seasonHi: meta.seasons[meta.seasons.length - 1],
        league: meta.league,
      })
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [players, metrics, meta.seasons, meta.league]);

  const traces = useMemo(() => {
    if (!data?.rows?.length) return [];
    if (data.single_season && view === "radar" && data.radar) {
      return Object.entries(data.radar.values as Record<string, number[]>).map(([name, vals]) => ({
        type: "scatterpolar",
        name,
        r: [...vals, vals[0]],
        theta: [...data.radar.features, data.radar.features[0]],
        fill: "toself",
        opacity: 0.55,
      }));
    }
    if (data.single_season) {
      // grouped bar
      return metrics.map((m) => ({
        type: "bar",
        name: m,
        x: data.rows.map((r: any) => r.player_name),
        y: data.rows.map((r: any) => r[m]),
      }));
    }
    // multi-season: small multiples by metric, lines per player
    return metrics.flatMap((m, i) =>
      players.map((p) => ({
        type: "scatter",
        mode: "lines+markers",
        name: `${p} · ${m}`,
        x: data.rows.filter((r: any) => r.player_name === p).map((r: any) => r.season),
        y: data.rows.filter((r: any) => r.player_name === p).map((r: any) => r[m]),
        xaxis: `x${i + 1}`,
        yaxis: `y${i + 1}`,
        legendgroup: p,
        showlegend: i === 0,
      }))
    );
  }, [data, metrics, players, view]);

  const layout = useMemo(() => {
    if (!data?.rows?.length) return {};
    if (data.single_season && view === "radar") {
      return {
        polar: {
          bgcolor: "rgba(0,0,0,0)",
          radialaxis: { range: [0, 1], gridcolor: "#1f2630", color: "#8a94a2" },
          angularaxis: { gridcolor: "#1f2630", color: "#cbd3de" },
        },
        margin: { t: 30, l: 30, r: 30, b: 30 },
      };
    }
    if (data.single_season) return { barmode: "group" };
    // facets
    const n = metrics.length;
    const layout: any = { grid: { rows: n, columns: 1, pattern: "independent" }, height: Math.min(900, 240 * n) };
    metrics.forEach((m, i) => {
      layout[`yaxis${i + 1}`] = { title: m, gridcolor: "#1f2630" };
      layout[`xaxis${i + 1}`] = { gridcolor: "#1f2630" };
    });
    return layout;
  }, [data, metrics, view]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Player comparison" subtitle="Compare up to 5 players on any subset of metrics" />
        <CardBody className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className="label mb-1.5">Players (up to 5)</div>
              <MultiSelect options={meta.players} value={players} onChange={setPlayers} max={5} />
            </div>
            <div>
              <div className="label mb-1.5">Metrics</div>
              <MultiSelect options={meta.metrics} value={metrics} onChange={setMetrics} />
            </div>
          </div>
          {data?.single_season && (
            <div className="flex items-center gap-2">
              {(["radar", "bar"] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className={cn("btn", view === v ? "btn-primary" : "btn-ghost")}
                >
                  {v === "radar" ? "Radar" : "Bars"}
                </button>
              ))}
            </div>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={data?.season_label ? `Season: ${data.season_label}` : "Across seasons"}
          subtitle={
            data?.single_season && view === "radar"
              ? "Values are league-normalized (0–1). drtg and tov are inverted so larger = better."
              : undefined
          }
        />
        <CardBody>
          {loading && <div className="text-sm text-mute">Loading…</div>}
          {!loading && !traces.length && (
            <div className="grid place-items-center py-10 text-sm text-mute">
              Pick at least one player and one metric.
            </div>
          )}
          {traces.length > 0 && <Plot data={traces as any} layout={layout} height={Math.max(460, (layout as any).height ?? 460)} />}
        </CardBody>
      </Card>
    </div>
  );
}
