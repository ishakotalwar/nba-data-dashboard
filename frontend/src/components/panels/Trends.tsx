import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { MultiSelect } from "@/components/ui/MultiSelect";
import { PlayerCombobox } from "@/components/ui/PlayerCombobox";
import { Plot } from "@/components/ui/Plot";

export function Trends({ meta }: { meta: Meta }) {
  const [player, setPlayer] = useState<string>(meta.players[0] ?? "");
  const [metrics, setMetrics] = useState<string[]>(
    ["pts", "ast", "reb", "ts_pct"].filter((m) => meta.metrics.includes(m))
  );
  const [league, setLeague] = useState(true);
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    if (!player || metrics.length === 0) return;
    api.trends({ player, metrics, league }).then(setData).catch(() => setData(null));
  }, [player, metrics, league]);

  const { traces, layout } = useMemo(() => {
    if (!data) return { traces: [], layout: {} };
    const n = data.metrics.length;
    const layout: any = {
      grid: { rows: n, columns: 1, pattern: "independent" },
      height: Math.min(900, 220 * n + 40),
      showlegend: true,
    };
    const traces: any[] = [];
    data.metrics.forEach((m: string, i: number) => {
      ["player", "League avg"].forEach((src) => {
        if (src === "League avg" && !league) return;
        const rows = data.series.filter((r: any) => r.metric === m && (src === "player" ? r.source === data.player : r.source === "League avg"));
        if (!rows.length) return;
        traces.push({
          type: "scatter",
          mode: "lines+markers",
          name: src === "player" ? data.player : "League avg",
          x: rows.map((r: any) => r.season),
          y: rows.map((r: any) => r.value),
          xaxis: `x${i + 1}`,
          yaxis: `y${i + 1}`,
          legendgroup: src,
          showlegend: i === 0,
          line: src === "League avg" ? { dash: "dot", width: 1.5 } : { width: 2.5 },
          marker: { size: 6 },
        });
      });
      layout[`yaxis${i + 1}`] = { title: m, gridcolor: "#1f2630" };
      layout[`xaxis${i + 1}`] = { gridcolor: "#1f2630" };
    });
    return { traces, layout };
  }, [data, league]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Trend lines" subtitle="Season-over-season trajectory for any player" />
        <CardBody className="grid gap-3 md:grid-cols-3">
          <div>
            <div className="label mb-1.5">Player</div>
            <PlayerCombobox options={meta.players} value={player} onChange={setPlayer} />
          </div>
          <div className="md:col-span-2">
            <div className="label mb-1.5">Metrics</div>
            <MultiSelect options={meta.metrics} value={metrics} onChange={setMetrics} />
          </div>
          <label className="flex items-center gap-2 text-sm text-mute md:col-span-3">
            <input type="checkbox" checked={league} onChange={(e) => setLeague(e.target.checked)} />
            Overlay league average
          </label>
        </CardBody>
      </Card>

      <Card>
        <CardHeader title={player || "—"} />
        <CardBody>
          {traces.length === 0 ? (
            <div className="grid place-items-center py-10 text-sm text-mute">Pick a player and at least one metric.</div>
          ) : (
            <Plot data={traces as any} layout={layout} height={(layout as any).height ?? 480} />
          )}
        </CardBody>
      </Card>
    </div>
  );
}
