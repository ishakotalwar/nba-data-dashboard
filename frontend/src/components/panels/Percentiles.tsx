import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { PlayerCombobox } from "@/components/ui/PlayerCombobox";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";

export function Percentiles({ meta }: { meta: Meta }) {
  const [player, setPlayer] = useState(meta.players[0] ?? "");
  const [season, setSeason] = useState(meta.seasons[meta.seasons.length - 1] ?? "");
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    if (!player || !season) return;
    api
      .percentiles(player, season)
      .then(setData)
      .catch(() => setData(null));
  }, [player, season]);

  const { traces, layout } = useMemo(() => {
    if (!data?.rows?.length) return { traces: [], layout: {} };
    const rows = [...data.rows].sort((a, b) => a.percentile - b.percentile);
    return {
      traces: [
        {
          type: "bar",
          orientation: "h",
          x: rows.map((r) => r.percentile),
          y: rows.map((r) => r.metric),
          text: rows.map((r) => (r.value != null ? (Math.abs(r.value) < 10 ? r.value.toFixed(2) : r.value.toFixed(1)) : "")),
          textposition: "outside",
          hovertemplate: "%{y}: %{x}th pct · value %{text}<extra></extra>",
          marker: {
            color: rows.map((r) => r.percentile),
            colorscale: [
              [0, "#d73027"],
              [0.5, "#ffffbf"],
              [1, "#1a9850"],
            ],
            cmin: 0,
            cmax: 100,
            colorbar: { title: "pct", tickfont: { color: "#8a94a2" } },
          },
        },
      ],
      layout: {
        xaxis: { range: [0, 115], title: "Percentile (100 = best)", gridcolor: "#1f2630" },
        yaxis: { gridcolor: "#1f2630" },
        height: Math.max(360, 32 * rows.length + 120),
      },
    };
  }, [data]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="League percentile rankings"
          subtitle="Ranks this player's season across every NBA player that season. drtg and tov are inverted (lower is better)."
        />
        <CardBody className="grid gap-3 md:grid-cols-2">
          <div>
            <div className="label mb-1.5">Player</div>
            <PlayerCombobox options={meta.players} value={player} onChange={setPlayer} />
          </div>
          <div>
            <div className="label mb-1.5">Season</div>
            <Select value={season} onChange={setSeason} options={meta.seasons} />
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader title={player && season ? `${player} — ${season}` : "—"} />
        <CardBody>
          {!traces.length ? (
            <div className="grid place-items-center py-10 text-sm text-mute">No data for this player/season.</div>
          ) : (
            <Plot data={traces as any} layout={layout} height={(layout as any).height} />
          )}
        </CardBody>
      </Card>
    </div>
  );
}
