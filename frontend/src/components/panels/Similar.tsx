import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { PlayerCombobox } from "@/components/ui/PlayerCombobox";
import { Slider } from "@/components/ui/Slider";
import { Plot } from "@/components/ui/Plot";

const FEATURES = ["pts", "ast", "reb", "tov", "ts_pct", "usg_pct", "ortg", "drtg"];

export function Similar({ meta }: { meta: Meta }) {
  const [anchor, setAnchor] = useState(meta.players[0] ?? "");
  const [k, setK] = useState(5);
  const [weights, setWeights] = useState<Record<string, number>>(
    Object.fromEntries(FEATURES.map((f) => [f, 1.0]))
  );
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!anchor) return;
    api
      .similar({
        anchor,
        k,
        weights,
        seasonLo: meta.seasons[0],
        seasonHi: meta.seasons[meta.seasons.length - 1],
      })
      .then((d) => {
        setData(d);
        setErr(null);
      })
      .catch((e) => setErr(e.message));
  }, [anchor, k, weights, meta.seasons]);

  const radarTraces = useMemo(() => {
    if (!data?.radar) return [];
    const feats = data.radar.features as string[];
    const close = (v: number[]) => [...v, v[0]];
    const out: any[] = [
      {
        type: "scatterpolar",
        name: data.radar.anchor.name,
        r: close(data.radar.anchor.values),
        theta: [...feats, feats[0]],
        fill: "toself",
        opacity: 0.65,
      },
    ];
    for (const peer of data.radar.peers) {
      out.push({
        type: "scatterpolar",
        name: peer.name,
        r: close(peer.values),
        theta: [...feats, feats[0]],
        fill: "toself",
        opacity: 0.28,
      });
    }
    return out;
  }, [data]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Similar players"
          subtitle="Weighted cosine similarity over standardized per-season averages"
        />
        <CardBody className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <div>
              <div className="label mb-1.5">Anchor</div>
              <PlayerCombobox options={meta.players} value={anchor} onChange={setAnchor} />
            </div>
            <div>
              <div className="label mb-1.5">Top-K: {k}</div>
              <Slider value={k} onChange={setK} min={3} max={10} />
            </div>
          </div>
          <div>
            <div className="label mb-2">Feature weights</div>
            <div className="grid grid-cols-2 gap-x-6 gap-y-3 md:grid-cols-4">
              {FEATURES.filter((f) => meta.metrics.includes(f)).map((f) => (
                <div key={f}>
                  <div className="flex items-center justify-between text-xs text-mute">
                    <span>{f}</span>
                    <span className="tabular-nums">{weights[f].toFixed(1)}</span>
                  </div>
                  <Slider
                    value={weights[f]}
                    onChange={(v) => setWeights((w) => ({ ...w, [f]: v }))}
                    min={0}
                    max={2}
                    step={0.1}
                  />
                </div>
              ))}
            </div>
          </div>
        </CardBody>
      </Card>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-2">
          <CardHeader title="Top matches" />
          <CardBody className="p-0">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wider text-mute">
                  <th className="px-5 py-2">#</th>
                  <th className="px-5 py-2">Player</th>
                  <th className="px-5 py-2 text-right">Similarity</th>
                </tr>
              </thead>
              <tbody>
                {err && (
                  <tr><td className="px-5 py-4 text-bad" colSpan={3}>{err}</td></tr>
                )}
                {data?.matches?.map((m: any, i: number) => (
                  <tr key={m.player_name} className="border-t border-border/60">
                    <td className="px-5 py-2 text-mute">{i + 1}</td>
                    <td className="px-5 py-2">{m.player_name}</td>
                    <td className="px-5 py-2 text-right tabular-nums">{(m.similarity * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardBody>
        </Card>

        <Card className="lg:col-span-3">
          <CardHeader
            title="Radar overlay"
            subtitle="League-normalized (0–1); drtg/tov inverted so larger area = better"
          />
          <CardBody>
            {radarTraces.length === 0 ? (
              <div className="grid place-items-center py-10 text-sm text-mute">No data.</div>
            ) : (
              <Plot
                data={radarTraces as any}
                layout={{
                  polar: {
                    bgcolor: "rgba(0,0,0,0)",
                    radialaxis: { range: [0, 1], gridcolor: "#1f2630", color: "#8a94a2" },
                    angularaxis: { gridcolor: "#1f2630", color: "#cbd3de" },
                  },
                  margin: { t: 20, l: 20, r: 20, b: 20 },
                }}
                height={520}
              />
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
