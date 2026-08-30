import { useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { PlayerCombobox } from "@/components/ui/PlayerCombobox";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";
import { ErrorNotice } from "@/components/ui/ErrorNotice";
import { buildCourtShapes, courtAxis } from "@/lib/court";
import { cn } from "@/lib/cn";

export function ShotChart({ meta }: { meta: Meta }) {
  const [player, setPlayer] = useState("Stephen Curry");
  const [season, setSeason] = useState(meta.seasons[meta.seasons.length - 1] ?? "");
  const [mode, setMode] = useState<"scatter" | "hex">("hex");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function fetchIt() {
    setLoading(true);
    setErr(null);
    try {
      const d = await api.shots(player, season, mode, meta.league);
      setData(d);
    } catch (e: any) {
      setErr(e.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  const traces = useMemo(() => {
    if (!data) return [];
    if (data.mode === "scatter") {
      const makes = data.shots.filter((s: any) => s.made === 1);
      const misses = data.shots.filter((s: any) => s.made === 0);
      return [
        {
          type: "scatter",
          mode: "markers",
          name: "Miss",
          x: misses.map((s: any) => s.x),
          y: misses.map((s: any) => s.y),
          marker: { symbol: "x", size: 6, color: "#4a5568", opacity: 0.5 },
          hoverinfo: "skip",
        },
        {
          type: "scatter",
          mode: "markers",
          name: "Make",
          x: makes.map((s: any) => s.x),
          y: makes.map((s: any) => s.y),
          marker: { symbol: "circle", size: 6, color: "#ff6a3d", opacity: 0.85 },
          hoverinfo: "skip",
        },
      ];
    }
    // hex
    const hexes = data.hexes;
    if (!hexes?.length) return [];
    const cmax = Math.max(...hexes.map((h: any) => h.count), 1);
    const fg = data.fg_pct;
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "FG% by zone",
        x: hexes.map((h: any) => h.x),
        y: hexes.map((h: any) => h.y),
        marker: {
          symbol: "hexagon",
          size: hexes.map((h: any) => 8 + 22 * Math.sqrt(h.count / cmax)),
          color: hexes.map((h: any) => h.pct),
          colorscale: [
            [0, "#2166ac"],
            [0.5, "#f7f7f7"],
            [1, "#b2182b"],
          ],
          cmin: Math.max(0, fg - 0.25),
          cmax: Math.min(1, fg + 0.25),
          colorbar: { title: "FG%", tickformat: ".0%" },
          line: { width: 0.5, color: "#444" },
        },
        customdata: hexes.map((h: any) => [h.pct, h.count]),
        hovertemplate: "FG%: %{customdata[0]:.0%} · %{customdata[1]} shots<extra></extra>",
        showlegend: false,
      },
    ];
  }, [data]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Shot chart"
          subtitle="Hex mode bins shots into zones — color shows FG% vs. this player's season average, size shows volume."
        />
        <CardBody className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <div>
              <div className="label mb-1.5">Player</div>
              <PlayerCombobox options={meta.players} value={player} onChange={setPlayer} />
            </div>
            <div>
              <div className="label mb-1.5">Season</div>
              <Select value={season} onChange={setSeason} options={meta.seasons} />
            </div>
            <div>
              <div className="label mb-1.5">View</div>
              <div className="flex gap-2">
                {(["hex", "scatter"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={cn("btn flex-1", mode === m ? "btn-primary" : "btn-ghost")}
                  >
                    {m === "hex" ? "Hexbin" : "Scatter"}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="btn btn-primary" onClick={fetchIt} disabled={loading}>
              {loading ? "Fetching…" : "Fetch shots"}
            </button>
            {data?.count != null && (
              <div className="text-sm text-mute">
                {data.count} shots · FG% {(data.fg_pct * 100).toFixed(1)}%
              </div>
            )}
          </div>
          {err && <ErrorNotice message={err} onRetry={fetchIt} />}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title={data ? `${data.player} — ${data.season}` : "—"} />
        <CardBody>
          {traces.length === 0 ? (
            <div className="grid place-items-center py-10 text-sm text-mute">
              Press “Fetch shots”.
            </div>
          ) : (
            <Plot
              data={traces as any}
              layout={{
                shapes: buildCourtShapes(meta.court) as any,
                ...courtAxis,
                margin: { t: 10, l: 10, r: 10, b: 10 },
                showlegend: false,
              }}
              height={640}
            />
          )}
        </CardBody>
      </Card>
    </div>
  );
}
