import { useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { MultiSelect } from "@/components/ui/MultiSelect";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";
import { ErrorNotice } from "@/components/ui/ErrorNotice";
import { playerAvatar } from "@/components/ui/Avatar";
import { PlayerLegend } from "@/components/ui/PlayerLegend";

export function AgeCurves({ meta }: { meta: Meta }) {
  const avatar = playerAvatar(meta);
  const [players, setPlayers] = useState<string[]>([]);
  const [metric, setMetric] = useState("");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function fetchIt() {
    if (!players.length || !metric) return;
    setLoading(true);
    setErr(null);
    try {
      const d = await api.ageCurves({ players, metric, league: meta.league });
      setData(d);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  const traces = useMemo(() => {
    if (!data?.curves?.length) return [];
    return data.curves
      .filter((c: any) => c.points.length > 0)
      .map((c: any) => ({
        type: "scatter",
        mode: "lines+markers",
        name: c.player,
        x: c.points.map((p: any) => p.age),
        y: c.points.map((p: any) => p.value),
        hovertemplate: "age %{x} · %{y}<extra>" + c.player + "</extra>",
      }));
  }, [data]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Age / development curves"
          subtitle="Plots a metric against the player's age at season start (Oct 1). Birthdates looked up live."
        />
        <CardBody className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="md:col-span-2">
              <div className="label mb-1.5">Players (up to 5)</div>
              <MultiSelect options={meta.players} value={players} onChange={setPlayers} max={5} renderAvatar={avatar} />
            </div>
            <div>
              <div className="label mb-1.5">Metric</div>
              <Select value={metric} onChange={setMetric} options={meta.metrics} placeholder="Select" />
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="btn btn-primary" onClick={fetchIt} disabled={loading || players.length === 0 || !metric}>
              {loading ? "Fetching…" : "Plot curves"}
            </button>
          </div>
          {err && <ErrorNotice message={err} onRetry={fetchIt} />}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title={metric ? `${metric} vs. age` : "—"} />
        <CardBody>
          {traces.length === 0 ? (
            <div className="grid place-items-center py-10 text-sm text-mute">
              Pick 1–5 players and press “Plot curves”.
            </div>
          ) : (
            <>
            <PlayerLegend
              names={(data?.curves ?? []).filter((c: any) => c.points?.length).map((c: any) => c.player)}
              renderAvatar={avatar}
              className="mb-1 px-1"
            />
            <Plot
              data={traces as any}
              layout={{
                showlegend: false,
                xaxis: { title: "Age at season start", gridcolor: "#1f2630" },
                yaxis: { title: metric, gridcolor: "#1f2630" },
                hovermode: "x unified",
              }}
              height={500}
            />
            </>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
