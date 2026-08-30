import { useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { PlayerCombobox } from "@/components/ui/PlayerCombobox";
import { Select } from "@/components/ui/Select";
import { Slider } from "@/components/ui/Slider";
import { Plot } from "@/components/ui/Plot";
import { ErrorNotice } from "@/components/ui/ErrorNotice";

const STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "FG_PCT", "FG3_PCT"];

export function GameLog({ meta }: { meta: Meta }) {
  const [player, setPlayer] = useState("Stephen Curry");
  const [season, setSeason] = useState(meta.seasons[meta.seasons.length - 1] ?? "");
  const [stat, setStat] = useState("PTS");
  const [window, setWindow] = useState(10);
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function fetchIt() {
    setLoading(true);
    setErr(null);
    try {
      const d = await api.gamelog(player, season, stat, window, meta.league);
      setData(d);
    } catch (e: any) {
      setErr(e.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  const traces = useMemo(() => {
    if (!data?.games?.length) return [];
    return [
      {
        type: "bar",
        name: `Per-game ${stat}`,
        x: data.games.map((g: any) => g.date),
        y: data.games.map((g: any) => g.value),
        opacity: 0.55,
        marker: { color: "#4dabff" },
        hovertemplate: "%{x|%b %d}: %{y}<br>%{customdata}<extra></extra>",
        customdata: data.games.map((g: any) => g.matchup ?? ""),
      },
      {
        type: "scatter",
        mode: "lines",
        name: `${window}-game avg`,
        x: data.games.map((g: any) => g.date),
        y: data.games.map((g: any) => g.rolling),
        line: { color: "#ff6a3d", width: 3 },
      },
    ];
  }, [data, stat, window]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Game log explorer"
          subtitle="Per-game line with a rolling average"
        />
        <CardBody className="space-y-4">
          <div className="grid gap-3 md:grid-cols-4">
            <div>
              <div className="label mb-1.5">Player</div>
              <PlayerCombobox options={meta.players} value={player} onChange={setPlayer} />
            </div>
            <div>
              <div className="label mb-1.5">Season</div>
              <Select value={season} onChange={setSeason} options={meta.seasons} />
            </div>
            <div>
              <div className="label mb-1.5">Stat</div>
              <Select value={stat} onChange={setStat} options={STATS} />
            </div>
            <div>
              <div className="label mb-1.5">Rolling window: {window}</div>
              <Slider value={window} onChange={setWindow} min={3} max={20} />
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="btn btn-primary" onClick={fetchIt} disabled={loading}>
              {loading ? "Fetching…" : "Fetch game log"}
            </button>
            {data?.season_avg != null && (
              <div className="text-sm text-mute">
                {data.games.length} games · season avg {data.season_avg.toFixed(2)}
              </div>
            )}
          </div>
          {err && <ErrorNotice message={err} onRetry={fetchIt} />}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title={data ? `${data.player} — ${data.season} — ${data.stat}` : "—"} />
        <CardBody>
          {!traces.length ? (
            <div className="grid place-items-center py-10 text-sm text-mute">
              Press “Fetch game log”.
            </div>
          ) : (
            <Plot
              data={traces as any}
              layout={{
                xaxis: { title: "Game date", type: "date", gridcolor: "#1f2630" },
                yaxis: { title: stat, gridcolor: "#1f2630" },
                hovermode: "x unified",
                bargap: 0.05,
              }}
              height={480}
            />
          )}
        </CardBody>
      </Card>
    </div>
  );
}
