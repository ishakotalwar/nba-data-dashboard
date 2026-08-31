import { useEffect, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { formatSeason } from "@/lib/season";

const METRICS = [
  { value: "net", label: "Net rating" },
  { value: "win_pct", label: "Win %" },
  { value: "wins", label: "Wins" },
  { value: "ortg", label: "Offensive rating" },
  { value: "drtg", label: "Defensive rating (lowest)" },
  { value: "eFG%", label: "eFG%" },
  { value: "TOV%", label: "Turnover rate (lowest)" },
  { value: "ORB%", label: "Offensive rebound rate" },
  { value: "FT rate", label: "Free throw rate" },
  { value: "pace", label: "Pace" },
];

const fmt = (metric: string, v: number | null) => {
  if (v == null) return "—";
  if (metric === "win_pct") return v.toFixed(3).replace(/^0/, "");
  if (["eFG%", "TOV%", "ORB%", "FT rate"].includes(metric)) return v.toFixed(3);
  if (metric === "wins") return String(Math.round(v));
  return v.toFixed(1);
};

export function TeamRankings({ meta }: { meta: Meta }) {
  const [metric, setMetric] = useState("net");
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    api.teamsRankings(metric, meta.league, 15).then(setData).catch(() => setData(null));
  }, [metric, meta.league]);

  const rows = data?.rows ?? [];
  const span = meta.seasons.length
    ? `${formatSeason(meta.seasons[0], meta.season_format)}–${formatSeason(meta.seasons[meta.seasons.length - 1], meta.season_format)}`
    : "";

  return (
    <Card>
      <CardHeader
        title="All-time leaders"
        right={
          <div className="w-56">
            <Select value={metric} onChange={setMetric} options={METRICS} />
          </div>
        }
      />
      <CardBody className="p-0">
        {rows.length === 0 ? (
          <div className="px-5 py-8 text-sm text-mute">No data.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wider text-mute">
                  <th className="px-4 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Team-season</th>
                  <th className="px-3 py-2 text-right font-medium">
                    {METRICS.find((m) => m.value === metric)?.label}
                  </th>
                  <th className="px-3 py-2 text-right font-medium">W–L</th>
                  <th className="px-3 py-2 text-right font-medium">ORtg</th>
                  <th className="px-4 py-2 text-right font-medium">DRtg</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r: any, i: number) => (
                  <tr key={`${r.team}-${r.season}`} className="border-t border-border/60">
                    <td className="px-4 py-2 text-mute tabular-nums">{i + 1}</td>
                    <td className="whitespace-nowrap px-3 py-2">
                      {formatSeason(r.season, meta.season_format)} {r.team}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-accent">
                      {fmt(metric, r[metric])}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {Math.round(r.wins)}–{Math.round(r.losses)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">{r.ortg?.toFixed(1)}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{r.drtg?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardBody>
    </Card>
  );
}
