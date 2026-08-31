import { useEffect, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

type Pick = { team: string; season: string };

const ROWS: { key: string; label: string; fmt: (v: number) => string; lowerBetter?: boolean }[] = [
  { key: "wins", label: "Wins", fmt: (v) => String(Math.round(v)) },
  { key: "losses", label: "Losses", fmt: (v) => String(Math.round(v)), lowerBetter: true },
  { key: "win_pct", label: "Win %", fmt: (v) => v.toFixed(3).replace(/^0/, "") },
  { key: "ortg", label: "Offensive rating", fmt: (v) => v.toFixed(1) },
  { key: "drtg", label: "Defensive rating", fmt: (v) => v.toFixed(1), lowerBetter: true },
  { key: "net", label: "Net rating", fmt: (v) => (v > 0 ? "+" : "") + v.toFixed(1) },
  { key: "pace", label: "Pace", fmt: (v) => v.toFixed(1) },
  { key: "eFG%", label: "eFG%", fmt: (v) => v.toFixed(3) },
  { key: "TOV%", label: "TOV%", fmt: (v) => v.toFixed(3), lowerBetter: true },
  { key: "ORB%", label: "ORB%", fmt: (v) => v.toFixed(3) },
  { key: "FT rate", label: "FT rate", fmt: (v) => v.toFixed(3) },
];

export function TeamMatchup({ meta }: { meta: Meta }) {
  const [a, setA] = useState<Pick>({ team: "", season: "" });
  const [b, setB] = useState<Pick>({ team: "", season: "" });
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!a.team || !a.season || !b.team || !b.season) {
      setData(null);
      return;
    }
    setErr(null);
    api
      .teamsCompare(a, b, meta.league)
      .then(setData)
      .catch((e) => {
        setErr(e.message);
        setData(null);
      });
  }, [a.team, a.season, b.team, b.season, meta.league]);

  const ffTraces = data
    ? ["eFG%", "TOV%", "ORB%", "FT rate"].length && [
        {
          type: "bar",
          name: `${formatSeason(data.a.season, meta.season_format)} ${data.a.team}`,
          x: ["eFG%", "TOV%", "ORB%", "FT rate"],
          y: ["eFG%", "TOV%", "ORB%", "FT rate"].map((k) => data.a.values[k]),
          marker: { color: "#ff6a3d" },
        },
        {
          type: "bar",
          name: `${formatSeason(data.b.season, meta.season_format)} ${data.b.team}`,
          x: ["eFG%", "TOV%", "ORB%", "FT rate"],
          y: ["eFG%", "TOV%", "ORB%", "FT rate"].map((k) => data.b.values[k]),
          marker: { color: "#4dabff" },
        },
      ]
    : [];

  const side = (p: Pick, set: (v: Pick) => void, label: string) => (
    <div className="grid gap-3 md:grid-cols-2">
      <div>
        <div className="label mb-1.5">{label}</div>
        <Select value={p.team} onChange={(v) => set({ ...p, team: v })} options={meta.teams} placeholder="Select" />
      </div>
      <div>
        <div className="label mb-1.5">Season</div>
        <Select
          value={p.season}
          onChange={(v) => set({ ...p, season: v })}
          options={meta.seasons.map((s) => ({ value: s, label: formatSeason(s, meta.season_format) }))}
          placeholder="Select"
        />
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Team matchup" />
        <CardBody className="space-y-3">
          {side(a, setA, "Team")}
          {side(b, setB, "Against")}
          {err && <div className="text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Side by side" />
          <CardBody className="p-0">
            {!data ? (
              <div className="px-5 py-8 text-sm text-mute">Pick two team-seasons.</div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-mute">
                    <th className="px-4 py-2 font-medium">Metric</th>
                    <th className="px-3 py-2 text-right font-medium">
                      {formatSeason(data.a.season, meta.season_format)} {data.a.team}
                    </th>
                    <th className="px-4 py-2 text-right font-medium">
                      {formatSeason(data.b.season, meta.season_format)} {data.b.team}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {ROWS.map((r) => {
                    const av = data.a.values[r.key];
                    const bv = data.b.values[r.key];
                    const better =
                      av == null || bv == null
                        ? null
                        : r.lowerBetter
                        ? av < bv
                          ? "a"
                          : av > bv
                          ? "b"
                          : null
                        : av > bv
                        ? "a"
                        : av < bv
                        ? "b"
                        : null;
                    return (
                      <tr key={r.key} className="border-t border-border/60">
                        <td className="px-4 py-2 text-mute">{r.label}</td>
                        {(["a", "b"] as const).map((k) => {
                          const v = k === "a" ? av : bv;
                          const vs = data[k].vs_league?.[r.key];
                          return (
                            <td
                              key={k}
                              className={cn(
                                "px-3 py-2 text-right tabular-nums",
                                k === "b" && "pr-4",
                                better === k && "text-accent"
                              )}
                            >
                              <div>{v == null ? "—" : r.fmt(v)}</div>
                              {vs != null && (
                                <div className="text-[11px] text-mute">
                                  {vs >= 0 ? "+" : "−"}
                                  {Math.abs(vs).toFixed(r.key === "win_pct" || r.key.includes("%") || r.key === "FT rate" ? 3 : 1)} vs lg
                                </div>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Four Factors" />
          <CardBody>
            <Plot
              data={(ffTraces || []) as any}
              layout={{
                barmode: "group",
                margin: { t: 20 },
                xaxis: { type: "category" },
                yaxis: { title: "Rate" },
                legend: { orientation: "h", y: -0.2 },
              }}
              height={380}
              placeholder="Pick two team-seasons"
            />
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
