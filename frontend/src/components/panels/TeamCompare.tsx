import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

// Net rating is polarity data (better/worse than break-even), so it gets a
// diverging scale: two poles through a neutral grey midpoint, centered on 0.
// Blue/red rather than green/red — green/red is the one pair red-green color
// blindness cannot separate.
const DIVERGING: [number, string][] = [
  [0, "#d73027"],
  [0.5, "#6b7685"],
  [1, "#4dabff"],
];

type Col = {
  key: string;
  label: string;
  fmt: (v: number) => string;
  /** true when a smaller number is the better number (drtg, TOV%) */
  lowerBetter?: boolean;
};

const COLS: Col[] = [
  { key: "wins", label: "W", fmt: (v) => String(v ?? "") },
  { key: "losses", label: "L", fmt: (v) => String(v ?? "") },
  { key: "win_pct", label: "Win%", fmt: (v) => (v == null ? "" : v.toFixed(3).replace(/^0/, "")) },
  { key: "ortg", label: "ORtg", fmt: (v) => (v == null ? "" : v.toFixed(1)) },
  { key: "drtg", label: "DRtg", fmt: (v) => (v == null ? "" : v.toFixed(1)), lowerBetter: true },
  { key: "net", label: "Net", fmt: (v) => (v == null ? "" : (v > 0 ? "+" : "") + v.toFixed(1)) },
  { key: "pace", label: "Pace", fmt: (v) => (v == null ? "" : v.toFixed(1)) },
  { key: "eFG%", label: "eFG%", fmt: (v) => (v == null ? "" : v.toFixed(3)) },
  { key: "TOV%", label: "TOV%", fmt: (v) => (v == null ? "" : v.toFixed(3)), lowerBetter: true },
  { key: "ORB%", label: "ORB%", fmt: (v) => (v == null ? "" : v.toFixed(3)) },
  { key: "FT rate", label: "FT rate", fmt: (v) => (v == null ? "" : v.toFixed(3)) },
];

export function TeamCompare({ meta }: { meta: Meta }) {
  const [season, setSeason] = useState("");
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: "net", dir: -1 });

  useEffect(() => {
    if (!season) {
      setData(null);
      return;
    }
    setErr(null);
    api
      .teamsLeague(season, meta.league)
      .then(setData)
      .catch((e) => {
        setErr(e.message);
        setData(null);
      });
  }, [season, meta.league]);

  const rows = data?.rows ?? [];
  const avg = data?.league_avg;

  const sorted = useMemo(() => {
    const r = [...rows];
    r.sort((a, b) => {
      const x = a[sort.key], y = b[sort.key];
      if (x == null) return 1;
      if (y == null) return -1;
      return x === y ? 0 : (x < y ? -1 : 1) * sort.dir;
    });
    return r;
  }, [rows, sort]);

  const traces = useMemo(() => {
    if (!rows.length) return [];
    // Label only the extremes. Naming all 30 teams collides into mush; the
    // table underneath carries full identity, and hover covers the rest.
    const byNet = [...rows].sort((a: any, b: any) => b.net - a.net);
    const labelled = new Set(
      [...byNet.slice(0, 3), ...byNet.slice(-3)].map((r: any) => r.team)
    );
    return [
      {
        type: "scatter",
        mode: "markers+text",
        x: rows.map((r: any) => r.ortg),
        y: rows.map((r: any) => r.drtg),
        text: rows.map((r: any) => (labelled.has(r.team) ? r.team : "")),
        textposition: "top center",
        textfont: { size: 10, color: "#8a94a2" },
        hovertext: rows.map((r: any) => r.team),
        customdata: rows.map((r: any) => [r.wins, r.losses, r.net, r.pace]),
        hovertemplate:
          "<b>%{hovertext}</b><br>%{customdata[0]}–%{customdata[1]}<br>" +
          "ORtg %{x:.1f} · DRtg %{y:.1f}<br>Net %{customdata[2]:+.1f} · Pace %{customdata[3]:.1f}<extra></extra>",
        marker: {
          size: 15,
          color: rows.map((r: any) => r.net),
          colorscale: DIVERGING,
          cmid: 0,
          // A ring in the surface color keeps overlapping teams separable.
          line: { color: "#111518", width: 2 },
          colorbar: {
            title: { text: "Net", side: "right" },
            thickness: 10,
            outlinewidth: 0,
            tickfont: { color: "#8a94a2", size: 10 },
          },
        },
      },
    ];
  }, [rows]);

  const layout = useMemo(() => {
    // Axis scaffolding is returned even with no rows, so the empty chart still
    // shows what it is going to plot.
    const base: any = {
      margin: { t: 16, r: 10, b: 48, l: 60 },
      showlegend: false,
      hovermode: "closest",
      // Defense improves downward, so invert it: up and to the right is better.
      xaxis: { title: "Offensive rating →", gridcolor: "#1f2630", zeroline: false },
      yaxis: { title: "← Defensive rating", gridcolor: "#1f2630", autorange: "reversed", zeroline: false },
      annotations: [
        { xref: "paper", yref: "paper", x: 1, y: 1, xanchor: "right", yanchor: "top",
          text: "good offense · good defense", showarrow: false,
          font: { color: "#6b7685", size: 10 } },
        { xref: "paper", yref: "paper", x: 0, y: 0, xanchor: "left", yanchor: "bottom",
          text: "poor offense · poor defense", showarrow: false,
          font: { color: "#6b7685", size: 10 } },
      ],
    };
    if (avg) {
      base.shapes = [
        { type: "line", x0: avg.ortg, x1: avg.ortg, yref: "paper", y0: 0, y1: 1,
          line: { color: "#3a4250", width: 1, dash: "dot" } },
        { type: "line", y0: avg.drtg, y1: avg.drtg, xref: "paper", x0: 0, x1: 1,
          line: { color: "#3a4250", width: 1, dash: "dot" } },
      ];
    }
    return base;
  }, [rows, avg]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="League table" />
        <CardBody>
          <div className="grid gap-3 md:grid-cols-3">
            <div>
              <div className="label mb-1.5">Season</div>
              <Select
                value={season}
                onChange={setSeason}
                options={meta.seasons.map((s) => ({ value: s, label: formatSeason(s, meta.season_format) }))}
                placeholder="Select"
              />
            </div>
          </div>
          {err && <div className="mt-3 text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={season ? `Offense vs. defense — ${formatSeason(season, meta.season_format)}` : "Offense vs. defense"}
        />
        <CardBody>
          <Plot data={traces as any} layout={layout} height={480} placeholder="Select a season" />
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={season ? `Every team — ${formatSeason(season, meta.season_format)}` : "Every team"}
        />
        <CardBody className="p-0">
          {sorted.length === 0 ? (
            <div className="py-10 text-center text-sm text-mute">No data.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-mute">
                    <th className="px-4 py-2 font-medium">Team</th>
                    {COLS.map((c) => (
                      <th key={c.key} className="px-3 py-2 text-right font-medium">
                        <button
                          onClick={() =>
                            setSort((s) =>
                              s.key === c.key
                                ? { key: c.key, dir: (s.dir * -1) as 1 | -1 }
                                : { key: c.key, dir: c.lowerBetter ? 1 : -1 }
                            )
                          }
                          className={cn(
                            "hover:text-ink",
                            sort.key === c.key && "text-accent"
                          )}
                        >
                          {c.label}
                          {sort.key === c.key ? (sort.dir === -1 ? " ↓" : " ↑") : ""}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((r: any) => (
                    <tr key={r.team} className="border-t border-border/60 hover:bg-border/30">
                      <td className="whitespace-nowrap px-4 py-2">{r.team}</td>
                      {COLS.map((c) => (
                        <td key={c.key} className="px-3 py-2 text-right tabular-nums">
                          {c.fmt(r[c.key])}
                        </td>
                      ))}
                    </tr>
                  ))}
                  {avg && (
                    <tr className="border-t border-border bg-border/20 text-mute">
                      <td className="whitespace-nowrap px-4 py-2 font-medium">League average</td>
                      {COLS.map((c) => (
                        <td key={c.key} className="px-3 py-2 text-right tabular-nums">
                          {avg[c.key] != null ? c.fmt(avg[c.key]) : ""}
                        </td>
                      ))}
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
