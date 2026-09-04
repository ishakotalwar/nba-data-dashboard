import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

const signed = (v: number | null, digits = 1) =>
  v == null ? "" : (v > 0 ? "+" : "") + v.toFixed(digits);

const COLS = [
  { key: "min", label: "MP", title: "Minutes", fmt: (v: number) => v?.toFixed(0) ?? "" },
  { key: "poss", label: "Poss", title: "Possessions", fmt: (v: number) => v?.toFixed(0) ?? "" },
  { key: "ortg", label: "ORtg", title: "Points scored per 100 possessions", fmt: (v: number) => v?.toFixed(1) ?? "" },
  { key: "drtg", label: "DRtg", title: "Points allowed per 100 possessions", fmt: (v: number) => v?.toFixed(1) ?? "" },
  { key: "net", label: "Net", title: "ORtg − DRtg", fmt: (v: number) => signed(v), strong: true },
];

/**
 * What a team did with a player on the floor, and what it did without him.
 *
 * Deliberately unadjusted: this is the raw split, teammates and opponents
 * included, which answers "what happened when he sat" rather than "how good is
 * he". The Impact page carries the adjusted version, and the two disagreeing is
 * usually the interesting part — a player whose team collapses without him may
 * only be telling you about his backup.
 */
export function WithOrWithout({ meta }: { meta: Meta }) {
  const seasons = meta.lineup_seasons ?? [];
  const [season, setSeason] = useState(seasons.at(-1) ?? "");
  const [team, setTeam] = useState("");
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  // Roster first, with no player picked, so the pickers can be filled.
  useEffect(() => {
    if (!season || !team) {
      setData(null);
      return;
    }
    setErr(null);
    setA("");
    setB("");
    api
      .teamWowy(season, team, meta.league)
      .then(setData)
      .catch((e) => {
        setErr(e.message);
        setData(null);
      });
  }, [season, team, meta.league]);

  useEffect(() => {
    if (!season || !team || !a) return;
    setErr(null);
    api
      .teamWowy(season, team, meta.league, { playerA: Number(a), playerB: b ? Number(b) : undefined })
      .then(setData)
      .catch((e) => setErr(e.message));
  }, [a, b]);

  const roster = data?.roster ?? [];
  const rows = data?.rows ?? [];
  const total = data?.team_total;

  const options = useMemo(
    () =>
      roster.map((p: any) => ({
        value: String(p.player_id),
        label: `${p.name} — ${p.min.toFixed(0)} min`,
      })),
    [roster]
  );

  const traces = useMemo(() => {
    if (!rows.length) return [];
    return [
      {
        type: "bar",
        orientation: "h",
        x: rows.map((r: any) => r.net),
        y: rows.map((r: any) => r.label),
        // No in-bar labels: a negative bar's text runs off the left edge, and
        // the table beside this one already carries every number.
        hovertemplate:
          "<b>%{y}</b><br>net %{x:+.1f} per 100<br>%{customdata:.0f} possessions<extra></extra>",
        customdata: rows.map((r: any) => r.poss),
        marker: { color: rows.map((r: any) => (r.net >= 0 ? "#4dabff" : "#d73027")) },
      },
    ];
  }, [rows]);

  const layout = useMemo(
    () => ({
      margin: { t: 10, r: 16, b: 42, l: 190 },
      showlegend: false,
      // Longest split first reads top-down like the table beside it.
      yaxis: { autorange: "reversed", automargin: true },
      xaxis: {
        title: "Net rating per 100 possessions",
        gridcolor: "#1f2630",
        zeroline: true,
        zerolinecolor: "#3a4250",
        zerolinewidth: 2,
      },
    }),
    []
  );

  const seasonLabel = season ? formatSeason(season, meta.season_format) : "";

  if (!seasons.length) {
    return (
      <Card>
        <CardHeader title="With or without" />
        <CardBody>
          <div className="text-sm text-mute">
            No {meta.league_label} stint data on disk. Build it with{" "}
            <code className="bg-border/60 px-1.5 py-0.5">
              python etl/lineup_etl.py --league {meta.league}
            </code>
            .
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="With or without"
          subtitle="What the team did with a player on the floor, and what it did while he sat. Pick a second player to split it four ways."
        />
        <CardBody>
          <div className="grid gap-3 md:grid-cols-4">
            <div>
              <div className="label mb-1.5">Season</div>
              <Select
                value={season}
                onChange={setSeason}
                options={seasons.map((s) => ({
                  value: s,
                  label: formatSeason(s, meta.season_format),
                }))}
              />
            </div>
            <div>
              <div className="label mb-1.5">Team</div>
              <Select value={team} onChange={setTeam} options={meta.teams} placeholder="Select" />
            </div>
            <div>
              <div className="label mb-1.5">Player</div>
              <Select
                value={a}
                onChange={setA}
                options={options}
                placeholder={team ? "Select" : "Pick a team first"}
              />
            </div>
            <div>
              <div className="label mb-1.5">And (optional)</div>
              <Select
                value={b}
                onChange={setB}
                options={options.filter((o: any) => o.value !== a)}
                placeholder={a ? "Nobody" : "Pick a player first"}
              />
            </div>
          </div>
          {err && <div className="mt-3 text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      {rows.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader
              title={`The split — ${team}, ${seasonLabel}`}
              subtitle={
                total ? `The whole season is ${total.min.toFixed(0)} minutes at ${signed(total.net)}` : undefined
              }
            />
            <CardBody className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wider text-mute">
                      <th className="px-4 py-2 font-medium">Floor time</th>
                      {COLS.map((c) => (
                        <th key={c.key} title={c.title} className="px-3 py-2 text-right font-medium">
                          {c.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r: any) => (
                      <tr key={r.label} className="border-t border-border/60">
                        <td className="px-4 py-2">{r.label}</td>
                        {COLS.map((c) => (
                          <td
                            key={c.key}
                            className={cn(
                              "px-3 py-2 text-right tabular-nums",
                              c.strong && "font-medium",
                              c.strong && r.net != null && (r.net >= 0 ? "text-good" : "text-bad")
                            )}
                          >
                            {c.fmt(r[c.key])}
                          </td>
                        ))}
                      </tr>
                    ))}
                    {total && (
                      <tr className="border-t border-border bg-border/20 text-mute">
                        <td className="px-4 py-2 font-medium">{total.label} overall</td>
                        {COLS.map((c) => (
                          <td key={c.key} className="px-3 py-2 text-right tabular-nums">
                            {c.fmt(total[c.key])}
                          </td>
                        ))}
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Net rating by split" />
            <CardBody>
              <Plot data={traces as any} layout={layout as any} height={300} />
            </CardBody>
          </Card>
        </div>
      )}
    </div>
  );
}
