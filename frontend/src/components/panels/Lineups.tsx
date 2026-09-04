import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Slider } from "@/components/ui/Slider";
import { Plot } from "@/components/ui/Plot";
import { Avatar } from "@/components/ui/Avatar";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

// Same diverging scale the league table uses for net rating: blue and red
// through a neutral middle, so a lineup reads the same way a team does.
const DIVERGING: [number, string][] = [
  [0, "#d73027"],
  [0.5, "#6b7685"],
  [1, "#4dabff"],
];

// Picked fives are drawn in a colour the scale above never uses. Orange or
// green would land inside it and read as a rating rather than a selection.
const PICKED = "#b084ff";

type Col = {
  key: string;
  label: string;
  title: string;
  fmt: (v: number) => string;
  lowerBetter?: boolean;
};

const COLS: Col[] = [
  { key: "games", label: "G", title: "Games this five appeared in", fmt: (v) => String(v ?? "") },
  { key: "min", label: "MP", title: "Minutes played together", fmt: (v) => (v == null ? "" : v.toFixed(0)) },
  { key: "share", label: "Share", title: "Share of the team's minutes",
    fmt: (v) => (v == null ? "" : `${(v * 100).toFixed(1)}%`) },
  { key: "ortg", label: "ORtg", title: "Points scored per 100 possessions",
    fmt: (v) => (v == null ? "" : v.toFixed(1)) },
  { key: "drtg", label: "DRtg", title: "Points allowed per 100 possessions",
    fmt: (v) => (v == null ? "" : v.toFixed(1)), lowerBetter: true },
  { key: "net", label: "Net", title: "ORtg − DRtg",
    fmt: (v) => (v == null ? "" : (v > 0 ? "+" : "") + v.toFixed(1)) },
  { key: "plus_minus", label: "+/−", title: "Raw points scored minus allowed",
    fmt: (v) => (v == null ? "" : (v > 0 ? "+" : "") + v.toFixed(0)) },
];

const ALL_TEAMS = "";

/** "Shai Gilgeous-Alexander" -> "Gilgeous-Alexander", for the chart's labels. */
const surname = (name: string) => name.trim().split(/\s+/).slice(1).join(" ") || name;

/**
 * Which five played best together. ESPN publishes no lineup data, so these are
 * rebuilt from substitutions in `etl/lineup_etl.py`; the ratings are on the
 * same per-100-possessions scale as team ORtg and DRtg elsewhere.
 */
export function Lineups({ meta }: { meta: Meta }) {
  const lineupSeasons = meta.lineup_seasons ?? [];
  const [season, setSeason] = useState(lineupSeasons.at(-1) ?? "");
  const [team, setTeam] = useState<string>(ALL_TEAMS);
  const [minMinutes, setMinMinutes] = useState(100);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: "min", dir: -1 });
  // Which fives are pinned onto the chart. Keyed by the five itself so the
  // selection survives re-sorting and re-fetching.
  const [picked, setPicked] = useState<string[]>([]);

  // A team's whole rotation is a few dozen lineups; the league's is thousands,
  // so the league view needs a higher bar to stay a list a person can read.
  useEffect(() => setMinMinutes(team ? 50 : 100), [team]);

  // A different season, team or floor is a different set of fives, so a
  // selection made against the old one would highlight nothing.
  useEffect(() => setPicked([]), [season, team, minMinutes, meta.league]);

  useEffect(() => {
    if (!season) return;
    setErr(null);
    api
      .teamLineups(season, meta.league, { team: team || undefined, minMinutes })
      .then(setData)
      .catch((e) => {
        setErr(e.message);
        setData(null);
      });
  }, [season, team, minMinutes, meta.league]);

  const rows = data?.rows ?? [];
  const keyOf = (r: any) =>
    `${r.team_abbr}-${r.players.map((p: any) => p.id).join("-")}`;
  const togglePick = (r: any) =>
    setPicked((current) => {
      const k = keyOf(r);
      return current.includes(k) ? current.filter((x) => x !== k) : [...current, k];
    });

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
    // Minutes drive the marker area, not its radius: at radius the biggest
    // lineup swallows the chart.
    const maxMin = Math.max(...rows.map((r: any) => r.min));
    const byNet = [...rows].sort((a: any, b: any) => b.net - a.net);
    // With nothing picked, label the extremes of the league view by team, the
    // way the league table does. One team's own lineups all carry the same
    // abbreviation, so there the marker goes unlabelled and hover names the
    // five. Once fives are picked, they are the only thing worth labelling.
    const auto = new Set(team || picked.length ? [] : [...byNet.slice(0, 3), ...byNet.slice(-2)]);
    const isPicked = (r: any) => picked.includes(keyOf(r));
    const chosen = rows.filter(isPicked);
    const rest = picked.length ? rows.filter((r: any) => !isPicked(r)) : rows;
    const size = (r: any) => 8 + 26 * Math.sqrt(r.min / maxMin);
    const hover =
      "<b>%{hovertext}</b><br>%{customdata[0]:.0f} min over %{customdata[2]} games<br>" +
      "ORtg %{x:.1f} · DRtg %{y:.1f} · Net %{customdata[1]:+.1f}<extra></extra>";
    const describe = (r: any) =>
      `${r.team_abbr} · ${r.players.map((p: any) => surname(p.name)).join(", ")}`;
    const facts = (r: any) => [r.min, r.net, r.games];

    const base = {
      type: "scatter",
      mode: "markers+text",
      textposition: "top center",
      hovertemplate: hover,
    };
    const out: any[] = [
      {
        ...base,
        x: rest.map((r: any) => r.ortg),
        y: rest.map((r: any) => r.drtg),
        text: rest.map((r: any) => (auto.has(r) ? r.team_abbr : "")),
        textfont: { size: 10, color: "#8a94a2" },
        hovertext: rest.map(describe),
        customdata: rest.map(facts),
        marker: {
          size: rest.map(size),
          color: rest.map((r: any) => r.net),
          colorscale: DIVERGING,
          cmid: 0,
          // Picked fives sit on top of a quieted field.
          opacity: picked.length ? 0.25 : 0.85,
          line: { color: "#111518", width: 1.5 },
          colorbar: {
            title: { text: "Net", side: "right" },
            thickness: 10,
            outlinewidth: 0,
            tickfont: { color: "#8a94a2", size: 10 },
          },
        },
      },
    ];
    if (chosen.length) {
      out.push({
        ...base,
        x: chosen.map((r: any) => r.ortg),
        y: chosen.map((r: any) => r.drtg),
        text: chosen.map((r: any) =>
          r.players.map((p: any) => surname(p.name)).join(" · ")
        ),
        textfont: { size: 10, color: PICKED },
        hovertext: chosen.map(describe),
        customdata: chosen.map(facts),
        marker: {
          size: chosen.map(size),
          color: PICKED,
          opacity: 1,
          line: { color: "#111518", width: 2 },
        },
      });
    }
    return out;
  }, [rows, team, picked.join(",")]);

  const layout = useMemo(
    () => ({
      margin: { t: 16, r: 10, b: 48, l: 60 },
      showlegend: false,
      hovermode: "closest",
      xaxis: { title: "Offensive rating →", gridcolor: "#1f2630", zeroline: false },
      yaxis: {
        title: "← Defensive rating",
        gridcolor: "#1f2630",
        autorange: "reversed",
        zeroline: false,
      },
      annotations: [
        { xref: "paper", yref: "paper", x: 1, y: 1, xanchor: "right", yanchor: "top",
          text: "outscores everyone", showarrow: false,
          font: { color: "#6b7685", size: 10 } },
      ],
    }),
    []
  );

  const seasonLabel = season ? formatSeason(season, meta.season_format) : "";
  const coverage =
    data?.team_minutes && data?.shown_minutes
      ? `${rows.length} lineups, ${Math.round(
          (data.shown_minutes / data.team_minutes) * 100
        )}% of the team's minutes`
      : rows.length
      ? `${rows.length} lineups`
      : "";

  if (!lineupSeasons.length) {
    return (
      <Card>
        <CardHeader title="Lineups" />
        <CardBody>
          <div className="text-sm text-mute">
            No {meta.league_label} lineup data on disk. Build it with{" "}
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
          title="Lineups"
        />
        <CardBody>
          <div className="grid gap-3 md:grid-cols-3">
            <div>
              <div className="label mb-1.5">Season</div>
              <Select
                value={season}
                onChange={setSeason}
                options={lineupSeasons.map((s) => ({
                  value: s,
                  label: formatSeason(s, meta.season_format),
                }))}
              />
            </div>
            <div>
              <div className="label mb-1.5">Team</div>
              <Select
                value={team}
                onChange={setTeam}
                options={meta.teams}
                placeholder="Every team"
              />
            </div>
            <div>
              <div className="label mb-1.5">
                Minutes together — at least {minMinutes}
              </div>
              <Slider
                value={minMinutes}
                onChange={setMinMinutes}
                min={5}
                max={400}
                step={5}
                className="mt-2.5"
              />
            </div>
          </div>
          {err && <div className="mt-3 text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={team ? `${team} — ${seasonLabel}` : `Every five — ${seasonLabel}`}
        />
        <CardBody>
          <Plot
            data={traces as any}
            layout={layout as any}
            height={480}
            placeholder="No lineup clears this minutes floor"
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Lineup table" subtitle={coverage} />
        <CardBody className="p-0">
          {sorted.length === 0 ? (
            <div className="py-10 text-center text-sm text-mute">
              No lineup played {minMinutes} minutes together.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-mute">
                    <th className="px-4 py-2 font-medium">Lineup</th>
                    {COLS.map((c) => (
                      <th key={c.key} className="px-3 py-2 text-right font-medium">
                        <button
                          title={c.title}
                          onClick={() =>
                            setSort((s) =>
                              s.key === c.key
                                ? { key: c.key, dir: (s.dir * -1) as 1 | -1 }
                                : { key: c.key, dir: c.lowerBetter ? 1 : -1 }
                            )
                          }
                          className={cn("hover:text-ink", sort.key === c.key && "text-accent")}
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
                    <tr
                      key={keyOf(r)}
                      onClick={() => togglePick(r)}
                      title="Show this five on the chart"
                      className={cn(
                        "cursor-pointer border-t border-border/60 hover:bg-border/30",
                        // Matches the chart's pick colour, not the accent.
                        picked.includes(keyOf(r)) && "bg-[#b084ff]/15"
                      )}
                    >
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-2">
                          {!team && (
                            <span className="w-9 shrink-0 text-xs text-mute">{r.team_abbr}</span>
                          )}
                          <div className="flex gap-1">
                            {r.players.map((p: any) => (
                              <Avatar
                                key={p.id}
                                name={p.name}
                                id={p.id}
                                league={meta.league}
                                size={30}
                              />
                            ))}
                          </div>
                          <span className="min-w-0 truncate">
                            {r.players.map((p: any) => surname(p.name)).join(" · ")}
                          </span>
                        </div>
                      </td>
                      {COLS.map((c) => (
                        <td key={c.key} className="px-3 py-2 text-right tabular-nums">
                          {c.fmt(r[c.key])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
