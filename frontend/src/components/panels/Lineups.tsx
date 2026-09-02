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

  // A team's whole rotation is a few dozen lineups; the league's is thousands,
  // so the league view needs a higher bar to stay a list a person can read.
  useEffect(() => setMinMinutes(team ? 50 : 100), [team]);

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
    // Label the extremes of the league view by team, the way the league table
    // does. One team's own lineups all carry the same abbreviation, so there
    // the marker goes unlabelled and hover names the five.
    const byNet = [...rows].sort((a: any, b: any) => b.net - a.net);
    const labelled = new Set(team ? [] : [...byNet.slice(0, 3), ...byNet.slice(-2)]);
    return [
      {
        type: "scatter",
        mode: "markers+text",
        x: rows.map((r: any) => r.ortg),
        y: rows.map((r: any) => r.drtg),
        text: rows.map((r: any) => (labelled.has(r) ? r.team_abbr : "")),
        textposition: "top center",
        textfont: { size: 10, color: "#8a94a2" },
        hovertext: rows.map((r: any) =>
          `${r.team_abbr} · ${r.players.map((p: any) => surname(p.name)).join(", ")}`
        ),
        customdata: rows.map((r: any) => [r.min, r.net, r.games]),
        hovertemplate:
          "<b>%{hovertext}</b><br>%{customdata[0]:.0f} min over %{customdata[2]} games<br>" +
          "ORtg %{x:.1f} · DRtg %{y:.1f} · Net %{customdata[1]:+.1f}<extra></extra>",
        marker: {
          size: rows.map((r: any) => 8 + 26 * Math.sqrt(r.min / maxMin)),
          color: rows.map((r: any) => r.net),
          colorscale: DIVERGING,
          cmid: 0,
          opacity: 0.85,
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
  }, [rows, team]);

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
          subtitle="Rebuilt from play-by-play substitutions — who was actually on the floor together, and what happened while they were."
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
          subtitle="Bubble size is minutes played together."
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
                      key={`${r.team_abbr}-${r.players.map((p: any) => p.id).join("-")}`}
                      className="border-t border-border/60 hover:bg-border/30"
                    >
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-2">
                          {!team && (
                            <span className="w-9 shrink-0 text-xs text-mute">{r.team_abbr}</span>
                          )}
                          <div className="flex -space-x-1.5">
                            {r.players.map((p: any) => (
                              <Avatar
                                key={p.id}
                                name={p.name}
                                id={p.id}
                                league={meta.league}
                                size={24}
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

      <div className="px-1 text-xs text-mute">
        Substitutions are the only record of who was on the floor, and ESPN's are
        not perfect. Rebuilt minutes land within a minute of the box score for
        99.5% of player-games in the latest NBA season but 92% in 2015, and the
        seasons before each league's first one here are worse still — which is
        where the list of seasons stops. Fives that played under five minutes
        together aren't stored.
      </div>
    </div>
  );
}
