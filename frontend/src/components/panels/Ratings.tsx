import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Slider } from "@/components/ui/Slider";
import { Plot } from "@/components/ui/Plot";
import { Avatar } from "@/components/ui/Avatar";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

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
};

const signed = (v: number, digits = 1) =>
  v == null ? "" : (v > 0 ? "+" : "") + v.toFixed(digits);

const COLS: Col[] = [
  { key: "games", label: "G", title: "Games played", fmt: (v) => String(v ?? "") },
  { key: "min", label: "MP", title: "Minutes on the floor", fmt: (v) => (v == null ? "" : v.toFixed(0)) },
  { key: "poss", label: "Poss", title: "Possessions played", fmt: (v) => (v == null ? "" : v.toFixed(0)) },
  { key: "rapm", label: "RAPM", title: "Points per 100 possessions credited to this player once the other nine on the floor are regressed out", fmt: (v) => signed(v, 2) },
  { key: "on_net", label: "On", title: "Raw team net rating while he was on the floor", fmt: (v) => signed(v) },
  { key: "off_net", label: "Off", title: "Raw team net rating while he sat", fmt: (v) => signed(v) },
  { key: "on_off", label: "On/Off", title: "On minus off — unadjusted, so it carries his teammates with it", fmt: (v) => signed(v) },
  { key: "plus_minus", label: "+/−", title: "Raw points scored minus allowed while on the floor", fmt: (v) => signed(v, 0) },
];

/** "Shai Gilgeous-Alexander" -> "Gilgeous-Alexander". */
const surname = (name: string) => name.trim().split(/\s+/).slice(1).join(" ") || name;

/**
 * Who was worth the most, in points per 100 possessions. The number is RAPM,
 * rebuilt from substitutions in `etl/lineup_etl.py` — raw plus-minus describes
 * a player's teammates as much as the player, and the regression is what pulls
 * those apart.
 */
export function Ratings({ meta }: { meta: Meta }) {
  const ratingSeasons = meta.rating_seasons ?? [];
  const [season, setSeason] = useState(ratingSeasons.at(-1) ?? "");
  const [team, setTeam] = useState("");
  const [minPoss, setMinPoss] = useState(1000);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: "rapm", dir: -1 });

  useEffect(() => {
    if (!season) return;
    setErr(null);
    api
      .playerRatings(season, meta.league, { team: team || undefined, minPoss, limit: 250 })
      .then(setData)
      .catch((e) => {
        setErr(e.message);
        setData(null);
      });
  }, [season, team, minPoss, meta.league]);

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

  // The adjustment is the story: how far each player sits from the raw on-off
  // he would have been credited with.
  const traces = useMemo(() => {
    if (!rows.length) return [];
    const top = [...rows].sort((a: any, b: any) => b.rapm - a.rapm).slice(0, 5);
    const labelled = new Set(top);
    return [
      {
        type: "scatter",
        mode: "markers+text",
        x: rows.map((r: any) => r.on_off),
        y: rows.map((r: any) => r.rapm),
        text: rows.map((r: any) => (labelled.has(r) ? surname(r.player_name) : "")),
        textposition: "top center",
        textfont: { size: 10, color: "#8a94a2" },
        hovertext: rows.map((r: any) => r.player_name),
        customdata: rows.map((r: any) => [r.team_abbr, r.poss]),
        hovertemplate:
          "<b>%{hovertext}</b> (%{customdata[0]})<br>" +
          "RAPM %{y:+.2f} · on/off %{x:+.1f}<br>%{customdata[1]:.0f} possessions<extra></extra>",
        marker: {
          size: 9,
          color: rows.map((r: any) => r.rapm),
          colorscale: DIVERGING,
          cmid: 0,
          line: { color: "#111518", width: 1 },
        },
      },
    ];
  }, [rows]);

  const layout = useMemo(
    () => ({
      margin: { t: 16, r: 16, b: 48, l: 56 },
      showlegend: false,
      hovermode: "closest",
      xaxis: { title: "Raw on/off →", gridcolor: "#1f2630", zeroline: true, zerolinecolor: "#3a4250" },
      yaxis: { title: "RAPM →", gridcolor: "#1f2630", zeroline: true, zerolinecolor: "#3a4250" },
    }),
    []
  );

  const seasonLabel = season ? formatSeason(season, meta.season_format) : "";

  if (!ratingSeasons.length) {
    return (
      <Card>
        <CardHeader title="Impact" />
        <CardBody>
          <div className="text-sm text-mute">
            No {meta.league_label} rating data on disk. Build it with{" "}
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
          title="Impact"
          subtitle="Points per 100 possessions each player is worth, once the other nine on the floor are regressed out of the margin."
        />
        <CardBody>
          <div className="grid gap-3 md:grid-cols-3">
            <div>
              <div className="label mb-1.5">Season</div>
              <Select
                value={season}
                onChange={setSeason}
                options={ratingSeasons.map((s) => ({
                  value: s,
                  label: formatSeason(s, meta.season_format),
                }))}
              />
            </div>
            <div>
              <div className="label mb-1.5">Team</div>
              <Select value={team} onChange={setTeam} options={meta.teams} placeholder="Every team" />
            </div>
            <div>
              <div className="label mb-1.5">
                Possessions played — at least {minPoss.toLocaleString()}
              </div>
              <Slider
                value={minPoss}
                onChange={setMinPoss}
                min={100}
                max={4000}
                step={100}
                className="mt-2.5"
              />
            </div>
          </div>
          {err && <div className="mt-3 text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={`What the adjustment does — ${seasonLabel}`}
          subtitle="Raw on/off across, adjusted rating up. Players far below the diagonal were carried by who they played with."
        />
        <CardBody>
          <Plot
            data={traces as any}
            layout={layout as any}
            height={420}
            placeholder="No player clears this possession floor"
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={team ? `${team} — ${seasonLabel}` : `Best players — ${seasonLabel}`}
          subtitle={
            data ? `${data.qualified} of ${data.pool} players clear ${minPoss.toLocaleString()} possessions` : undefined
          }
        />
        <CardBody className="p-0">
          {sorted.length === 0 ? (
            <div className="py-10 text-center text-sm text-mute">
              No player played {minPoss.toLocaleString()} possessions.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-mute">
                    <th className="px-4 py-2 font-medium">Player</th>
                    {COLS.map((c) => (
                      <th key={c.key} className="px-3 py-2 text-right font-medium">
                        <button
                          title={c.title}
                          onClick={() =>
                            setSort((s) =>
                              s.key === c.key
                                ? { key: c.key, dir: (s.dir * -1) as 1 | -1 }
                                : { key: c.key, dir: -1 }
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
                  {sorted.map((r: any, i: number) => (
                    <tr key={r.player_id} className="border-t border-border/60 hover:bg-border/30">
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-2.5">
                          <span className="w-6 shrink-0 text-right text-xs tabular-nums text-mute">
                            {sort.key === "rapm" && sort.dir === -1 ? i + 1 : ""}
                          </span>
                          <Avatar
                            name={r.player_name}
                            id={r.player_id}
                            league={meta.league}
                            size={24}
                          />
                          <span className="truncate">{r.player_name}</span>
                          {!team && <span className="text-xs text-mute">{r.team_abbr}</span>}
                        </div>
                      </td>
                      {COLS.map((c) => (
                        <td
                          key={c.key}
                          className={cn(
                            "px-3 py-2 text-right tabular-nums",
                            c.key === "rapm" && "font-medium"
                          )}
                        >
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
