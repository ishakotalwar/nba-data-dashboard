import { useEffect, useMemo, useState } from "react";
import { api, type Meta, type PlayerInfo, type PlayerSeason } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Plot } from "@/components/ui/Plot";
import { Select } from "@/components/ui/Select";
import { Avatar } from "@/components/ui/Avatar";
import { PlayerSeasonSelector, emptySelection } from "@/components/ui/PlayerSeasonSelector";
import { formatValue, label, ordinal, shortLabel, sortMetrics } from "@/lib/metrics";
import { cn } from "@/lib/cn";

/** Percentile → color. Deliberately three steps, not a gradient: the point is
 *  "clearly above / around / clearly below average", not a precise reading. */
function pctTone(p: number | null): string {
  if (p == null) return "text-mute";
  if (p >= 66) return "text-accent2";
  if (p <= 33) return "text-bad";
  return "text-mute";
}

export function Players({ meta }: { meta: Meta }) {
  const [sel, setSel] = useState<PlayerSeason>(emptySelection);
  const [info, setInfo] = useState<PlayerInfo | null>(null);
  const [season, setSeason] = useState<any>(null);
  const [career, setCareer] = useState<any>(null);
  const [trendMetric, setTrendMetric] = useState("pts");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!sel.playerId || !sel.season) {
      setSeason(null);
      return;
    }
    setErr(null);
    api
      .playerSeason(sel.playerId, sel.season, meta.league)
      .then(setSeason)
      .catch((e) => {
        setErr(e.message);
        setSeason(null);
      });
  }, [sel.playerId, sel.season, meta.league]);

  useEffect(() => {
    if (!sel.playerId) {
      setCareer(null);
      return;
    }
    api.playerCareer(sel.playerId, meta.league).then(setCareer).catch(() => setCareer(null));
  }, [sel.playerId, meta.league]);

  const stats = season?.stats ?? [];

  const trendTrace = useMemo(() => {
    const rows = career?.seasons ?? [];
    if (!rows.length) return [];
    return [
      {
        type: "scatter",
        mode: "lines+markers",
        name: shortLabel(trendMetric),
        x: rows.map((r: any) => r.season),
        y: rows.map((r: any) => r[trendMetric]),
        hovertemplate: `%{x}: %{y}<extra>${label(trendMetric)}</extra>`,
      },
    ];
  }, [career, trendMetric]);

  const recent = career?.recent_games ?? [];
  const rollingTrace = useMemo(() => {
    if (!recent.length) return [];
    const x = recent.map((g: any) => g.date);
    const y = recent.map((g: any) => g.pts);
    const roll = y.map((_: number, i: number) => {
      const w = y.slice(Math.max(0, i - 4), i + 1).filter((v: number) => v != null);
      return w.length ? w.reduce((a: number, b: number) => a + b, 0) / w.length : null;
    });
    return [
      { type: "bar", name: "PTS", x, y, marker: { color: "#4dabff" } },
      { type: "scatter", mode: "lines", name: "5-game avg", x, y: roll, line: { color: "#ff6a3d", width: 2 } },
    ];
  }, [recent]);

  const careerMetrics = sortMetrics(career?.metrics ?? meta.metrics);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Player overview" />
        <CardBody>
          <PlayerSeasonSelector meta={meta} value={sel} onChange={setSel} onInfo={setInfo} />
          {err && <div className="mt-3 text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      {/* identity + headline numbers */}
      <Card>
        <CardBody>
          <div className="flex flex-wrap items-start gap-5">
            <Avatar
              name={sel.playerName || "?"}
              id={sel.playerId}
              league={meta.league}
              size={84}
            />
            <div className="min-w-0 flex-1">
              <div className="text-xl font-semibold text-ink">
                {sel.playerName || "Select a player"}
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-sm text-mute">
                {season?.season && <span>{season.season}</span>}
                {season?.team && <span>· {season.team}</span>}
                {info?.bio?.position && <span>· {info.bio.position}</span>}
                {info?.bio?.height && <span>· {info.bio.height}</span>}
                {info?.bio?.weight && <span>· {info.bio.weight}</span>}
                {/* age during the selected season, falling back to age today */}
                {(season?.bio?.age ?? info?.bio?.age) != null && (
                  <span>· age {season?.bio?.age ?? info?.bio?.age}</span>
                )}
                {info?.bio?.birthplace && <span>· {info.bio.birthplace}</span>}
              </div>
              {season && (
                <div className="mt-1 text-xs text-mute">
                  {season.gp} games · {season.min?.toFixed?.(1) ?? season.min} min per game ·
                  ranked among {season.pool_size} qualifying players
                </div>
              )}
            </div>
          </div>

          <div className="mt-5 grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4 lg:grid-cols-8">
            {(stats.length ? stats : Array.from({ length: 8 }, () => null)).map(
              (s: any, i: number) => (
                <div key={s?.metric ?? i}>
                  <div className="label">{s ? shortLabel(s.metric) : "—"}</div>
                  <div className="mt-0.5 text-lg font-semibold tabular-nums text-ink">
                    {s ? formatValue(s.metric, s.value) : "—"}
                  </div>
                  <div className={cn("text-xs tabular-nums", pctTone(s?.percentile ?? null))}>
                    {s?.percentile != null ? `${ordinal(s.percentile)} %ile` : " "}
                  </div>
                  <div className="text-[11px] text-mute">
                    {s?.rank != null ? `#${s.rank} in league` : " "}
                  </div>
                </div>
              )
            )}
          </div>
        </CardBody>
      </Card>

      {/* percentiles — the old Percentiles tab */}
      <Card>
        <CardHeader
          title="Performance percentiles"
        />
        <CardBody>
          <Plot
            data={
              stats.length
                ? ([
                    {
                      type: "bar",
                      orientation: "h",
                      x: stats.map((s: any) => s.percentile),
                      y: stats.map((s: any) => label(s.metric)),
                      text: stats.map((s: any) =>
                        s.percentile != null ? ordinal(s.percentile) : ""
                      ),
                      textposition: "outside",
                      marker: {
                        color: stats.map((s: any) => s.percentile),
                        colorscale: [[0, "#d73027"], [0.5, "#6b7685"], [1, "#4dabff"]],
                        cmin: 0,
                        cmax: 100,
                      },
                      hovertemplate: "%{y}: %{x:.1f}th percentile<extra></extra>",
                    },
                  ] as any)
                : ([] as any)
            }
            layout={{
              margin: { l: 140, t: 10, r: 40 },
              showlegend: false,
              xaxis: { title: "Percentile (100 = best)", range: [0, 108] },
              yaxis: { automargin: true },
            }}
            height={Math.max(260, stats.length * 38 + 80)}
            placeholder="Select a player and season"
          />
        </CardBody>
      </Card>

      {/* career trend — the old Trends tab */}
      <Card>
        <CardHeader
          title="Career trend"
          right={
            <div className="w-44">
              <Select
                value={trendMetric}
                onChange={setTrendMetric}
                options={careerMetrics.map((k) => ({ value: k, label: label(k) }))}
              />
            </div>
          }
        />
        <CardBody>
          <Plot
            data={trendTrace as any}
            layout={{
              margin: { t: 16 },
              showlegend: false,
              xaxis: { title: "Season", type: "category", nticks: 12, tickangle: 0 },
              yaxis: { title: label(trendMetric) },
            }}
            height={320}
            placeholder="Select a player"
          />
        </CardBody>
      </Card>

      {/* recent games — the old Game Log tab */}
      <Card>
        <CardHeader
          title="Recent games"
        />
        <CardBody className="space-y-4">
          <Plot
            data={rollingTrace as any}
            layout={{
              margin: { t: 16 },
              xaxis: { title: "Game date", type: "date" },
              yaxis: { title: "Points" },
            }}
            height={260}
            placeholder="Select a player"
          />
          {recent.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-mute">
                    <th className="py-2 pr-3 font-medium">Date</th>
                    <th className="py-2 pr-3 font-medium">Matchup</th>
                    {["min", "pts", "reb", "ast", "stl", "blk", "tov"].map((k) => (
                      <th key={k} className="py-2 pr-3 text-right font-medium">
                        {shortLabel(k)}
                      </th>
                    ))}
                    <th className="py-2 text-right font-medium">FG%</th>
                  </tr>
                </thead>
                <tbody>
                  {[...recent].reverse().map((g: any) => (
                    <tr key={g.date + g.matchup} className="border-t border-border/60">
                      <td className="whitespace-nowrap py-2 pr-3 text-mute">
                        {g.date?.slice(0, 10)}
                      </td>
                      <td className="whitespace-nowrap py-2 pr-3">{g.matchup}</td>
                      {["min", "pts", "reb", "ast", "stl", "blk", "tov"].map((k) => (
                        <td key={k} className="py-2 pr-3 text-right tabular-nums">
                          {g[k] ?? "—"}
                        </td>
                      ))}
                      <td className="py-2 text-right tabular-nums">
                        {g.fg_pct != null ? `${(g.fg_pct * 100).toFixed(0)}%` : "—"}
                      </td>
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
