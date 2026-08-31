import { useEffect, useMemo, useState } from "react";
import { api, type Meta, type PlayerSeason } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Plot } from "@/components/ui/Plot";
import { Slider } from "@/components/ui/Slider";
import { PlayerSeasonSelector, emptySelection } from "@/components/ui/PlayerSeasonSelector";
import { PlayerLegend } from "@/components/ui/PlayerLegend";
import { playerAvatar } from "@/components/ui/Avatar";
import { formatValue, shortLabel } from "@/lib/metrics";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

const PRESETS = ["Overall", "Scoring", "Shooting", "Playmaking", "Defense", "Custom"];

/** `seed` arrives from Ask Full Court: the structured query it just ran, so
 *  "Open in Similarity" lands on the answer rather than an empty panel. */
export function Similar({ meta, seed }: { meta: Meta; seed?: any }) {
  const avatar = playerAvatar(meta);
  const [sel, setSel] = useState<PlayerSeason>(emptySelection);
  const [preset, setPreset] = useState("Overall");
  const [weights, setWeights] = useState<Record<string, number>>({});
  const [k, setK] = useState(8);
  const [minGp, setMinGp] = useState(20);
  const [sameSeason, setSameSeason] = useState(false);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!seed) return;
    if (seed.player_id && seed.season) {
      setSel({
        playerId: seed.player_id,
        playerName: seed.player_name ?? "",
        season: String(seed.season),
      });
    }
    if (seed.preset) setPreset(seed.preset);
    if (typeof seed.k === "number") setK(seed.k);
    if (typeof seed.min_gp === "number") setMinGp(seed.min_gp);
  }, [seed]);

  const features: string[] = data?.features ?? [];

  useEffect(() => {
    if (!sel.playerId || !sel.season) {
      setData(null);
      return;
    }
    setErr(null);
    api
      .similarity({
        player_id: sel.playerId,
        season: sel.season,
        league: meta.league,
        preset: preset === "Custom" ? "Overall" : preset,
        weights: preset === "Custom" ? weights : {},
        k,
        min_gp: minGp,
        same_season_only: sameSeason,
      })
      .then(setData)
      .catch((e) => {
        setErr(e.message);
        setData(null);
      });
  }, [sel.playerId, sel.season, preset, JSON.stringify(weights), k, minGp, sameSeason, meta.league]);

  const matches = data?.matches ?? [];

  const radarTraces = useMemo(() => {
    const series = data?.radar?.series ?? [];
    if (!series.length) return [];
    const feats: string[] = data.radar.features.map(shortLabel);
    return series.slice(0, 5).map((s: any, i: number) => ({
      type: "scatterpolar",
      name: s.name,
      r: [...s.values, s.values[0]],
      theta: [...feats, feats[0]],
      fill: "toself",
      opacity: i === 0 ? 0.55 : 0.22,
      hovertemplate: `%{theta}: %{r:.0%}<extra>${s.name}</extra>`,
    }));
  }, [data]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Find similar seasons"
        />
        <CardBody className="space-y-4">
          <PlayerSeasonSelector meta={meta} value={sel} onChange={setSel} />

          <div>
            <div className="label mb-1.5">Weighting</div>
            <div className="flex flex-wrap gap-4 text-sm">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  onClick={() => setPreset(p)}
                  className={cn(
                    "border-b-2 pb-0.5 transition",
                    preset === p ? "border-accent text-ink" : "border-transparent text-mute hover:text-ink"
                  )}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          {preset === "Custom" && features.length > 0 && (
            <div className="grid gap-x-6 gap-y-3 border border-border p-4 md:grid-cols-2 lg:grid-cols-4">
              {features.map((f) => (
                <div key={f}>
                  <div className="mb-1 flex justify-between text-xs">
                    <span className="text-mute">{shortLabel(f)}</span>
                    <span className="tabular-nums text-ink">{(weights[f] ?? 1).toFixed(1)}</span>
                  </div>
                  <Slider
                    min={0}
                    max={3}
                    step={0.1}
                    value={weights[f] ?? 1}
                    onChange={(v) => setWeights((w) => ({ ...w, [f]: v }))}
                  />
                </div>
              ))}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-6 text-sm">
            <label className="flex items-center gap-2 text-mute">
              <input
                type="checkbox"
                checked={sameSeason}
                onChange={(e) => setSameSeason(e.target.checked)}
              />
              Same season only
            </label>
            <div className="flex items-center gap-3">
              <span className="text-mute">Results</span>
              <div className="w-32">
                <Slider min={3} max={20} step={1} value={k} onChange={setK} />
              </div>
              <span className="tabular-nums text-ink">{k}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-mute">Min games</span>
              <div className="w-32">
                <Slider min={1} max={60} step={1} value={minGp} onChange={setMinGp} />
              </div>
              <span className="tabular-nums text-ink">{minGp}</span>
            </div>
            {data && (
              <span className="text-xs text-mute">
                searching {data.pool_size.toLocaleString()} player-seasons
              </span>
            )}
          </div>
          {err && <div className="text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader
            title="Most similar seasons"
          />
          <CardBody className="p-0">
            {matches.length === 0 ? (
              <div className="px-5 py-8 text-sm text-mute">Select a player and season.</div>
            ) : (
              <ul>
                {matches.map((m: any) => (
                  <li key={`${m.player_id}-${m.season}`} className="border-t border-border/60 px-5 py-3">
                    <div className="flex items-start gap-3">
                      {avatar(m.player_name, 34)}
                      <div className="min-w-0 flex-1">
                        <div className="flex items-baseline justify-between gap-3">
                          <span className="truncate">
                            {formatSeason(m.season, meta.season_format)} {m.player_name}
                            <span className="ml-2 text-xs text-mute">{m.team}</span>
                          </span>
                          <span className="shrink-0 tabular-nums text-accent">
                            {(m.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="mt-1 grid gap-x-6 gap-y-0.5 text-xs sm:grid-cols-2">
                          <div className="text-mute">
                            <span className="text-accent2">Most alike:</span>{" "}
                            {m.most_similar.slice(0, 3).join(", ")}
                          </div>
                          <div className="text-mute">
                            <span className="text-bad">Biggest gap:</span>{" "}
                            {m.biggest_difference.join(", ")}
                          </div>
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </CardBody>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader
            title="Profile overlay"
          />
          <CardBody>
            {radarTraces.length > 0 && (
              <PlayerLegend
                names={(data?.radar?.series ?? []).slice(0, 5).map((s: any) => s.name)}
                renderAvatar={(n, size) => avatar(n.split(" ").slice(1).join(" "), size)}
                className="mb-2"
                size={24}
              />
            )}
            <Plot
              data={radarTraces as any}
              layout={{
                showlegend: false,
                polar: {
                  bgcolor: "rgba(0,0,0,0)",
                  radialaxis: { range: [0, 1], tickformat: ".0%", gridcolor: "#1f2630", color: "#8a94a2" },
                  angularaxis: { gridcolor: "#1f2630", color: "#cbd3de" },
                },
                margin: { t: 30, l: 40, r: 40, b: 30 },
              }}
              height={420}
              placeholder="Select a player and season"
            />
          </CardBody>
        </Card>
      </div>

      {matches.length > 0 && (
        <Card>
          <CardHeader title="Underlying numbers" />
          <CardBody className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-mute">
                    <th className="px-4 py-2 font-medium">Season</th>
                    {features.map((f) => (
                      <th key={f} className="px-3 py-2 text-right font-medium">{shortLabel(f)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[data.anchor, ...matches].map((r: any, i: number) => (
                    <tr
                      key={`${r.player_id}-${r.season}`}
                      className={cn("border-t border-border/60", i === 0 && "bg-border/25")}
                    >
                      <td className="whitespace-nowrap px-4 py-2">
                        {formatSeason(r.season, meta.season_format)} {r.player_name}
                        {i === 0 && <span className="ml-2 text-xs text-accent">selected</span>}
                      </td>
                      {features.map((f) => (
                        <td key={f} className="px-3 py-2 text-right tabular-nums">
                          {formatValue(f, r.values?.[f])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}
