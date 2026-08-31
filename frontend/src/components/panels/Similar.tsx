import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Meta, type PlayerSeason } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Plot, traceColor } from "@/components/ui/Plot";
import { Slider } from "@/components/ui/Slider";
import { PlayerSeasonSelector, emptySelection } from "@/components/ui/PlayerSeasonSelector";
import { PlayerLegend } from "@/components/ui/PlayerLegend";
import { playerAvatar } from "@/components/ui/Avatar";
import { formatValue, shortLabel } from "@/lib/metrics";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

const PRESETS = ["Overall", "Scoring", "Shooting", "Playmaking", "Defense", "Custom"];

const rowKey = (r: any) => `${r.player_id}-${r.season}`;

type OverlayRow = {
  key: string;
  playerName: string;
  season: string;
  values: number[];
  color: string;
  isAnchor: boolean;
};

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
  /** Keys of the seasons drawn on the radar, chosen from the result set. */
  const [shown, setShown] = useState<string[]>([]);

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

  /** Every season the overlay can draw — the anchor first, then the matches in
   *  similarity order. Colors are pinned to this order so a season keeps its
   *  color as the selection changes. */
  const overlayRows: OverlayRow[] = useMemo(() => {
    const series: any[] = data?.radar?.series ?? [];
    if (!series.length) return [];
    const source = [data.anchor, ...matches];
    return series.map((s: any, i: number): OverlayRow => ({
      key: rowKey(source[i]),
      playerName: source[i].player_name as string,
      season: String(source[i].season),
      values: s.values as number[],
      color: traceColor(i),
      isAnchor: i === 0,
    }));
  }, [data]);

  const anchorKey = data?.anchor ? rowKey(data.anchor) : null;
  const shownAnchor = useRef<string | null>(null);

  // The overlay starts as the chosen season alone — comparisons appear only
  // when picked from the list. A new anchor clears them; changing k or the
  // weighting only drops picks that fell out of the results, so hand-picked
  // seasons survive a slider nudge.
  useEffect(() => {
    if (!anchorKey) {
      shownAnchor.current = null;
      setShown([]);
      return;
    }
    if (shownAnchor.current !== anchorKey) {
      shownAnchor.current = anchorKey;
      setShown([anchorKey]);
      return;
    }
    setShown((prev) => {
      const valid = new Set(overlayRows.map((r) => r.key));
      const kept = prev.filter((k) => valid.has(k));
      return kept.length === prev.length ? prev : kept;
    });
  }, [overlayRows, anchorKey]);

  const isShown = (key: string) => shown.includes(key);
  /** The anchor is the baseline every comparison is drawn against, so it stays. */
  const toggleShown = (key: string) => {
    if (key === anchorKey) return;
    setShown((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]));
  };

  /** Rows actually on the radar, in similarity order. */
  const shownRows = useMemo(
    () => overlayRows.filter((r) => shown.includes(r.key)),
    [overlayRows, shown]
  );

  const radarTraces = useMemo(() => {
    if (!shownRows.length) return [];
    const feats: string[] = data.radar.features.map(shortLabel);
    return shownRows.map((r) => ({
      type: "scatterpolar",
      name: `${r.season} ${r.playerName}`,
      r: [...r.values, r.values[0]],
      theta: [...feats, feats[0]],
      fill: "toself",
      line: { color: r.color },
      opacity: r.isAnchor ? 0.55 : 0.22,
      hovertemplate: `%{theta}: %{r:.0%}<extra>${r.season} ${r.playerName}</extra>`,
    }));
  }, [data, shownRows]);

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
            subtitle={matches.length > 0 ? "Click a season to add or remove it from the overlay" : undefined}
          />
          <CardBody className="p-0">
            {matches.length === 0 ? (
              <div className="px-5 py-8 text-sm text-mute">Select a player and season.</div>
            ) : (
              <ul>
                {matches.map((m: any) => {
                  const key = rowKey(m);
                  const on = isShown(key);
                  const color = overlayRows.find((r) => r.key === key)?.color;
                  return (
                    <li key={key} className="border-t border-border/60">
                      <button
                        type="button"
                        onClick={() => toggleShown(key)}
                        aria-pressed={on}
                        title={on ? "Remove from the overlay" : "Add to the overlay"}
                        className={cn(
                          "flex w-full items-start gap-3 px-5 py-3 text-left transition hover:bg-border/25",
                          on && "bg-border/15"
                        )}
                      >
                        <span
                          aria-hidden
                          className={cn(
                            "mt-2.5 h-3.5 w-3.5 shrink-0 border transition",
                            on ? "border-transparent" : "border-border"
                          )}
                          style={on ? { background: color } : undefined}
                        />
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
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </CardBody>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader
            title="Profile overlay"
            right={
              overlayRows.length > 0 ? (
                <div className="flex shrink-0 items-center gap-3 text-xs">
                  <button
                    onClick={() => setShown(overlayRows.map((r) => r.key))}
                    className="text-mute transition hover:text-ink"
                  >
                    All
                  </button>
                  <button
                    onClick={() => setShown(anchorKey ? [anchorKey] : [])}
                    className="text-mute transition hover:text-ink"
                  >
                    Clear
                  </button>
                </div>
              ) : undefined
            }
          />
          <CardBody>
            {shownRows.length > 0 && (
              <div className="mb-2 max-h-36 overflow-y-auto pr-1">
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                  {shownRows.map((r) => (
                    <span key={r.key} className="flex max-w-full items-center gap-2 text-sm text-ink">
                      <span className="relative">
                        {avatar(r.playerName, 24)}
                        <span
                          aria-hidden
                          className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 ring-2 ring-panel"
                          style={{ background: r.color }}
                        />
                      </span>
                      <span className="truncate">
                        {formatSeason(r.season, meta.season_format)} {r.playerName}
                      </span>
                      {!r.isAnchor && (
                        <button
                          onClick={() => toggleShown(r.key)}
                          title="Remove from the overlay"
                          aria-label={`Remove ${r.playerName} from the overlay`}
                          className="text-mute transition hover:text-ink"
                        >
                          ×
                        </button>
                      )}
                    </span>
                  ))}
                </div>
              </div>
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
                        <span
                          aria-hidden
                          className="mr-2 inline-block h-2 w-2 align-middle"
                          style={{
                            background: isShown(rowKey(r))
                              ? overlayRows.find((o) => o.key === rowKey(r))?.color
                              : "transparent",
                          }}
                        />
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
