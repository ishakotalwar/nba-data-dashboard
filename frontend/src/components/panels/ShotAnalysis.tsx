import { useEffect, useMemo, useState } from "react";
import { api, type Meta, type PlayerSeason } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Plot } from "@/components/ui/Plot";
import { PlayerSeasonSelector, emptySelection } from "@/components/ui/PlayerSeasonSelector";
import { playerAvatar } from "@/components/ui/Avatar";
import { buildCourtShapes, courtAxis } from "@/lib/court";
import { cn } from "@/lib/cn";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const signed = (v: number | null | undefined) =>
  v == null ? "" : `${v >= 0 ? "+" : "−"}${(Math.abs(v) * 100).toFixed(1)}%`;

function CourtPlot({
  meta,
  data,
  mode,
  height = 560,
}: {
  meta: Meta;
  data: any;
  mode: "hex" | "scatter";
  height?: number;
}) {
  const traces = useMemo(() => {
    if (!data) return [];
    if (mode === "scatter") {
      const made = (data.shots ?? []).filter((s: any) => s.made === 1);
      const miss = (data.shots ?? []).filter((s: any) => s.made !== 1);
      return [
        {
          type: "scattergl", mode: "markers", name: "Miss",
          x: miss.map((s: any) => s.x), y: miss.map((s: any) => s.y),
          marker: { symbol: "x", size: 5, color: "#4a5568", opacity: 0.55 },
          hoverinfo: "skip",
        },
        {
          type: "scattergl", mode: "markers", name: "Make",
          x: made.map((s: any) => s.x), y: made.map((s: any) => s.y),
          marker: { size: 6, color: "#ff6a3d", opacity: 0.75 },
          hoverinfo: "skip",
        },
      ];
    }
    const hexes = data.hexes ?? [];
    if (!hexes.length) return [];
    const avg = data.fg_pct ?? 0;
    const maxCount = Math.max(...hexes.map((h: any) => h.count));
    return [
      {
        type: "scatter", mode: "markers",
        x: hexes.map((h: any) => h.x), y: hexes.map((h: any) => h.y),
        marker: {
          symbol: "hexagon",
          size: hexes.map((h: any) => 6 + 22 * Math.sqrt(h.count / maxCount)),
          color: hexes.map((h: any) => h.pct - avg),
          colorscale: [[0, "#d73027"], [0.5, "#6b7685"], [1, "#4dabff"]],
          cmid: 0,
          line: { color: "#111518", width: 1 },
          colorbar: {
            title: { text: "vs. own avg", side: "right" },
            thickness: 9, outlinewidth: 0, tickformat: ".0%",
            tickfont: { color: "#8a94a2", size: 9 },
          },
        },
        customdata: hexes.map((h: any) => [h.count, h.pct]),
        hovertemplate: "%{customdata[0]} shots · %{customdata[1]:.0%}<extra></extra>",
      },
    ];
  }, [data, mode]);

  return (
    <Plot
      data={traces as any}
      layout={{
        shapes: buildCourtShapes(meta.court) as any,
        ...courtAxis,
        margin: { t: 10, l: 10, r: 10, b: 10 },
        showlegend: false,
      }}
      height={height}
      placeholder="Select a player and season"
    />
  );
}

export function ShotAnalysis({ meta }: { meta: Meta }) {
  const avatar = playerAvatar(meta);
  const [a, setA] = useState<PlayerSeason>(emptySelection);
  const [b, setB] = useState<PlayerSeason>(emptySelection);
  const [comparing, setComparing] = useState(false);
  const [mode, setMode] = useState<"hex" | "scatter">("hex");
  const [shotsA, setShotsA] = useState<any>(null);
  const [shotsB, setShotsB] = useState<any>(null);
  const [zonesA, setZonesA] = useState<any>(null);
  const [zonesB, setZonesB] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = (
    sel: PlayerSeason,
    setShots: (v: any) => void,
    setZones: (v: any) => void
  ) => {
    if (!sel.playerId || !sel.season) {
      setShots(null);
      setZones(null);
      return;
    }
    api.shots(sel.playerId, sel.season, mode, meta.league).then(setShots).catch(() => setShots(null));
    api
      .shotZones(sel.playerId, sel.season, meta.league)
      .then(setZones)
      .catch((e) => {
        setZones(null);
        setErr(e.message);
      });
  };

  useEffect(() => {
    setErr(null);
    load(a, setShotsA, setZonesA);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a.playerId, a.season, mode, meta.league]);

  useEffect(() => {
    if (!comparing) {
      setShotsB(null);
      setZonesB(null);
      return;
    }
    load(b, setShotsB, setZonesB);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b.playerId, b.season, comparing, mode, meta.league]);

  const zoneRows = zonesA?.zones ?? [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Shot analysis"
          right={
            <div className="flex gap-4 text-sm">
              {(["hex", "scatter"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={cn(
                    "border-b-2 pb-0.5 transition",
                    mode === m ? "border-accent text-ink" : "border-transparent text-mute hover:text-ink"
                  )}
                >
                  {m === "hex" ? "Hexbin" : "Scatter"}
                </button>
              ))}
            </div>
          }
        />
        <CardBody className="space-y-3">
          <PlayerSeasonSelector meta={meta} value={a} onChange={setA} />
          <label className="flex items-center gap-2 text-sm text-mute">
            <input type="checkbox" checked={comparing} onChange={(e) => setComparing(e.target.checked)} />
            Compare with another player-season
          </label>
          {comparing && (
            <PlayerSeasonSelector meta={meta} value={b} onChange={setB} playerLabel="Compare with" />
          )}
          {err && <div className="text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      <div className={cn("grid gap-4", comparing && "lg:grid-cols-2")}>
        <Card>
          <CardHeader
            lead={a.playerName ? avatar(a.playerName, 40) : undefined}
            title={zonesA ? `${zonesA.season} ${zonesA.player_name}` : "—"}
            subtitle={
              zonesA ? `${zonesA.total_fga} field goal attempts · ${pct(zonesA.fg_pct)} overall` : undefined
            }
          />
          <CardBody>
            <CourtPlot meta={meta} data={shotsA} mode={mode} height={comparing ? 460 : 560} />
          </CardBody>
        </Card>

        {comparing && (
          <Card>
            <CardHeader
              lead={b.playerName ? avatar(b.playerName, 40) : undefined}
              title={zonesB ? `${zonesB.season} ${zonesB.player_name}` : "—"}
              subtitle={
                zonesB ? `${zonesB.total_fga} field goal attempts · ${pct(zonesB.fg_pct)} overall` : " "
              }
            />
            <CardBody>
              <CourtPlot meta={meta} data={shotsB} mode={mode} height={460} />
            </CardBody>
          </Card>
        )}
      </div>

      <Card>
        <CardHeader
          title="Zone breakdown"
        />
        <CardBody className="p-0">
          {zoneRows.length === 0 ? (
            <div className="px-5 py-8 text-sm text-mute">Select a player and season.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-mute">
                    <th className="px-4 py-2 font-medium">Zone</th>
                    <th className="px-3 py-2 text-right font-medium">FGA</th>
                    <th className="px-3 py-2 text-right font-medium">FGM</th>
                    <th className="px-3 py-2 text-right font-medium">FG%</th>
                    <th className="px-3 py-2 text-right font-medium">League</th>
                    <th className="px-3 py-2 text-right font-medium">Diff</th>
                    <th className="px-3 py-2 text-right font-medium">Share</th>
                    {comparing && zonesB && (
                      <>
                        <th className="border-l border-border px-3 py-2 text-right font-medium">
                          {zonesB.player_name.split(" ").at(-1)} FGA
                        </th>
                        <th className="px-3 py-2 text-right font-medium">FG%</th>
                        <th className="px-4 py-2 text-right font-medium">Diff</th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {zoneRows.map((z: any, i: number) => {
                    const zb = zonesB?.zones?.[i];
                    return (
                      <tr key={z.zone} className="border-t border-border/60">
                        <td className="whitespace-nowrap px-4 py-2">{z.zone}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{z.fga}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{z.fgm}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{pct(z.fg_pct)}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-mute">
                          {pct(z.league_fg_pct)}
                        </td>
                        <td
                          className={cn(
                            "px-3 py-2 text-right tabular-nums",
                            z.diff == null ? "text-mute" : z.diff >= 0 ? "text-accent2" : "text-bad"
                          )}
                        >
                          {signed(z.diff)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-mute">
                          {pct(z.share)}
                        </td>
                        {comparing && zonesB && (
                          <>
                            <td className="border-l border-border px-3 py-2 text-right tabular-nums">
                              {zb?.fga ?? "—"}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums">{pct(zb?.fg_pct)}</td>
                            <td
                              className={cn(
                                "px-4 py-2 text-right tabular-nums",
                                zb?.diff == null ? "text-mute" : zb.diff >= 0 ? "text-accent2" : "text-bad"
                              )}
                            >
                              {signed(zb?.diff)}
                            </td>
                          </>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
