import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Meta, type PlayerSeason } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Plot } from "@/components/ui/Plot";
import { PlayerSeasonSelector, emptySelection } from "@/components/ui/PlayerSeasonSelector";
import { playerAvatar } from "@/components/ui/Avatar";
import {
  buildCourtShapes, courtAxis, courtLines, zoneOf, zonePolygons,
  type CourtGeometry, type Zone,
} from "@/lib/court";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

export type ChartMode = "hex" | "scatter" | "3d";

const MODES: { value: ChartMode; label: string }[] = [
  { value: "hex", label: "Hexbin" },
  { value: "scatter", label: "Scatter" },
  { value: "3d", label: "3D" },
];

const CARD_WIDTH = 300;   // the zone card, parked beside the court
const CARD_GAP = 14;

/** Rough floor area per zone, biggest first — only used for draw order. */
const AREA_RANK: Record<string, number> = {
  "Top of Arc 3": 9, "Left Wing 3": 8, "Right Wing 3": 7,
  "Long Midrange": 6, "Left Corner 3": 5, "Right Corner 3": 4,
  "Short Midrange": 3, Paint: 2, Rim: 1,
};

/** Axis styling shared by the three dimensions of the 3D scene. The court's
 *  own lines carry the ground plane, so the scene grid only gets in the way. */
const SCENE_AXIS = {
  showgrid: false,
  zeroline: false,
  showticklabels: false,
  showspikes: false,
  title: { text: "" },
  showbackground: false,
  color: "#8a94a2",
};

/** Reads the scale for the 3D terrain, which carries no colorbar of its own. */
function SurfaceLegend() {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-mute">
      <span className="flex items-center gap-2">
        colder
        <span
          className="h-2 w-28"
          style={{
            background:
              "linear-gradient(90deg,#22d3ee,#7dd3c8,#facc15,#f59e6d,#f0559b)",
          }}
        />
        hotter vs league
      </span>
      <span>height = attempt volume vs the player's peak · colour = efficiency vs league</span>
      <span className="ml-auto">Drag to orbit · scroll to zoom</span>
    </div>
  );
}

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const signed = (v: number | null | undefined) =>
  v == null ? "" : `${v >= 0 ? "+" : "−"}${(Math.abs(v) * 100).toFixed(1)}%`;

function CourtPlot({
  meta,
  data,
  zones,
  mode,
  height = 560,
  label,
  totalFga,
  selected,
  onSelect,
}: {
  meta: Meta;
  data: any;
  /** Zone rows from `/shots/zones`, for the numbers a zone shows on hover. */
  zones?: any[];
  mode: ChartMode;
  height?: number;
  /** Whose court this is, for the popover's subtitle. */
  label?: string;
  totalFga?: number;
  /** The zone the court is sliced to, or null for the whole floor. */
  selected: Zone | null;
  onSelect: (zone: Zone) => void;
}) {
  const court = meta.court as CourtGeometry;
  const frame = useRef<HTMLDivElement>(null);
  // A 3D scene emits no click events, so the last hovered spot on the terrain
  // stands in for one, and a press that does not move counts as a click rather
  // than the start of an orbit.
  const hovered = useRef<{ x: number; y: number } | null>(null);
  const pressed = useRef<{ x: number; y: number } | null>(null);
  const [pop, setPop] = useState<{ zone: Zone; left: number; top: number } | null>(null);

  /** Where the card goes: beside the court, in the gutter away from the click.
   *  Without a click — a zone picked from the table below — it parks in the
   *  right-hand gutter, level with the middle of the court. */
  const place = (at?: { clientX: number; clientY: number }) => {
    const box = frame.current?.getBoundingClientRect();
    if (!box) return null;
    // A half-court leaves wide margins either side of the chart. `.nsewdrag` is
    // Plotly's plot area; a 3D scene has none, so that falls back to the middle.
    const drag = frame.current?.querySelector(".nsewdrag");
    const courtBox = drag?.getBoundingClientRect();
    const courtLeft = courtBox ? courtBox.left - box.left : box.width * 0.28;
    const courtRight = courtBox ? courtBox.right - box.left : box.width * 0.72;
    const rightGutter = box.width - courtRight;
    const onLeft = at ? at.clientX - box.left < box.width / 2 : true;

    let left: number;
    if (onLeft && rightGutter >= CARD_WIDTH + CARD_GAP) {
      left = courtRight + CARD_GAP;
    } else if (courtLeft >= CARD_WIDTH + CARD_GAP) {
      left = courtLeft - CARD_WIDTH - CARD_GAP;
    } else if (rightGutter >= CARD_WIDTH + CARD_GAP) {
      left = courtRight + CARD_GAP;
    } else {
      // Nowhere to put it beside the court; hug the edge away from the click.
      left = onLeft ? box.width - CARD_WIDTH - CARD_GAP : CARD_GAP;
    }
    const top = at ? at.clientY - box.top : box.height / 2;
    // Level with the click, but never hanging off the top or bottom.
    return { left, top: Math.min(Math.max(top, 120), Math.max(box.height - 120, 120)) };
  };

  // The card follows the selection wherever it was made: cleared when the
  // selection is, and opened beside the court when a zone is picked from the
  // table rather than from the chart.
  useEffect(() => {
    if (!selected) {
      setPop(null);
      return;
    }
    if (pop?.zone === selected) return;
    const at = place();
    setPop(at ? { zone: selected, ...at } : null);
  }, [selected]);   // eslint-disable-line react-hooks/exhaustive-deps

  const pick = (zone: Zone, at: { clientX: number; clientY: number }) => {
    const spot = place(at);
    if (spot) setPop({ zone, ...spot });
    onSelect(zone);
  };

  // The court's own regions, drawn as filled traces so Plotly can hover and
  // click them. `hoveron: "fills"` is what makes the whole region live rather
  // than just its outline.
  const zoneTraces = useMemo(() => {
    if (mode === "3d") return [];
    const byZone = new Map<string, any>((zones ?? []).map((z: any) => [z.zone, z]));
    // Small regions last: Plotly hovers whichever trace was drawn on top, and
    // the rim sits inside the paint, so painting it first made it unhoverable.
    const order = zonePolygons(court).slice().sort((a, b) => AREA_RANK[b.zone] - AREA_RANK[a.zone]);
    return order.map((poly) => {
      const row = byZone.get(poly.zone);
      const on = selected === poly.zone;
      const dimmed = selected !== null && !on;
      const numbers = row
        ? `${row.fga} attempts · ${(row.fg_pct * 100).toFixed(1)}%` +
          ` · league ${(row.league_fg_pct * 100).toFixed(1)}%`
        : "no attempts";
      return {
        type: "scatter",
        mode: "lines",
        name: poly.zone,
        x: [...poly.x, poly.x[0]],
        y: [...poly.y, poly.y[0]],
        fill: "toself",
        fillcolor: on ? "rgba(255,106,61,0.16)" : "rgba(120,132,148,0.05)",
        line: {
          color: on ? "#ff6a3d" : "rgba(120,132,148,0.3)",
          width: on ? 2.5 : 1,
        },
        opacity: dimmed ? 0.3 : 1,
        // A filled region hovers through `text`, not `hovertemplate` — with
        // hoveron "fills" there is no point under the cursor to template. The
        // label steps aside once a card is open, since the card says more and
        // the two would sit on top of each other. "none" and not "skip": skip
        // would take the click events with it.
        hoveron: "fills",
        text: `<b>${poly.zone}</b><br>${numbers}`,
        hoverinfo: selected ? "none" : "text",
        hoverlabel: { bgcolor: "#12161c", bordercolor: "#2a3240",
                      font: { color: "#e6eaf0" }, align: "left" },
      };
    });
  }, [court, zones, selected, mode]);

  // A smoothed terrain over the hex bins: height is how much a player shoots
  // from around a spot against their own busiest spot, colour is how well they
  // shoot there against the league's rate for that zone. Cells with almost no
  // attempts are left out rather than drawn flat, so the surface has the shape
  // of where they actually shoot.
  const surfaceTraces = useMemo(() => {
    if (mode !== "3d") return [];
    const floor = courtLines(court, { inner: false }).map((l) => ({
      type: "scatter3d", mode: "lines",
      x: l.x, y: l.y, z: l.x.map(() => 0),
      line: { color: "rgba(180,190,205,0.35)", width: 2 },
      hoverinfo: "skip", showlegend: false,
    }));
    const hoop = {
      type: "scatter3d", mode: "lines",
      x: Array.from({ length: 41 }, (_, i) => 7.5 * Math.cos((i / 40) * 2 * Math.PI)),
      y: Array.from({ length: 41 }, (_, i) => 7.5 * Math.sin((i / 40) * 2 * Math.PI)),
      z: Array.from({ length: 41 }, () => 0),
      line: { color: "#f0559b", width: 4 },
      hoverinfo: "skip", showlegend: false,
    };
    const hexes = data?.hexes ?? [];
    if (!hexes.length) return [...floor, hoop];

    // The league's rate for the zone a shot came from is the yardstick — the
    // API gives it per zone, which is as fine-grained as the league data goes.
    const leagueRate = new Map<string, number>(
      (zones ?? []).map((z: any) => [z.zone, z.league_fg_pct]),
    );
    const edge = hexes.map((h: any) => ({
      x: h.x, y: h.y, count: h.count,
      vsLeague: h.pct - (leagueRate.get(zoneOf(h.x, h.y, court)) ?? data.fg_pct ?? h.pct),
    }));

    const STEP = 10;
    const SIGMA = 28;          // ~2.8 ft of blur, about one hex across
    const TWO_SIGMA_SQ = 2 * SIGMA * SIGMA;
    const CUTOFF = 9 * SIGMA * SIGMA;
    const xs: number[] = [];
    for (let x = -250; x <= 250; x += STEP) xs.push(x);
    const ys: number[] = [];
    for (let y = -52.5; y <= 400; y += STEP) ys.push(y);

    const mass: number[][] = [];
    const colour: number[][] = [];
    for (const y of ys) {
      const mRow: number[] = [];
      const cRow: number[] = [];
      for (const x of xs) {
        let total = 0;
        let weighted = 0;
        for (const h of edge) {
          const dx = h.x - x;
          const dy = h.y - y;
          const d2 = dx * dx + dy * dy;
          if (d2 > CUTOFF) continue;
          const w = h.count * Math.exp(-d2 / TWO_SIGMA_SQ);
          total += w;
          weighted += w * h.vsLeague;
        }
        mRow.push(total);
        cRow.push(total > 0 ? weighted / total : 0);
      }
      mass.push(mRow);
      colour.push(cRow);
    }
    const peak = Math.max(...mass.flat(), 1);
    const FLOOR_SHARE = 0.008;   // below this it is noise, not a shooting spot

    const height = mass.map((row) => row.map((v) => (v / peak < FLOOR_SHARE ? null : v / peak)));
    const terrain = {
      type: "surface",
      x: xs,
      y: ys,
      surfacecolor: colour,
      cmid: 0,
      cmin: -0.12,
      cmax: 0.12,
      showscale: false,
      opacity: 1,
      lighting: { ambient: 0.9, diffuse: 0.35, specular: 0.04, roughness: 1 },
      contours: { z: { show: false, highlight: false } },
      // No label on the terrain — but "none" and not "skip", because the hover
      // events it keeps firing are what stand in for the clicks a 3D scene
      // never sends.
      hoverinfo: "none",
    };
    // Cold to hot against the league, the way a heat map reads.
    const HEAT = [[0, "#22d3ee"], [0.42, "#7dd3c8"], [0.5, "#facc15"],
                  [0.75, "#f59e6d"], [1, "#f0559b"]];
    // The same scale washed out: the rest of the floor stays readable as heat,
    // just pale enough that the chosen zone is what your eye lands on.
    const FADED = [[0, "#bfeef7"], [0.42, "#d3efea"], [0.5, "#fdf0b4"],
                   [0.75, "#fbdfd0"], [1, "#fbc7dd"]];

    if (!selected) {
      return [...floor, hoop, { ...terrain, z: height, colorscale: HEAT }];
    }

    // Sliced, the way the flat court slices: the chosen zone keeps its full
    // colour, the rest of the terrain fades back, and the zone is outlined on
    // the floor underneath it.
    const zoneAt = ys.map((y) => xs.map((x) => zoneOf(x, y, court)));
    const only = (want: boolean) =>
      height.map((row, j) =>
        row.map((v, i) => (v == null || (zoneAt[j][i] === selected) !== want ? null : v)),
      );
    const outline = zonePolygons(court).find((z) => z.zone === selected);

    return [
      ...floor,
      hoop,
      { ...terrain, z: only(false), colorscale: FADED },
      { ...terrain, z: only(true), colorscale: HEAT },
      ...(outline
        ? [{
            type: "scatter3d", mode: "lines",
            x: [...outline.x, outline.x[0]],
            y: [...outline.y, outline.y[0]],
            z: outline.x.map(() => 0.004).concat(0.004),
            line: { color: "#ff6a3d", width: 5 },
            hoverinfo: "skip", showlegend: false,
          }]
        : []),
    ];
  }, [data, zones, mode, court, selected]);

  const traces = useMemo(() => {
    if (!data || mode === "3d") return [];
    // A picked zone is highlighted underneath, not cut out: every shot stays on
    // the floor so the slice can be read against the rest of the court.
    if (mode === "scatter") {
      const shots = data.shots ?? [];
      const made = shots.filter((s: any) => s.made === 1);
      const miss = shots.filter((s: any) => s.made !== 1);
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
        // The zone underneath carries the hover; a tooltip per hexagon fights
        // with it for the same pixels.
        hoverinfo: "skip",
      },
    ];
  }, [data, mode, court]);

  // The chosen zone is marked in place — the whole floor stays visible, so a
  // slice can still be read against the rest of the court.
  const popover = pop && selected === pop.zone && (
    <div
      className="absolute z-20"
      onMouseDownCapture={(e) => e.stopPropagation()}
      onMouseUpCapture={(e) => e.stopPropagation()}
      style={{
        left: pop.left,
        top: pop.top,
        width: CARD_WIDTH,
        transform: "translateY(-50%)",
      }}
    >
      <ZoneCard
        zone={pop.zone}
        label={label ?? ""}
        row={(zones ?? []).find((z: any) => z.zone === pop.zone)}
        totalFga={totalFga}
        onClose={() => onSelect(pop.zone)}
      />
    </div>
  );

  if (mode === "3d") {
    return (
      <div
        ref={frame}
        className="relative"
        onMouseDownCapture={(e) => {
          pressed.current = { x: e.clientX, y: e.clientY };
        }}
        onMouseUpCapture={(e) => {
          const from = pressed.current;
          pressed.current = null;
          const at = hovered.current;
          if (!from || !at) return;
          if (Math.hypot(e.clientX - from.x, e.clientY - from.y) > 4) return;  // an orbit
          pick(zoneOf(at.x, at.y, court), e);
        }}
      >
      <Plot
        data={surfaceTraces as any}
        layout={{
          scene: {
            xaxis: { ...SCENE_AXIS, range: [-260, 260] },
            // Cropped past the arc: nobody shoots from the far half, and the
            // empty floor only pushed the terrain into a corner of the frame.
            yaxis: { ...SCENE_AXIS, range: [-60, 330] },
            // No ticks or grid on height: the legend under the chart says what
            // it means, and the shelves the grid draws sit in front of the
            // terrain.
            zaxis: { ...SCENE_AXIS, range: [0, 1.05] },
            aspectmode: "manual",
            aspectratio: { x: 1, y: 0.75, z: 0.42 },
            // Looking in from half court, so the rim and its peak are at the
            // front of the scene rather than buried at the back.
            camera: { eye: { x: 0, y: 1.05, z: 0.5 }, center: { x: 0, y: 0.06, z: -0.1 } },
          },
          margin: { t: 0, l: 0, r: 0, b: 0 },
          showlegend: false,
        }}
        height={height}
        placeholder="Select a player and season"
        // The terrain is one surface, so the zone comes from where on the
        // floor the click landed rather than from a trace name.
        onPointHover={(p) => {
          if (p.x != null && p.y != null) hovered.current = { x: p.x, y: p.y };
        }}
      />
      {popover}
      </div>
    );
  }

  return (
    <div ref={frame} className="relative">
    <Plot
      data={[...zoneTraces, ...traces] as any}
      layout={{
        shapes: buildCourtShapes(meta.court) as any,
        ...courtAxis,
        margin: { t: 10, l: 10, r: 10, b: 10 },
        showlegend: false,
      }}
      height={height}
      placeholder="Select a player and season"
      onPointClick={(p) => p.name && pick(p.name as Zone, p)}
    />
    {popover}
    </div>
  );
}


/**
 * The breakdown for one zone, shown where it was clicked: what the player did
 * there against what the league does from the same part of the floor.
 */
function ZoneCard({
  zone,
  label,
  row,
  totalFga,
  onClose,
}: {
  zone: Zone;
  label: string;
  row: any | undefined;
  totalFga: number | undefined;
  onClose: () => void;
}) {
  if (!row) {
    return (
      <div className="card p-4 text-sm text-mute shadow-card">
        <div className="mb-1 font-semibold text-ink">{zone}</div>
        No attempts from here.
      </div>
    );
  }
  const diff = row.fg_pct - row.league_fg_pct;
  const scale = Math.max(row.fg_pct, row.league_fg_pct, 0.05);
  return (
    <div className="card p-4 shadow-card">
      <div className="mb-3 flex items-start justify-between gap-4">
        <div>
          <div className="text-base font-semibold text-ink">{zone}</div>
          <div className="text-xs text-mute">{label}</div>
        </div>
        <button
          onClick={onClose}
          aria-label="Close"
          className="-mr-1 -mt-1 px-1.5 text-lg leading-none text-mute transition hover:text-ink"
        >
          ×
        </button>
      </div>

      <div className="flex items-end justify-between gap-4">
        <div>
          <div className="label mb-0.5">Player</div>
          <div className="text-2xl font-semibold tabular-nums text-ink">{pct(row.fg_pct)}</div>
        </div>
        <div>
          <div className="label mb-0.5">League</div>
          <div className="text-2xl font-semibold tabular-nums text-mute">
            {pct(row.league_fg_pct)}
          </div>
        </div>
        <div className="text-right">
          <div className="label mb-0.5">Diff</div>
          <div
            className={cn(
              "text-2xl font-semibold tabular-nums",
              diff >= 0 ? "text-good" : "text-bad",
            )}
          >
            {signed(diff)}
          </div>
        </div>
      </div>

      {/* Both rates on one scale, so the gap is a length and not just a number. */}
      <div className="mt-3 space-y-1.5">
        {[
          { name: "Player", value: row.fg_pct, cls: "bg-accent" },
          { name: "League", value: row.league_fg_pct, cls: "bg-mute/60" },
        ].map((bar) => (
          <div key={bar.name} className="flex items-center gap-2 text-xs">
            <span className="w-11 shrink-0 text-mute">{bar.name}</span>
            <span className="h-2 flex-1 bg-border/50">
              <span
                className={cn("block h-full", bar.cls)}
                style={{ width: `${(bar.value / scale) * 100}%` }}
              />
            </span>
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-mute">
        <span>
          <span className="tabular-nums text-ink">{row.fga}</span> attempts
        </span>
        <span>
          <span className="tabular-nums text-ink">{row.fgm}</span> made
        </span>
        <span>
          <span className="tabular-nums text-ink">{pct(row.share)}</span> of their shots
          {totalFga ? ` (${totalFga} in all)` : ""}
        </span>
      </div>
    </div>
  );
}

/** `seed` is the player-season Ask Full Court just analysed. */
export function ShotAnalysis({ meta, seed }: { meta: Meta; seed?: any }) {
  const avatar = playerAvatar(meta);
  const [a, setA] = useState<PlayerSeason>(emptySelection);
  const [b, setB] = useState<PlayerSeason>(emptySelection);
  const [comparing, setComparing] = useState(false);
  const [mode, setMode] = useState<ChartMode>("hex");
  const [seasonType, setSeasonType] = useState("regular");
  const [shotsA, setShotsA] = useState<any>(null);
  const [shotsB, setShotsB] = useState<any>(null);
  const [zonesA, setZonesA] = useState<any>(null);
  const [zonesB, setZonesB] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  /** The zone the courts are sliced to, shared by both charts so a comparison
   *  stays like-for-like. Picking one opens its breakdown below. */
  const [picked, setPicked] = useState<Zone | null>(null);
  const toggleZone = (z: Zone) => setPicked((prev) => (prev === z ? null : z));

  useEffect(() => {
    if (!seed?.player_id || !seed?.season) return;
    setA({
      playerId: seed.player_id,
      playerName: seed.player_name ?? "",
      season: String(seed.season),
    });
  }, [seed]);

  // Requests come back out of order — scrubbing the season selector fires one
  // per stop, and a slow early one resolving last would overwrite the answer
  // to the selection actually on screen. Each side counts its requests and
  // ignores anything but the newest.
  const latest = useRef({ a: 0, b: 0 });

  const load = (
    side: "a" | "b",
    sel: PlayerSeason,
    setShots: (v: any) => void,
    setZones: (v: any) => void
  ) => {
    const ticket = ++latest.current[side];
    const current = () => latest.current[side] === ticket;
    if (!sel.playerId || !sel.season) {
      setShots(null);
      setZones(null);
      return;
    }
    api
      .shots(sel.playerId, sel.season, mode === "scatter" ? "scatter" : "hex",
             meta.league, seasonType)
      .then((v) => current() && setShots(v))
      .catch(() => current() && setShots(null));
    api
      .shotZones(sel.playerId, sel.season, meta.league, seasonType)
      .then((v) => current() && setZones(v))
      .catch((e) => {
        if (!current()) return;
        setZones(null);
        // "No shots" is the ordinary answer for a player whose season ended
        // before the playoffs, not a failure — and the panel knows his name,
        // where the API only had an id to report.
        setErr(
          /no .* shots/i.test(e.message)
            ? `${sel.playerName || "This player"} has no ` +
              `${(meta.shot_season_types?.[seasonType] ?? seasonType).toLowerCase()} ` +
              `shots in ${formatSeason(sel.season, meta.season_format)}.`
            : e.message
        );
      });
  };

  useEffect(() => {
    setErr(null);
    load("a", a, setShotsA, setZonesA);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a.playerId, a.season, mode, seasonType, meta.league]);

  useEffect(() => {
    if (!comparing) {
      setShotsB(null);
      setZonesB(null);
      return;
    }
    load("b", b, setShotsB, setZonesB);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b.playerId, b.season, comparing, mode, seasonType, meta.league]);

  const zoneRows = zonesA?.zones ?? [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Shot analysis"
          subtitle="Hover a court zone to peek its numbers — click to highlight that section and open its breakdown against the league."
          right={
            <div className="flex items-center gap-5 text-sm">
              <SeasonTypeToggle meta={meta} value={seasonType} onChange={setSeasonType} />
              <span aria-hidden className="h-4 w-px bg-border" />
              {MODES.map(({ value: m, label }) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={cn(
                    "border-b-2 pb-0.5 transition",
                    mode === m ? "border-accent text-ink" : "border-transparent text-mute hover:text-ink"
                  )}
                >
                  {label}
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
            title={zonesA ? `${formatSeason(zonesA.season, meta.season_format)} ${zonesA.player_name}` : "—"}
            subtitle={
              zonesA ? `${zonesA.total_fga} field goal attempts · ${pct(zonesA.fg_pct)} overall` : undefined
            }
          />
          <CardBody>
            <CourtPlot
              meta={meta}
              data={shotsA}
              zones={zonesA?.zones}
              mode={mode}
              height={comparing ? 460 : 560}
              label={
                zonesA
                  ? `${formatSeason(zonesA.season, meta.season_format)} ${zonesA.player_name}`
                  : ""
              }
              totalFga={zonesA?.total_fga}
              selected={picked}
              onSelect={toggleZone}
            />
            {mode === "3d" && <SurfaceLegend />}
          </CardBody>
        </Card>

        {comparing && (
          <Card>
            <CardHeader
              lead={b.playerName ? avatar(b.playerName, 40) : undefined}
              title={zonesB ? `${formatSeason(zonesB.season, meta.season_format)} ${zonesB.player_name}` : "—"}
              subtitle={
                zonesB ? `${zonesB.total_fga} field goal attempts · ${pct(zonesB.fg_pct)} overall` : " "
              }
            />
            <CardBody>
              <CourtPlot
                meta={meta}
                data={shotsB}
                zones={zonesB?.zones}
                mode={mode}
                height={460}
                label={
                  zonesB
                    ? `${formatSeason(zonesB.season, meta.season_format)} ${zonesB.player_name}`
                    : ""
                }
                totalFga={zonesB?.total_fga}
                selected={picked}
                onSelect={toggleZone}
              />
              {mode === "3d" && <SurfaceLegend />}
            </CardBody>
          </Card>
        )}
      </div>

      <Card>
        <CardHeader
          title="Zone breakdown"
          right={
            picked ? (
              <button
                onClick={() => setPicked(null)}
                className="text-xs text-mute transition hover:text-ink"
              >
                Clear selection
              </button>
            ) : undefined
          }
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
                      <tr
                        key={z.zone}
                        onClick={() => toggleZone(z.zone as Zone)}
                        className={cn(
                          "cursor-pointer border-t border-border/60 transition hover:bg-border/40",
                          picked === z.zone && "bg-border/30",
                        )}
                      >
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


/**
 * Which games the chart draws from. Playoff shooting sits against a playoff
 * league average, so the zone comparison stays like for like.
 */
function SeasonTypeToggle({
  meta,
  value,
  onChange,
}: {
  meta: Meta;
  value: string;
  onChange: (v: string) => void;
}) {
  const types = meta.shot_season_types ?? {};
  const keys = Object.keys(types);
  if (keys.length < 2) return null;
  return (
    <div className="flex items-center gap-4">
      {keys.map((k) => (
        <button
          key={k}
          type="button"
          onClick={() => onChange(k)}
          className={cn(
            "border-b-2 pb-0.5 transition",
            k === value ? "border-accent text-ink" : "border-transparent text-mute hover:text-ink"
          )}
        >
          {types[k]}
        </button>
      ))}
    </div>
  );
}
