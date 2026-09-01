// Plotly.js shapes for a half-court, in stats.nba.com coordinates
// (tenths of a foot). Hoop at (0, 0); y increases toward half court.
//
// Court, paint, and rim geometry are shared by the NBA and WNBA; only the
// three-point line differs, so it is passed in per league.
const main = "#6b7685";
const three = "#6b7685";
const orange = "#ec7607";

// NBA: 23' 9" arc, 22' 0" corners.
export const NBA_THREE = { arc: 237.5, corner: 220.0 };

function arcPath(cx: number, cy: number, a: number, b: number, a0: number, a1: number, N = 120) {
  const t = (i: number) => a0 + (i / (N - 1)) * (a1 - a0);
  let d = "";
  for (let i = 0; i < N; i++) {
    const x = cx + a * Math.cos(t(i));
    const y = cy + b * Math.sin(t(i));
    d += (i === 0 ? "M " : "L") + x + "," + y;
  }
  return d;
}

/**
 * Half-court shapes for a league's three-point geometry.
 *
 * `corner` must be < `arc`, so the straight corner lines meet the arc; the
 * junction height and the arc's sweep angle both follow from that.
 */
export function buildCourtShapes({ arc, corner }: { arc: number; corner: number }) {
  const junctionY = Math.sqrt(Math.max(arc * arc - corner * corner, 0));
  const theta = Math.atan2(junctionY, corner);

  return [
    // Outer court
    { type: "rect", x0: -250, y0: -52.5, x1: 250, y1: 417.5, line: { color: main, width: 1 }, layer: "below" },
    // Paint
    { type: "rect", x0: -80, y0: -52.5, x1: 80, y1: 137.5, line: { color: main, width: 1 }, layer: "below" },
    { type: "rect", x0: -60, y0: -52.5, x1: 60, y1: 137.5, line: { color: main, width: 1 }, layer: "below" },
    // FT circle
    { type: "circle", x0: -60, y0: 77.5, x1: 60, y1: 197.5, line: { color: main, width: 1 }, layer: "below" },
    // FT line
    { type: "line", x0: -60, y0: 137.5, x1: 60, y1: 137.5, line: { color: main, width: 1 }, layer: "below" },
    // Backboard + rim
    { type: "rect", x0: -30, y0: -12.5, x1: 30, y1: -11.5, line: { color: orange, width: 1 }, fillcolor: orange },
    { type: "circle", x0: -7.5, y0: -7.5, x1: 7.5, y1: 7.5, line: { color: orange, width: 1 } },
    // Restricted area arc
    { type: "path", path: arcPath(0, 0, 40, 40, 0, Math.PI), line: { color: main, width: 1 }, layer: "below" },
    // Three-point arc
    { type: "path", path: arcPath(0, 0, arc, arc, theta, Math.PI - theta), line: { color: three, width: 1 }, layer: "below" },
    // Corner 3s
    { type: "line", x0: -corner, y0: -52.5, x1: -corner, y1: junctionY, line: { color: three, width: 1 }, layer: "below" },
    { type: "line", x0: corner, y0: -52.5, x1: corner, y1: junctionY, line: { color: three, width: 1 }, layer: "below" },
    // Half-court arc
    { type: "path", path: arcPath(0, 417.5, 60, 60, -0, -Math.PI), line: { color: main, width: 1 }, layer: "below" },
  ] as const;
}

/** Default (NBA) shapes, kept for callers that don't pass geometry. */
export const courtShapes = buildCourtShapes(NBA_THREE);

export const courtAxis = {
  xaxis: {
    range: [-260, 260],
    showgrid: false,
    zeroline: false,
    showticklabels: false,
    fixedrange: true,
    showline: false,
  },
  yaxis: {
    range: [-60, 425],
    scaleanchor: "x",
    scaleratio: 1,
    showgrid: false,
    zeroline: false,
    showticklabels: false,
    fixedrange: true,
    showline: false,
  },
};

// --- Zones -----------------------------------------------------------------
// The same eight regions the backend sorts shots into (`backend/routers/
// shots.py`), rebuilt here as polygons so the court itself can be hovered and
// clicked. Both sides derive them from the league's geometry, so they agree.

export type CourtGeometry = {
  arc: number;
  corner: number;
  rim: number;
  paint_width: number;
  paint_depth: number;
  paint_near: number;
  wing_angle: number;
};

export const NBA_COURT: CourtGeometry = {
  ...NBA_THREE, rim: 40, paint_width: 80, paint_depth: 137.5, paint_near: 80,
  wing_angle: 60,
};

export const ZONE_ORDER = [
  "Rim", "Paint", "Short Midrange", "Long Midrange",
  "Left Corner 3", "Left Wing 3", "Top of Arc 3", "Right Wing 3", "Right Corner 3",
] as const;

export type Zone = (typeof ZONE_ORDER)[number];

const BASELINE = -52.5;
const FAR = 470;           // past the top of the chart, so 3-point zones close
const deg = (d: number) => (d * Math.PI) / 180;

/** Which zone a point falls in. Mirrors `classify()` on the backend. */
export function zoneOf(x: number, y: number, c: CourtGeometry): Zone {
  const dist = Math.hypot(x, y);
  const junctionY = Math.sqrt(Math.max(c.arc * c.arc - c.corner * c.corner, 0));
  const isCorner3 = Math.abs(x) >= c.corner && y <= junctionY;
  if (isCorner3) return x < 0 ? "Left Corner 3" : "Right Corner 3";
  if (dist >= c.arc) {
    const angle = (Math.atan2(y, x) * 180) / Math.PI;
    if (angle < 90 - c.wing_angle / 2) return "Right Wing 3";
    if (angle > 90 + c.wing_angle / 2) return "Left Wing 3";
    return "Top of Arc 3";
  }
  if (dist <= c.rim) return "Rim";
  if (Math.abs(x) <= c.paint_width && y <= c.paint_depth) {
    return dist <= c.paint_near ? "Paint" : "Short Midrange";
  }
  return "Long Midrange";
}

function arcPoints(r: number, a0: number, a1: number, n = 48) {
  return Array.from({ length: n }, (_, i) => {
    const t = a0 + (i / (n - 1)) * (a1 - a0);
    return [r * Math.cos(t), r * Math.sin(t)] as [number, number];
  });
}

/**
 * Closed outline of every zone, as x/y arrays ready for a filled trace.
 *
 * The three-point regions are closed off the top and sides of the chart rather
 * than at a real boundary — a shot from 40 feet still belongs to the arc it
 * came from.
 */
export function zonePolygons(c: CourtGeometry): { zone: Zone; x: number[]; y: number[] }[] {
  const junctionY = Math.sqrt(Math.max(c.arc * c.arc - c.corner * c.corner, 0));
  const theta = Math.atan2(junctionY, c.corner);
  const halfWing = deg(c.wing_angle / 2);
  const wall = 262;  // just outside the chart's x range

  const poly = (zone: Zone, pts: [number, number][]) => ({
    zone,
    x: pts.map((p) => p[0]),
    y: pts.map((p) => p[1]),
  });

  const rim = arcPoints(c.rim, Math.PI, -Math.PI, 60);
  const lane: [number, number][] = [
    [-c.paint_width, BASELINE], [-c.paint_width, c.paint_depth],
    [c.paint_width, c.paint_depth], [c.paint_width, BASELINE],
  ];
  // The near half of the lane: a disk around the rim, squared off where it
  // runs past the baseline.
  const cut = Math.asin(Math.min(1, -BASELINE / c.paint_near));
  const paint: [number, number][] = [
    ...arcPoints(c.paint_near, -cut, Math.PI + cut, 48),
  ];
  // Long Midrange is everything inside the arc that is not the paint: traced down
  // the arc, then back along the outside of the lane.
  const midrange: [number, number][] = [
    [-c.corner, BASELINE], [-c.corner, junctionY],
    ...arcPoints(c.arc, Math.PI - theta, theta),
    [c.corner, junctionY], [c.corner, BASELINE],
    [c.paint_width, BASELINE], [c.paint_width, c.paint_depth],
    [-c.paint_width, c.paint_depth], [-c.paint_width, BASELINE],
  ];
  const corner = (side: -1 | 1): [number, number][] => [
    [side * c.corner, BASELINE], [side * c.corner, junctionY],
    [side * wall, junctionY], [side * wall, BASELINE],
  ];
  const band = (a0: number, a1: number, corners: [number, number][]): [number, number][] =>
    [...arcPoints(c.arc, a0, a1), ...corners];

  return [
    poly("Rim", rim),
    poly("Paint", paint),
    poly("Short Midrange", lane),
    poly("Long Midrange", midrange),
    poly("Left Corner 3", corner(-1)),
    poly("Left Wing 3", band(Math.PI - theta, Math.PI / 2 + halfWing,
                             [[-wall, FAR], [-wall, junctionY]])),
    poly("Top of Arc 3", band(Math.PI / 2 + halfWing, Math.PI / 2 - halfWing,
                              [[wall, FAR], [-wall, FAR]])),
    poly("Right Wing 3", band(Math.PI / 2 - halfWing, theta,
                              [[wall, junctionY], [wall, FAR]])),
    poly("Right Corner 3", corner(1)),
  ];
}

/** Axis ranges that frame one zone, with a little air around it. */
export function zoneBounds(zone: Zone, c: CourtGeometry) {
  const p = zonePolygons(c).find((z) => z.zone === zone);
  if (!p) return null;
  const pad = 22;
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  return {
    x: [clamp(Math.min(...p.x) - pad, -262, 262), clamp(Math.max(...p.x) + pad, -262, 262)],
    y: [clamp(Math.min(...p.y) - pad, -60, 430), clamp(Math.max(...p.y) + pad, -60, 430)],
  };
}

/**
 * The same court as `buildCourtShapes`, but as plain polylines — a 3D scene
 * has no shape layer, so the floor has to be drawn as line traces.
 */
export function courtLines(c: CourtGeometry): { x: number[]; y: number[] }[] {
  const junctionY = Math.sqrt(Math.max(c.arc * c.arc - c.corner * c.corner, 0));
  const theta = Math.atan2(junctionY, c.corner);
  const line = (pts: [number, number][]) => ({ x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) });
  const fromArc = (pts: [number, number][]) => line(pts);
  const box = (x0: number, y0: number, x1: number, y1: number): [number, number][] =>
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];

  return [
    line(box(-250, -52.5, 250, 417.5)),          // floor
    line(box(-c.paint_width, -52.5, c.paint_width, c.paint_depth)),  // lane
    fromArc(arcPoints(c.rim, 0, Math.PI, 40).map(([x, y]) => [x, y])),  // restricted arc
    fromArc(arcPoints(60, 0, Math.PI, 40).map(([x, y]) => [x, y + c.paint_depth])), // FT circle
    fromArc(arcPoints(c.arc, theta, Math.PI - theta, 80).map(([x, y]) => [x, y])),  // arc
    line([[-c.corner, -52.5], [-c.corner, junctionY]]),
    line([[c.corner, -52.5], [c.corner, junctionY]]),
  ];
}
