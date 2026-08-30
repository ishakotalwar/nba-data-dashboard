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
