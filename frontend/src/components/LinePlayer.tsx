/**
 * A dribbling player — original artwork, generated rather than traced.
 *
 * The figure is one closed silhouette. A skeleton of joints is defined below;
 * each limb is a chain of those joints with a half-width at each, and the
 * outline is computed by walking the *outside* of the whole body once — down
 * one arm and back, round a leg and back, and so on. Because it is a single
 * contour there are no seams where parts meet, which is what separates line
 * art from an articulated puppet.
 *
 * Stroked in `currentColor`, so it follows the page theme.
 */

type P = [number, number];
type Node = { p: P; w: number };

const sub = (a: P, b: P): P => [a[0] - b[0], a[1] - b[1]];
const add = (a: P, b: P): P => [a[0] + b[0], a[1] + b[1]];
const mul = (a: P, k: number): P => [a[0] * k, a[1] * k];
const norm = (a: P): P => {
  const l = Math.hypot(a[0], a[1]) || 1;
  return [a[0] / l, a[1] / l];
};
const perp = (a: P): P => [-a[1], a[0]];
const f = (n: number) => n.toFixed(1);

/**
 * Offset points running along one side of a joint chain.
 *
 * At an interior joint the two adjacent segments disagree about which way is
 * "sideways", so the averaged direction is used — a cheap miter that keeps the
 * outline smooth through a bend like an elbow or a knee.
 */
function side(chain: Node[], sign: 1 | -1): P[] {
  return chain.map((node, i) => {
    const before = i > 0 ? norm(sub(node.p, chain[i - 1].p)) : null;
    const after = i < chain.length - 1 ? norm(sub(chain[i + 1].p, node.p)) : null;
    const dir = before && after ? norm(add(before, after)) : (after ?? before)!;
    return add(node.p, mul(perp(dir), node.w * sign));
  });
}

/** A smooth polyline: vertices become controls, midpoints become anchors. */
function through(points: P[], moveFirst = false): string {
  if (points.length === 0) return "";
  let d = moveFirst ? `M${f(points[0][0])},${f(points[0][1])}` : ` L${f(points[0][0])},${f(points[0][1])}`;
  for (let i = 1; i < points.length; i++) {
    const c = points[i - 1];
    const nxt = points[i];
    const anchor: P = i === points.length - 1 ? nxt : [(c[0] + nxt[0]) / 2, (c[1] + nxt[1]) / 2];
    d += ` Q${f(c[0])},${f(c[1])} ${f(anchor[0])},${f(anchor[1])}`;
  }
  return d;
}

/** Round cap across the end of a chain, from one side to the other. */
function cap(end: Node, from: P): string {
  const d = norm(sub(end.p, from));
  const a = add(end.p, mul(perp(d), -end.w));
  return ` A${f(end.w)},${f(end.w)} 0 0 1 ${f(a[0])},${f(a[1])}`;
}

/** A connector that bows outward by `amount`, so the torso reads as a body
 *  rather than a flat panel between two offset points. */
function bow(from: P, to: P, amount: number): string {
  const mid: P = [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2];
  const n = perp(norm(sub(to, from)));
  const c = add(mid, mul(n, amount));
  return ` Q${f(c[0])},${f(c[1])} ${f(to[0])},${f(to[1])}`;
}

// --- the pose -------------------------------------------------------------
// Driving right, low dribble, off arm out for balance.
const HEAD: P = [356, 118];
const SH_NEAR: P = [305, 196];
const SH_FAR: P = [404, 188];
const ELB_NEAR: P = [246, 262];
const WRI_NEAR: P = [224, 330];
const ELB_FAR: P = [486, 206];
const WRI_FAR: P = [566, 232];
const HIP_NEAR: P = [318, 340];
const HIP_FAR: P = [386, 334];
const KNEE_FRONT: P = [276, 438];
const ANK_FRONT: P = [214, 524];
const KNEE_BACK: P = [414, 436];
const ANK_BACK: P = [436, 532];
// Toes carry on from the ankle at an angle, so the contour turns a corner
// there and the leg reads as ending in a foot rather than a stump.
const TOE_FRONT: P = [166, 556];
const TOE_BACK: P = [488, 556];
const BALL: P = [188, 388];
const BALL_R = 54;

const armNear: Node[] = [{ p: SH_NEAR, w: 21 }, { p: ELB_NEAR, w: 15 }, { p: WRI_NEAR, w: 11 }];
const armFar: Node[] = [{ p: SH_FAR, w: 21 }, { p: ELB_FAR, w: 15 }, { p: WRI_FAR, w: 11 }];
const legFront: Node[] = [
  { p: HIP_NEAR, w: 30 },
  { p: KNEE_FRONT, w: 20 },
  { p: ANK_FRONT, w: 13 },
  { p: TOE_FRONT, w: 8 },
];
const legBack: Node[] = [
  { p: HIP_FAR, w: 30 },
  { p: KNEE_BACK, w: 20 },
  { p: ANK_BACK, w: 13 },
  { p: TOE_BACK, w: 8 },
];

/** Out along one side of a limb, around the end, and back up the other. */
function limbLoop(chain: Node[], outer: 1 | -1): string {
  const out = side(chain, outer);
  const back = side(chain, (outer * -1) as 1 | -1).reverse();
  return through(out) + cap(chain[chain.length - 1], chain[chain.length - 2].p) + through(back);
}

export function LinePlayer() {
  const startFar = side(armFar, -1)[0]; // top of the far shoulder
  const armpitFar = side(armFar, 1)[0];
  const armpitNear = side(armNear, -1)[0];
  const hipFarOut = side(legBack, -1)[0];
  const crotchBack = side(legBack, 1)[0];
  const crotchFront = side(legFront, -1)[0];
  const hipNearOut = side(legFront, 1)[0];
  const shoulderNear = side(armNear, 1)[0];

  const body =
    `M${f(startFar[0])},${f(startFar[1])}` +
    limbLoop(armFar, -1) +              // out along the extended arm, back to the armpit
    bow(armpitFar, hipFarOut, 16) +     // far flank, bowed for the ribcage
    limbLoop(legBack, -1) +             // down the trailing leg and back
    bow(crotchBack, crotchFront, -14) + // the seat, hollowed between the legs
    limbLoop(legFront, 1) +             // down the front leg and back up
    bow(hipNearOut, armpitNear, 16) +   // near flank
    limbLoop(armNear, 1) +              // down the dribbling arm and back
    bow(shoulderNear, [HEAD[0] - 26, HEAD[1] + 40], 10) + // near shoulder into the neck
    bow([HEAD[0] - 26, HEAD[1] + 40], [HEAD[0] + 22, HEAD[1] + 42], -12) + // under the jaw
    bow([HEAD[0] + 22, HEAD[1] + 42], startFar, 10) +     // far shoulder line
    "Z";

  return (
    <svg
      viewBox="120 50 500 545"
      role="presentation"
      aria-hidden
      className="mx-auto my-6 h-64 w-auto text-ink"
      fill="none"
      stroke="currentColor"
      strokeWidth={3.2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={body} />
      <ellipse cx={HEAD[0]} cy={HEAD[1]} rx={40} ry={46} />
      <circle cx={BALL[0]} cy={BALL[1]} r={BALL_R} />
      <path
        d={[
          `M${BALL[0] - BALL_R},${BALL[1] - 5} Q${BALL[0]},${BALL[1] + 15} ${BALL[0] + BALL_R},${BALL[1] - 5}`,
          `M${BALL[0] - 33},${BALL[1] - 43} Q${BALL[0] - 5},${BALL[1]} ${BALL[0] - 29},${BALL[1] + 44}`,
          `M${BALL[0] + 29},${BALL[1] - 44} Q${BALL[0] + 7},${BALL[1]} ${BALL[0] + 33},${BALL[1] + 43}`,
        ].join(" ")}
      />
    </svg>
  );
}
