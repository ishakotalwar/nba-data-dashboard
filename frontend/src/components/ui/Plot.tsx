import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

type Props = {
  data: any[];
  layout?: any;
  config?: any;
  className?: string;
  height?: number | string;
};

/** Trace colors, assigned by trace index. Exported so a custom legend can
 *  match the chart exactly instead of guessing. */
export const COLORWAY = [
  "#ff6a3d", "#4dabff", "#1a9850", "#d73027",
  "#b084ff", "#f2c94c", "#15d0c9", "#f778ba",
];

export const traceColor = (i: number) => COLORWAY[i % COLORWAY.length];

/**
 * Built fresh per render on purpose. Plotly writes computed state (axis `type`,
 * `range`, `_categories`) back onto the layout object it is handed, so a shared
 * module-level object would leak one chart's axis into the next — a categorical
 * bar chart inheriting a numeric season range puts every bar at NaN.
 */
const makeBaseLayout = () => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, sans-serif", color: "#cbd3de", size: 12 },
  colorway: COLORWAY,
  margin: { l: 50, r: 20, t: 30, b: 40 },
  legend: { orientation: "h", y: -0.15, font: { color: "#cbd3de" } },
  xaxis: { gridcolor: "#1f2630", zerolinecolor: "#1f2630", tickcolor: "#2a3240" },
  yaxis: { gridcolor: "#1f2630", zerolinecolor: "#1f2630", tickcolor: "#2a3240" },
});

const baseConfig = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
};

function deepMerge<T>(a: any, b: any): T {
  if (!b) return a;
  const out: any = Array.isArray(a) ? [...(a || [])] : { ...(a || {}) };
  for (const k of Object.keys(b)) {
    const av = a?.[k];
    const bv = b[k];
    if (av && bv && typeof av === "object" && typeof bv === "object" && !Array.isArray(bv)) {
      out[k] = deepMerge(av, bv);
    } else {
      out[k] = bv;
    }
  }
  return out;
}

export function Plot({ data, layout, config, className, height = 420 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const mergedLayout = deepMerge(makeBaseLayout(), layout);
    const mergedConfig = deepMerge(baseConfig, config);
    Plotly.react(ref.current, data, mergedLayout, mergedConfig);
  }, [data, layout, config]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} className={className} style={{ width: "100%", height }} />;
}
