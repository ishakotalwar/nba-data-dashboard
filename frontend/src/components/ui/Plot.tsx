import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

import { themeColor, useTheme } from "@/lib/theme";

type Props = {
  data: any[];
  layout?: any;
  config?: any;
  className?: string;
  height?: number | string;
  /**
   * Shown centred when `data` is empty. The chart still renders its axes, so
   * the shape of the answer is visible before anything is selected.
   */
  placeholder?: string;
  /** Fired for a clicked point: its trace name, its data coordinates and where
   *  on screen the click landed, for charts you can drill into. */
  onPointClick?: (point: {
    name?: string;
    x?: number;
    y?: number;
    clientX: number;
    clientY: number;
  }) => void;
  /** Fired with the hovered point's data coordinates. A 3D scene never emits
   *  clicks, so a caller that needs them tracks the hover instead. */
  onPointHover?: (point: { x?: number; y?: number }) => void;
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
const makeBaseLayout = () => {
  // Plotly draws to canvas and cannot read CSS variables, so the palette is
  // sampled at render time and the chart is redrawn when the theme changes.
  const ink = themeColor("ink", "#cbd3de");
  const grid = themeColor("grid", "#1f2630");
  const axis = themeColor("axis", "#2a3240");
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, sans-serif", color: ink, size: 12 },
    colorway: COLORWAY,
    margin: { l: 50, r: 20, t: 30, b: 40 },
    legend: { orientation: "h", y: -0.15, font: { color: ink } },
    xaxis: { gridcolor: grid, zerolinecolor: grid, tickcolor: axis },
    yaxis: { gridcolor: grid, zerolinecolor: grid, tickcolor: axis },
  };
};

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

export function Plot({ data, layout, config, className, height = 420, placeholder,
                      onPointClick, onPointHover }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useTheme();
  useEffect(() => {
    if (!ref.current) return;
    const empty = !data || data.length === 0;
    const mergedLayout: any = deepMerge(makeBaseLayout(), layout);
    const mergedConfig: any = deepMerge(baseConfig, config);
    if (empty && placeholder) {
      mergedLayout.annotations = [
        ...(mergedLayout.annotations ?? []),
        {
          xref: "paper", yref: "paper", x: 0.5, y: 0.5,
          xanchor: "center", yanchor: "middle",
          text: placeholder, showarrow: false,
          font: { color: themeColor("mute", "#6b7685"), size: 13 },
        },
      ];
      // Nothing to zoom or download yet.
      mergedConfig.displayModeBar = false;
      mergedLayout.showlegend = false;
    }
    const el = ref.current;
    // `react` resolves a tick later. Switching view in that gap unmounts the
    // chart and purges the node, and everything below then runs against an
    // element whose Plotly internals are gone — which throws where nobody is
    // listening for it.
    let stale = false;
    Plotly.react(el, data, mergedLayout, mergedConfig).then(() => {
      if (stale || !(el as any)._fullLayout) return;
      // Plotly keeps its own listener list, so clear ours before re-adding it
      // or a click fires once per render that has happened.
      const withEvents = el as any;
      withEvents.removeAllListeners?.("plotly_click");
      withEvents.removeAllListeners?.("plotly_hover");
      if (onPointHover) {
        withEvents.on("plotly_hover", (e: any) => {
          const hit = e?.points?.[0];
          if (hit) onPointHover({ x: hit.x, y: hit.y });
        });
      }
      if (onPointClick) {
        withEvents.on("plotly_click", (e: any) => {
          const hit = e?.points?.[0];
          if (!hit) return;
          onPointClick({
            name: hit.data?.name,
            x: typeof hit.x === "number" ? hit.x : undefined,
            y: typeof hit.y === "number" ? hit.y : undefined,
            clientX: e?.event?.clientX ?? 0,
            clientY: e?.event?.clientY ?? 0,
          });
        });
      }
      // `responsive` only reacts to window resizes. When the container itself
      // changes size — a height prop that grows once data arrives — Plotly keeps
      // its first measurement and the chart is drawn short inside a taller box.
      Plotly.Plots.resize(el);
    });
    return () => {
      stale = true;
    };
  }, [data, layout, config, placeholder, theme, onPointClick, onPointHover]);

  // Same problem from the other direction: the card can be resized by layout
  // changes that never touch this component's props.
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if ((el as any)._fullLayout) Plotly.Plots.resize(el);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} className={className} style={{ width: "100%", height }} />;
}
