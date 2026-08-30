import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

type Props = {
  data: any[];
  layout?: any;
  config?: any;
  className?: string;
  height?: number | string;
};

const baseLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, sans-serif", color: "#cbd3de", size: 12 },
  colorway: ["#ff6a3d", "#4dabff", "#1a9850", "#d73027", "#b084ff", "#f2c94c", "#15d0c9", "#f778ba"],
  margin: { l: 50, r: 20, t: 30, b: 40 },
  legend: { orientation: "h", y: -0.15, font: { color: "#cbd3de" } },
  xaxis: { gridcolor: "#1f2630", zerolinecolor: "#1f2630", tickcolor: "#2a3240" },
  yaxis: { gridcolor: "#1f2630", zerolinecolor: "#1f2630", tickcolor: "#2a3240" },
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

export function Plot({ data, layout, config, className, height = 420 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const mergedLayout = deepMerge(baseLayout, layout);
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
