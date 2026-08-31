/** Light/dark theme, stored per browser.
 *
 * The palette itself lives in index.css as CSS custom properties, so switching
 * themes is one attribute on <html> rather than a re-render of every component.
 * This module owns only the choice: what it is, how to change it, and how to
 * subscribe so canvas-based things (Plotly) can redraw with new colors.
 */
import { useSyncExternalStore } from "react";

export type Theme = "dark" | "light";

const STORAGE_KEY = "full-court-theme";

function systemTheme(): Theme {
  if (typeof window === "undefined" || !window.matchMedia) return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function readStoredTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    // Private mode or blocked storage — fall back to the system preference.
  }
  return systemTheme();
}

function apply(theme: Theme) {
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
}

let current: Theme = typeof document === "undefined" ? "dark" : readStoredTheme();
const listeners = new Set<() => void>();

export function setTheme(theme: Theme) {
  current = theme;
  apply(theme);
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Not being able to remember the choice shouldn't break the toggle.
  }
  listeners.forEach((l) => l());
}

export function toggleTheme() {
  setTheme(current === "dark" ? "light" : "dark");
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useTheme(): Theme {
  return useSyncExternalStore(
    subscribe,
    () => current,
    () => "dark" as Theme,
  );
}

/** Read one palette value as a CSS color, for canvas libraries that can't use
 *  CSS variables. Returns an rgb() string built from the "R G B" triplet. */
export function themeColor(name: string, fallback = "#888"): string {
  if (typeof document === "undefined") return fallback;
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(`--c-${name}`)
    .trim();
  return raw ? `rgb(${raw})` : fallback;
}

/** Applied once at startup; index.html also sets it before first paint so the
 *  page never flashes the wrong theme. */
export function initTheme() {
  apply(current);
}
