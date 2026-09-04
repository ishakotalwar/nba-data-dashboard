/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Every colour resolves through a CSS variable (defined per theme in
      // index.css), so light/dark is a single attribute on <html> rather than
      // a conditional in each component. <alpha-value> keeps bg-panel/60 etc.
      colors: {
        bg: "rgb(var(--c-bg) / <alpha-value>)",
        panel: "rgb(var(--c-panel) / <alpha-value>)",
        border: "rgb(var(--c-border) / <alpha-value>)",
        ink: "rgb(var(--c-ink) / <alpha-value>)",
        mute: "rgb(var(--c-mute) / <alpha-value>)",
        accent: "rgb(var(--c-accent) / <alpha-value>)",
        accent2: "rgb(var(--c-accent2) / <alpha-value>)",
        good: "rgb(var(--c-good) / <alpha-value>)",
        bad: "rgb(var(--c-bad) / <alpha-value>)",
        onAccent: "rgb(var(--c-on-accent) / <alpha-value>)",
      },
      fontFamily: {
        display: [
          "Space Grotesk",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "sans-serif",
        ],
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
      },
      boxShadow: {
        card: "0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px rgba(0,0,0,0.3)",
      },
    },
  },
  plugins: [],
};
