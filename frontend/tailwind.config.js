/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0d10",
        panel: "#12161c",
        border: "#1f2630",
        ink: "#e6eaf0",
        mute: "#8a94a2",
        accent: "#ff6a3d",
        accent2: "#4dabff",
        good: "#1a9850",
        bad: "#d73027",
      },
      fontFamily: {
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
