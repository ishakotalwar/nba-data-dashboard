import { useEffect, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { api, type LeagueInfo, type LeagueKey, type Meta } from "./lib/api";
import { cn } from "./lib/cn";
import { Compare } from "./components/panels/Compare";
import { Trends } from "./components/panels/Trends";
import { Percentiles } from "./components/panels/Percentiles";
import { Similar } from "./components/panels/Similar";
import { GameLog } from "./components/panels/GameLog";
import { AgeCurves } from "./components/panels/AgeCurves";
import { Teams } from "./components/panels/Teams";
import { ShotChart } from "./components/panels/ShotChart";

const TABS = [
  { v: "compare", label: "Compare" },
  { v: "trends", label: "Trends" },
  { v: "pct", label: "Percentiles" },
  { v: "similar", label: "Similar Players" },
  { v: "gamelog", label: "Game Log" },
  { v: "age", label: "Age Curves" },
  { v: "teams", label: "Teams" },
  { v: "shots", label: "Shot Chart" },
] as const;

export default function App() {
  const [leagues, setLeagues] = useState<LeagueInfo[] | null>(null);
  const [league, setLeague] = useState<LeagueKey | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // Which leagues exist, and which one to open on.
  useEffect(() => {
    api
      .leagues()
      .then(({ leagues, default: dflt }) => {
        setLeagues(leagues);
        const first = leagues.find((l) => l.available)?.key ?? dflt;
        setLeague(first);
      })
      .catch((e) => setErr(e.message));
  }, []);

  // Reload metadata whenever the league changes.
  useEffect(() => {
    if (!league) return;
    setMeta(null);
    setErr(null);
    api.meta(league).then(setMeta).catch((e) => setErr(e.message));
  }, [league]);

  if (err) return <Bootstrap state="error" msg={err} leagues={leagues} league={league} onLeague={setLeague} />;
  if (!meta || !league) return <Bootstrap state="loading" />;

  return (
    <div className="min-h-screen bg-bg">
      <header className="sticky top-0 z-20 border-b border-border bg-bg/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div>
              <div className="text-[15px] font-semibold leading-tight">Hoops Data Dashboard</div>
              <div className="text-xs text-mute">
                {meta.league_label} · {meta.players.length} players ·{" "}
                {meta.seasons.length} season{meta.seasons.length === 1 ? "" : "s"}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs text-mute">
            <LeagueToggle leagues={leagues} active={league} onChange={setLeague} />
            <span className="chip">
              {meta.seasons.length > 1
                ? `${meta.seasons[0]}–${meta.seasons[meta.seasons.length - 1]}`
                : meta.seasons[0]}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        {/* Keying on league remounts every panel, clearing player/team picks
            that don't exist in the league being switched to. */}
        <Tabs.Root defaultValue="compare" key={league}>
          <Tabs.List className="no-scrollbar mb-5 flex gap-1 overflow-x-auto rounded-xl border border-border bg-panel p-1">
            {TABS.map((t) => (
              <Tabs.Trigger
                key={t.v}
                value={t.v}
                className={cn(
                  "whitespace-nowrap rounded-lg px-3.5 py-1.5 text-sm text-mute transition",
                  "data-[state=active]:bg-accent data-[state=active]:text-black data-[state=active]:shadow",
                  "hover:text-ink"
                )}
              >
                {t.label}
              </Tabs.Trigger>
            ))}
          </Tabs.List>

          <Tabs.Content value="compare"><Compare meta={meta} /></Tabs.Content>
          <Tabs.Content value="trends"><Trends meta={meta} /></Tabs.Content>
          <Tabs.Content value="pct"><Percentiles meta={meta} /></Tabs.Content>
          <Tabs.Content value="similar"><Similar meta={meta} /></Tabs.Content>
          <Tabs.Content value="gamelog"><GameLog meta={meta} /></Tabs.Content>
          <Tabs.Content value="age"><AgeCurves meta={meta} /></Tabs.Content>
          <Tabs.Content value="teams"><Teams meta={meta} /></Tabs.Content>
          <Tabs.Content value="shots"><ShotChart meta={meta} /></Tabs.Content>
        </Tabs.Root>
      </main>
    </div>
  );
}

function LeagueToggle({
  leagues,
  active,
  onChange,
}: {
  leagues: LeagueInfo[] | null;
  active: LeagueKey;
  onChange: (k: LeagueKey) => void;
}) {
  if (!leagues || leagues.length < 2) return null;
  return (
    <div className="flex items-center gap-0.5 rounded-lg border border-border bg-panel p-0.5">
      {leagues.map((l) => (
        <button
          key={l.key}
          onClick={() => onChange(l.key)}
          disabled={!l.available && l.key !== active}
          title={l.available ? `Show ${l.label} data` : `No ${l.label} data yet — run the ETL for this league`}
          className={cn(
            "rounded-md px-2.5 py-1 text-xs font-medium transition",
            l.key === active
              ? "bg-accent text-black shadow"
              : l.available
              ? "text-mute hover:text-ink"
              : "cursor-not-allowed text-mute/40"
          )}
        >
          {l.label}
        </button>
      ))}
    </div>
  );
}

function Bootstrap({
  state,
  msg,
  leagues,
  league,
  onLeague,
}: {
  state: "loading" | "error";
  msg?: string;
  leagues?: LeagueInfo[] | null;
  league?: LeagueKey | null;
  onLeague?: (k: LeagueKey) => void;
}) {
  return (
    <div className="grid min-h-screen place-items-center bg-bg text-ink">
      <div className="card max-w-md p-6 text-center">
        {state === "loading" ? (
          <>
            <div className="mb-2 text-lg font-semibold">Loading dashboard…</div>
            <div className="text-sm text-mute">Reading local parquet data</div>
          </>
        ) : (
          <>
            <div className="mb-2 text-lg font-semibold text-bad">Couldn't load data</div>
            <div className="text-sm text-mute">{msg}</div>
            <div className="mt-4 text-xs text-mute">
              If the API isn't running, start it with:{" "}
              <code className="rounded bg-border/60 px-1.5 py-0.5">uvicorn backend.main:app --reload</code>
            </div>
            {leagues && league && onLeague && (
              <div className="mt-4 flex justify-center">
                <LeagueToggle leagues={leagues} active={league} onChange={onLeague} />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
