import { Fragment, useEffect, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { api, type LeagueInfo, type LeagueKey, type Meta } from "./lib/api";
import { cn } from "./lib/cn";
import { Players } from "./components/panels/Players";
import { Compare } from "./components/panels/Compare";
import { Similar } from "./components/panels/Similar";
import { TeamsSection } from "./components/panels/TeamsSection";
import { Explorer } from "./components/panels/Explorer";
import { ShotAnalysis } from "./components/panels/ShotAnalysis";
import { formatSeason } from "@/lib/season";

/** One flat nav. `group` only draws a separator — it is not a second click. */
const TABS = [
  { v: "players", label: "Players", group: "player" },
  { v: "compare", label: "Compare", group: "player" },
  { v: "similar", label: "Similarity", group: "player" },
  { v: "shots", label: "Shot Analysis", group: "player" },
  { v: "teams", label: "Teams", group: "team" },
  { v: "explorer", label: "Explorer", group: "team" },
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
              <div className="text-[15px] font-semibold leading-tight">Full Court</div>
              <div className="text-xs text-mute">
                {meta.players.length.toLocaleString()} players ·{" "}
                {meta.seasons.length > 1
                  ? `${formatSeason(meta.seasons[0], meta.season_format)}–${formatSeason(
                      meta.seasons[meta.seasons.length - 1],
                      meta.season_format
                    )}`
                  : formatSeason(meta.seasons[0], meta.season_format)}
              </div>
            </div>
          </div>
          <LeagueToggle leagues={leagues} active={league} onChange={setLeague} />
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        <Tabs.Root defaultValue="players" key={league}>
          <Tabs.List className="no-scrollbar mb-6 flex items-center gap-5 overflow-x-auto border-b border-border">
            {TABS.map((t, i) => (
              <Fragment key={t.v}>
                {i > 0 && TABS[i - 1].group !== t.group && (
                  <span aria-hidden className="h-4 w-px shrink-0 bg-border" />
                )}
                <Tabs.Trigger
                  value={t.v}
                  className={cn(
                   "-mb-px whitespace-nowrap border-b-2 border-transparent pb-2.5 text-sm text-mute transition",
                   "data-[state=active]:border-accent data-[state=active]:text-ink",
                   "hover:text-ink"
                  )}
                >
                  {t.label}
                </Tabs.Trigger>
              </Fragment>
            ))}
          </Tabs.List>

          <Tabs.Content value="players"><Players meta={meta} /></Tabs.Content>
          <Tabs.Content value="compare"><Compare meta={meta} /></Tabs.Content>
          <Tabs.Content value="similar"><Similar meta={meta} /></Tabs.Content>
          <Tabs.Content value="shots"><ShotAnalysis meta={meta} /></Tabs.Content>
          <Tabs.Content value="teams"><TeamsSection meta={meta} /></Tabs.Content>
          <Tabs.Content value="explorer"><Explorer meta={meta} /></Tabs.Content>
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
    <div className="flex items-center gap-4">
      {leagues.map((l) => (
        <button
          key={l.key}
          onClick={() => onChange(l.key)}
          disabled={!l.available && l.key !== active}
          title={l.available ? `Show ${l.label} data` : `No ${l.label} data yet — run the ETL for this league`}
          className={cn(
           "border-b-2 pb-0.5 text-sm font-medium transition",
            l.key === active
              ? "border-accent text-ink"
              : l.available
              ? "border-transparent text-mute hover:text-ink"
              : "cursor-not-allowed border-transparent text-mute/40"
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
