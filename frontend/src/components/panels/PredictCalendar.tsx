import { useEffect, useMemo, useState } from "react";

import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { ErrorNotice } from "@/components/ui/ErrorNotice";
import { api, type Meta } from "@/lib/api";
import { cn } from "@/lib/cn";

const WEEKDAYS = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];
const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

/** "2026-10-20" -> {y, m, d}, parsed as a plain calendar date. Going through
 *  `new Date("2026-10-20")` would parse as UTC and shift the day for anyone
 *  west of Greenwich. */
function parseDay(iso: string) {
  const [y, m, d] = iso.split("-").map(Number);
  return { y, m: m - 1, d };
}

const toISO = (y: number, m: number, d: number) =>
  `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;

type DayInfo = { date: string; games: number; upcoming: number };

/**
 * Browse the schedule a month at a time and see what the model makes of each
 * game. The schedule is the one thing the historical data cannot supply — it
 * comes from `etl/schedule_etl.py`.
 */
export function PredictCalendar({ meta }: { meta: Meta }) {
  const [index, setIndex] = useState<{ dates: DayInfo[]; default_date: string; today: string } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [month, setMonth] = useState<{ y: number; m: number } | null>(null);
  const [day, setDay] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [openGame, setOpenGame] = useState<string | null>(null);

  useEffect(() => {
    setIndex(null);
    setSelected(null);
    setMonth(null);
    setDay(null);
    setErr(null);
    api
      .predictCalendar(meta.league)
      .then((d) => {
        setIndex(d);
        if (d.default_date) {
          const { y, m } = parseDay(d.default_date);
          setSelected(d.default_date);
          setMonth({ y, m });
        }
      })
      .catch((e) => setErr(e.message));
  }, [meta.league]);

  useEffect(() => {
    if (!selected) return;
    setOpenGame(null);
    api.predictSchedule(meta.league, selected).then(setDay).catch(() => setDay(null));
  }, [selected, meta.league]);

  /** date -> counts, for marking the grid. */
  const byDate = useMemo(() => {
    const map = new Map<string, DayInfo>();
    (index?.dates ?? []).forEach((d) => map.set(d.date, d));
    return map;
  }, [index]);

  if (err) return <ErrorNotice message={err} />;
  if (!index || !month) {
    return (
      <Card>
        <CardBody>
          <div className="text-sm text-mute">Loading the schedule…</div>
        </CardBody>
      </Card>
    );
  }

  const first = new Date(month.y, month.m, 1);
  const daysInMonth = new Date(month.y, month.m + 1, 0).getDate();
  const leading = first.getDay();
  const cells: (number | null)[] = [
    ...Array<null>(leading).fill(null),
    ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
  ];
  while (cells.length % 7 !== 0) cells.push(null);

  const shiftMonth = (delta: number) => {
    const next = new Date(month.y, month.m + delta, 1);
    setMonth({ y: next.getFullYear(), m: next.getMonth() });
  };

  const games: any[] = day?.games ?? [];

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr] lg:items-start">
      <Card>
        <CardHeader
          title={`${MONTHS[month.m]} ${month.y}`}
          right={
            <span className="flex gap-1">
              <button
                type="button"
                className="btn btn-ghost px-2 py-1 text-xs"
                onClick={() => shiftMonth(-1)}
                aria-label="Previous month"
              >
                ‹
              </button>
              <button
                type="button"
                className="btn btn-ghost px-2 py-1 text-xs"
                onClick={() => shiftMonth(1)}
                aria-label="Next month"
              >
                ›
              </button>
            </span>
          }
        />
        <CardBody>
          <div className="grid grid-cols-7 gap-1 text-center">
            {WEEKDAYS.map((w) => (
              <div key={w} className="pb-1 text-[11px] font-medium text-mute">
                {w}
              </div>
            ))}
            {cells.map((d, i) => {
              if (d === null) return <div key={`pad-${i}`} />;
              const iso = toISO(month.y, month.m, d);
              const info = byDate.get(iso);
              const isSelected = iso === selected;
              const isToday = iso === index.today;
              return (
                <button
                  key={iso}
                  type="button"
                  disabled={!info}
                  onClick={() => setSelected(iso)}
                  className={cn(
                    "relative aspect-square text-sm transition",
                    info ? "text-ink hover:bg-border" : "cursor-default text-mute/40",
                    isSelected && "bg-accent text-onAccent hover:bg-accent",
                    !isSelected && isToday && "ring-1 ring-inset ring-accent2",
                  )}
                  title={info ? `${info.games} game${info.games === 1 ? "" : "s"}` : undefined}
                >
                  {d}
                  {info && (
                    <span
                      aria-hidden
                      className={cn(
                        "absolute inset-x-0 bottom-1 mx-auto h-1 w-1 rounded-full",
                        isSelected
                          ? "bg-onAccent"
                          : info.upcoming > 0
                            ? "bg-accent"
                            : "bg-mute",
                      )}
                    />
                  )}
                </button>
              );
            })}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={selected ? longDate(selected) : "Pick a date"}
          right={
            games.length ? (
              <span className="text-xs text-mute">
                {games.length} game{games.length === 1 ? "" : "s"}
              </span>
            ) : undefined
          }
        />
        <CardBody className={games.length ? "p-0" : undefined}>
          {games.length === 0 ? (
            <div className="text-sm text-mute">No games scheduled on this date.</div>
          ) : (
            <ul>
              {games.map((g) => (
                <GameRow
                  key={g.game_id}
                  game={g}
                  open={openGame === g.game_id}
                  onToggle={() => setOpenGame(openGame === g.game_id ? null : g.game_id)}
                />
              ))}
            </ul>
          )}
          {day?.note && <p className="px-5 py-3 text-xs text-mute">{day.note}</p>}
        </CardBody>
      </Card>
    </div>
  );
}

function longDate(iso: string) {
  const { y, m, d } = parseDay(iso);
  return `${MONTHS[m]} ${d}, ${y}`;
}

function GameRow({
  game,
  open,
  onToggle,
}: {
  game: any;
  open: boolean;
  onToggle: () => void;
}) {
  const p = game.prediction;
  const homeFavoured = p ? p.home_win_prob >= 0.5 : false;
  const favourite = p ? (homeFavoured ? game.home : game.away) : null;
  const favProb = p ? Math.max(p.home_win_prob, p.away_win_prob) : 0;

  return (
    <li className="border-t border-border/60 first:border-t-0">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-4 px-5 py-3 text-left transition hover:bg-border/40"
      >
        <span className="w-14 shrink-0 text-xs tabular-nums text-mute">{game.tipoff}</span>
        <span className="flex-1 text-sm text-ink">
          {game.away} <span className="text-mute">@</span> {game.home}
        </span>
        {p ? (
          <span className="text-right text-sm">
            <span className="tabular-nums text-accent">{(favProb * 100).toFixed(0)}%</span>{" "}
            <span className="text-mute">{favourite}</span>
          </span>
        ) : (
          <span className="text-xs text-mute">no rating</span>
        )}
        {game.completed && <span className="text-xs text-mute">final</span>}
      </button>

      {open && p && (
        <div className="px-5 pb-4">
          <div className="flex items-end justify-between text-sm">
            <div>
              <div className="font-semibold text-ink">{game.away}</div>
              <div className="text-xs text-mute">away · Elo {p.away_elo}</div>
            </div>
            <div className="text-center text-xs text-mute">
              {favourite} by {Math.abs(p.projected_margin).toFixed(1)}
            </div>
            <div className="text-right">
              <div className="font-semibold text-ink">{game.home}</div>
              <div className="text-xs text-mute">home · Elo {p.home_elo}</div>
            </div>
          </div>
          <div className="mt-2 flex h-2 overflow-hidden bg-border">
            <div className="bg-accent2" style={{ width: `${p.away_win_prob * 100}%` }} />
            <div className="flex-1 bg-accent" />
          </div>
          <div className="mt-1 flex justify-between text-xs tabular-nums text-mute">
            <span>{(p.away_win_prob * 100).toFixed(0)}%</span>
            <span>{(p.home_win_prob * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}
    </li>
  );
}
