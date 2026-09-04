import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";
import { Avatar } from "@/components/ui/Avatar";
import { cn } from "@/lib/cn";
import { formatSeason } from "@/lib/season";

const signed = (v: number | null, digits = 1) =>
  v == null ? "" : (v > 0 ? "+" : "") + v.toFixed(digits);

const COLS = [
  { key: "min", label: "MP", title: "Minutes", fmt: (v: number) => v?.toFixed(0) ?? "" },
  { key: "poss", label: "Poss", title: "Possessions", fmt: (v: number) => v?.toFixed(0) ?? "" },
  { key: "ortg", label: "ORtg", title: "Points scored per 100 possessions",
    fmt: (v: number) => v?.toFixed(1) ?? "" },
  { key: "drtg", label: "DRtg", title: "Points allowed per 100 possessions",
    fmt: (v: number) => v?.toFixed(1) ?? "" },
  { key: "net", label: "Net", title: "ORtg − DRtg", fmt: (v: number) => signed(v), strong: true },
];

/**
 * WOWY — with or without you. What a team did with a group of players on the
 * floor, and what it did without them. Every possession belongs to exactly one five, so any group is
 * answered by adding up the fives containing it, and the combinations come out
 * exhaustive — they add back to the team's whole season.
 *
 * Deliberately unadjusted: this is the raw split, teammates and opponents
 * included, which answers "what happened when he sat" rather than "how good is
 * he". The Impact page carries the adjusted version, and the two disagreeing is
 * usually the interesting part — a team that collapses without a player may
 * only be telling you about his backup.
 */
export function Wowy({ meta }: { meta: Meta }) {
  const seasons = meta.lineup_seasons ?? [];
  const [season, setSeason] = useState(seasons.at(-1) ?? "");
  const [team, setTeam] = useState("");
  const [picked, setPicked] = useState<number[]>([]);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  // A new team means a new roster, so nobody stays selected across the change.
  useEffect(() => {
    setPicked([]);
  }, [team, season, meta.league]);

  useEffect(() => {
    if (!season || !team) {
      setData(null);
      return;
    }
    setErr(null);
    api
      .teamWowy(season, team, meta.league, picked)
      .then(setData)
      .catch((e) => {
        setErr(e.message);
        setData(null);
      });
  }, [season, team, picked.join(","), meta.league]);

  const roster = data?.roster ?? [];
  // Who is selected, in the order the roster lists them, for the marker columns.
  const chosen = useMemo(
    () => picked.map((id) => roster.find((p: any) => p.player_id === id)).filter(Boolean),
    [picked.join(","), roster]
  );
  const surname = (name: string) => {
    const [, ...rest] = name.trim().split(/\s+/);
    return rest.join(" ") || name;
  };
  const rows = data?.rows ?? [];
  const total = data?.team_total;
  const max = data?.max_players ?? 4;
  const full = picked.length >= max;

  const toggle = (id: number) =>
    setPicked((current) =>
      current.includes(id)
        ? current.filter((p) => p !== id)
        : current.length >= max
        ? current
        : [...current, id]
    );

  const traces = useMemo(() => {
    if (!rows.length) return [];
    return [
      {
        type: "bar",
        orientation: "h",
        x: rows.map((r: any) => r.net),
        y: rows.map((r: any) => r.label),
        // No in-bar labels: a negative bar's text runs off the left edge, and
        // the table beside this one already carries every number.
        hovertemplate:
          "<b>%{y}</b><br>net %{x:+.1f} per 100<br>%{customdata:.0f} minutes<extra></extra>",
        customdata: rows.map((r: any) => r.min),
        marker: { color: rows.map((r: any) => (r.net >= 0 ? "#4dabff" : "#d73027")) },
      },
    ];
  }, [rows]);

  const layout = useMemo(
    () => ({
      margin: { t: 10, r: 16, b: 42, l: 10 },
      showlegend: false,
      // Longest split first, reading top-down like the table beside it.
      yaxis: { autorange: "reversed", automargin: true, ticksuffix: "  " },
      xaxis: {
        title: "Net rating per 100 possessions",
        gridcolor: "#1f2630",
        zeroline: true,
        zerolinecolor: "#3a4250",
        zerolinewidth: 2,
      },
    }),
    []
  );

  if (!seasons.length) {
    return (
      <Card>
        <CardHeader title="With or Without You" />
        <CardBody>
          <div className="text-sm text-mute">
            No {meta.league_label} lineup data on disk. Build it with{" "}
            <code className="bg-border/60 px-1.5 py-0.5">
              python etl/lineup_etl.py --league {meta.league}
            </code>
            .
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="With or Without You"
        />
        <CardBody>
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className="label mb-1.5">Season</div>
              <Select
                value={season}
                onChange={setSeason}
                options={seasons.map((s) => ({
                  value: s,
                  label: formatSeason(s, meta.season_format),
                }))}
              />
            </div>
            <div>
              <div className="label mb-1.5">Team</div>
              <Select value={team} onChange={setTeam} options={meta.teams} placeholder="Select" />
            </div>
          </div>
          {err && <div className="mt-3 text-sm text-bad">{err}</div>}
        </CardBody>
      </Card>

      {roster.length > 0 && (
        <Card>
          <CardHeader
            title="Squad"
            right={
              picked.length ? (
                <button className="btn btn-ghost text-sm" onClick={() => setPicked([])}>
                  Clear
                </button>
              ) : undefined
            }
          />
          <CardBody>
            <div className="grid grid-cols-3 gap-x-3 gap-y-5 sm:grid-cols-5 md:grid-cols-6 lg:grid-cols-8">
              {roster.map((p: any) => {
                const on = picked.includes(p.player_id);
                // A full selection greys out the rest rather than silently
                // ignoring the click.
                const blocked = !on && full;
                return (
                  <button
                    key={p.player_id}
                    type="button"
                    onClick={() => toggle(p.player_id)}
                    disabled={blocked}
                    title={blocked ? `Deselect someone to pick more than ${max}` : p.name}
                    className={cn(
                      "group flex flex-col items-center gap-1.5 text-center transition",
                      blocked && "cursor-not-allowed opacity-40"
                    )}
                  >
                    {/* The ring lives on a wrapper rather than the avatar: the
                        avatar carries its own, and these class names merge by
                        stylesheet order rather than by intent. */}
                    <span
                      className={cn(
                        "relative inline-flex rounded-full p-[3px] ring-2 transition",
                        on
                          ? "ring-accent"
                          : "ring-transparent group-hover:ring-border"
                      )}
                    >
                      <Avatar name={p.name} id={p.player_id} league={meta.league} size={76} />
                      <span
                        className={cn(
                          "absolute bottom-0 right-0 grid h-[22px] w-[22px]",
                          "place-items-center rounded-full border-2 border-bg text-[13px]",
                          "font-semibold leading-none transition",
                          on ? "bg-accent text-onAccent" : "bg-border text-mute"
                        )}
                      >
                        +
                      </span>
                    </span>
                    <div
                      className={cn(
                        "w-full truncate text-[13px] leading-tight",
                        on ? "text-ink" : "text-mute"
                      )}
                    >
                      {p.name}
                    </div>
                  </button>
                );
              })}
            </div>
          </CardBody>
        </Card>
      )}

      {rows.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardBody className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wider text-mute">
                      {chosen.length > 1 ? (
                        chosen.map((p: any) => (
                          <th
                            key={p.player_id}
                            title={p.name}
                            className="whitespace-nowrap px-2 py-2 text-center font-medium"
                          >
                            {surname(p.name)}
                          </th>
                        ))
                      ) : (
                        <th className="px-4 py-2 font-medium">Floor time</th>
                      )}
                      {COLS.map((c) => (
                        <th key={c.key} title={c.title} className="px-3 py-2 text-right font-medium">
                          {c.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r: any) => (
                      <tr key={r.label} className="border-t border-border/60">
                        {chosen.length > 1 ? (
                          chosen.map((p: any) => {
                            const on = r.on.includes(p.name);
                            return (
                              <td
                                key={p.player_id}
                                title={`${p.name} ${on ? "on" : "off"} the floor`}
                                className={cn(
                                  "px-2 py-2 text-center",
                                  on ? "text-accent" : "text-mute/40"
                                )}
                              >
                                {on ? "●" : "○"}
                              </td>
                            );
                          })
                        ) : (
                          <td className="px-4 py-2">{r.label}</td>
                        )}
                        {COLS.map((c) => (
                          <td
                            key={c.key}
                            className={cn(
                              "px-3 py-2 text-right tabular-nums",
                              c.strong && "font-medium",
                              c.strong && r.net != null && (r.net >= 0 ? "text-good" : "text-bad")
                            )}
                          >
                            {c.fmt(r[c.key])}
                          </td>
                        ))}
                      </tr>
                    ))}
                    {total && (
                      <tr className="border-t border-border bg-border/20 text-mute">
                        <td
                          colSpan={Math.max(1, chosen.length > 1 ? chosen.length : 1)}
                          className="px-4 py-2 font-medium"
                        >
                          {total.label} overall
                        </td>
                        {COLS.map((c) => (
                          <td key={c.key} className="px-3 py-2 text-right tabular-nums">
                            {c.fmt(total[c.key])}
                          </td>
                        ))}
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Net rating by split" />
            <CardBody>
              <Plot
                data={traces as any}
                layout={layout as any}
                height={Math.max(240, 60 + rows.length * 42)}
              />
            </CardBody>
          </Card>
        </div>
      )}
    </div>
  );
}
