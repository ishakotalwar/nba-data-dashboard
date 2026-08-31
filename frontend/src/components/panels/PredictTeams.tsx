import { useEffect, useState } from "react";

import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { ErrorNotice } from "@/components/ui/ErrorNotice";
import { Plot } from "@/components/ui/Plot";
import { Select } from "@/components/ui/Select";
import { api, type Meta } from "@/lib/api";
import { cn } from "@/lib/cn";

/**
 * Team forecasting: a power ranking, a matchup calculator, and — deliberately
 * given equal billing — how the model actually scores against the naive
 * "always pick the home team" baseline.
 */
export function PredictTeams({ meta }: { meta: Meta }) {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [matchup, setMatchup] = useState<any>(null);

  useEffect(() => {
    setData(null);
    setErr(null);
    setMatchup(null);
    setHome("");
    setAway("");
    api.predictTeams(meta.league).then(setData).catch((e) => setErr(e.message));
  }, [meta.league]);

  useEffect(() => {
    if (!home || !away || home === away) {
      setMatchup(null);
      return;
    }
    api
      .predictMatchup(home, away, meta.league)
      .then(setMatchup)
      .catch(() => setMatchup(null));
  }, [home, away, meta.league]);

  if (err) return <ErrorNotice message={err} />;

  const ratings: any[] = data?.ratings ?? [];
  const teamOptions = ratings.map((r) => ({ value: r.team, label: r.team }));
  const overall = data?.backtest?.overall;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Matchup" />
        <CardBody>
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className="label mb-1.5">Home</div>
              <Select value={home} onChange={setHome} options={teamOptions} placeholder="Select" />
            </div>
            <div>
              <div className="label mb-1.5">Away</div>
              <Select value={away} onChange={setAway} options={teamOptions} placeholder="Select" />
            </div>
          </div>

          {matchup ? (
            <div className="mt-5">
              <div className="flex items-end justify-between text-sm">
                <div>
                  <div className="text-xl font-semibold text-ink">{matchup.home}</div>
                  <div className="text-xs text-mute">home · Elo {matchup.home_elo}</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-semibold tabular-nums text-accent">
                    {(matchup.home_win_prob * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-mute">
                    {matchup.home} by {Math.abs(matchup.projected_margin).toFixed(1)}
                    {matchup.projected_margin < 0 ? " (away favoured)" : ""}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xl font-semibold text-ink">{matchup.away}</div>
                  <div className="text-xs text-mute">away · Elo {matchup.away_elo}</div>
                </div>
              </div>
              <div className="mt-3 flex h-2 overflow-hidden bg-border">
                <div
                  className="bg-accent"
                  style={{ width: `${matchup.home_win_prob * 100}%` }}
                  aria-label={`${matchup.home} win probability`}
                />
                <div className="flex-1 bg-accent2" aria-label={`${matchup.away} win probability`} />
              </div>
            </div>
          ) : (
            <div className="mt-5 text-sm text-mute">
              {home && away && home === away
                ? "Pick two different teams."
                : "Pick a home and away team."}
            </div>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="Power ratings"
          right={
            data ? (
              <span className="text-xs text-mute">
                {data.games_used.toLocaleString()} games through {data.through}
              </span>
            ) : undefined
          }
        />
        <CardBody className="p-0">
          <div className="max-h-[520px] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-panel text-mute">
                <tr className="border-b border-border">
                  <th className="px-4 py-2 text-left font-medium">#</th>
                  <th className="px-4 py-2 text-left font-medium">Team</th>
                  <th className="px-4 py-2 text-right font-medium">Elo</th>
                  <th className="px-4 py-2 text-right font-medium">Latest season</th>
                </tr>
              </thead>
              <tbody>
                {ratings.map((r) => (
                  <tr key={r.team} className="border-t border-border/60">
                    <td className="px-4 py-2 tabular-nums text-mute">{r.rank}</td>
                    <td className="px-4 py-2 text-ink">{r.team}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{r.elo.toFixed(0)}</td>
                    <td className="px-4 py-2 text-right tabular-nums text-mute">
                      {r.games ? `${r.wins}–${r.losses}` : "—"}
                    </td>
                  </tr>
                ))}
                {ratings.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-4 py-6 text-mute">
                      No ratings yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="How good is this?" />
        <CardBody>
          {overall ? (
            <>
              <div className="grid gap-4 sm:grid-cols-3">
                <Stat
                  label="Accuracy"
                  value={`${(overall.accuracy * 100).toFixed(1)}%`}
                  note={`vs ${(overall.baseline_accuracy * 100).toFixed(1)}% always-home`}
                  good={overall.accuracy > overall.baseline_accuracy}
                />
                <Stat
                  label="Brier score"
                  value={overall.brier.toFixed(4)}
                  note={`vs ${overall.baseline_brier.toFixed(4)} · lower is better`}
                  good={overall.brier < overall.baseline_brier}
                />
                <Stat label="Log loss" value={overall.log_loss.toFixed(4)} note="lower is better" />
              </div>
              <p className="mt-4 text-xs text-mute">
                Scored on {overall.games.toLocaleString()} games from{" "}
                {data.backtest.test_from} onward, each predicted before it was played and
                before it updated the ratings.
              </p>
            </>
          ) : (
            <div className="text-sm text-mute">No backtest available.</div>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Calibration" />
        <CardBody>
          <Plot
            height={320}
            data={
              data?.calibration?.length
                ? [
                    {
                      type: "scatter",
                      mode: "lines",
                      name: "perfect",
                      x: [0, 1],
                      y: [0, 1],
                      line: { dash: "dot", width: 1 },
                      hoverinfo: "skip",
                    },
                    {
                      type: "scatter",
                      mode: "markers+lines",
                      name: "model",
                      x: data.calibration.map((c: any) => c.predicted),
                      y: data.calibration.map((c: any) => c.actual),
                      text: data.calibration.map((c: any) => `${c.games} games`),
                      marker: { size: 9 },
                    },
                  ]
                : []
            }
            layout={{
              xaxis: { title: "predicted win probability", range: [0, 1], tickformat: ".0%" },
              yaxis: { title: "actual win rate", range: [0, 1], tickformat: ".0%" },
              showlegend: false,
            }}
            placeholder="No calibration data"
          />
          <p className="mt-2 text-xs text-mute">
            Points on the dotted line mean the stated probability matched reality — a 70%
            prediction won about 70% of the time.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}

function Stat({
  label,
  value,
  note,
  good,
}: {
  label: string;
  value: string;
  note?: string;
  good?: boolean;
}) {
  return (
    <div>
      <div className="label">{label}</div>
      <div
        className={cn(
          "mt-1 text-2xl font-semibold tabular-nums",
          good === undefined ? "text-ink" : good ? "text-good" : "text-bad",
        )}
      >
        {value}
      </div>
      {note && <div className="mt-0.5 text-xs text-mute">{note}</div>}
    </div>
  );
}
