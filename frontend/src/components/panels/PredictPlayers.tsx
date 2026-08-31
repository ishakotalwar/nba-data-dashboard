import { useEffect, useState } from "react";

import { Avatar } from "@/components/ui/Avatar";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { ErrorNotice } from "@/components/ui/ErrorNotice";
import { Select } from "@/components/ui/Select";
import { api, type Meta } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatValue, label as metricLabel } from "@/lib/metrics";
import { formatSeason } from "@/lib/season";

const ORDERS = [
  { value: "top", label: "Highest projected" },
  { value: "risers", label: "Biggest risers" },
  { value: "fallers", label: "Biggest fallers" },
];

/**
 * Player forecasting: next season's projected line for everyone who played a
 * real share of the last one, with the projection's own error shown next to it.
 */
export function PredictPlayers({ meta }: { meta: Meta }) {
  const [metric, setMetric] = useState("pts");
  const [order, setOrder] = useState("top");
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<any>(null);

  useEffect(() => {
    setData(null);
    setErr(null);
    setSelected(null);
    const sortKey = order === "top" ? metric : `${metric}_delta`;
    const dir = order === "fallers" ? "asc" : "desc";
    api
      .predictPlayers(meta.league, sortKey, 25, dir)
      .then(setData)
      .catch((e) => setErr(e.message));
  }, [meta.league, metric, order]);

  if (err) return <ErrorNotice message={err} />;

  const rows: any[] = data?.rows ?? [];
  const accuracy = data?.accuracy;
  const metricOptions = (data?.metrics ?? ["pts", "reb", "ast", "stl", "blk", "tov", "ts_pct"]).map(
    (m: string) => ({ value: m, label: metricLabel(m) }),
  );

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Projected next season" />
        <CardBody>
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className="label mb-1.5">Metric</div>
              <Select value={metric} onChange={setMetric} options={metricOptions} />
            </div>
            <div>
              <div className="label mb-1.5">Show</div>
              <Select value={order} onChange={setOrder} options={ORDERS} />
            </div>
          </div>
          {accuracy?.projection_mae != null && (
            <p className="mt-4 text-xs text-mute">
              On the last {accuracy.seasons?.length ?? 3} seasons this projection was off by{" "}
              <span className="tabular-nums text-ink">{accuracy.projection_mae}</span> on average,
              against <span className="tabular-nums">{accuracy.baseline_mae}</span> for simply
              repeating last season — {accuracy.improvement >= 0 ? "an improvement of " : "worse by "}
              <span className="tabular-nums">
                {Math.abs(accuracy.improvement * 100).toFixed(1)}%
              </span>
              , over {accuracy.players?.toLocaleString()} player-seasons.
            </p>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={ORDERS.find((o) => o.value === order)?.label ?? "Projected"}
          right={
            data ? (
              <span className="text-xs text-mute">
                for {formatSeason(rows[0]?.target_season, data.season_format)}
              </span>
            ) : undefined
          }
        />
        <CardBody className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-mute">
                <tr className="border-b border-border">
                  <th className="px-4 py-2 text-left font-medium">Player</th>
                  <th className="px-4 py-2 text-right font-medium">Age</th>
                  <th className="px-4 py-2 text-right font-medium">Last</th>
                  <th className="px-4 py-2 text-right font-medium">Projected</th>
                  <th className="px-4 py-2 text-right font-medium">Change</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r.player_id}
                    className="cursor-pointer border-t border-border/60 hover:bg-border/40"
                    onClick={() =>
                      api
                        .predictPlayer(r.player_id, meta.league)
                        .then(setSelected)
                        .catch(() => setSelected(null))
                    }
                  >
                    <td className="px-4 py-2">
                      <span className="flex items-center gap-2">
                        <Avatar
                          name={r.player_name}
                          id={r.player_id}
                          league={meta.league}
                          size={24}
                        />
                        <span className="text-ink">{r.player_name}</span>
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-mute">
                      {r.age_at_target ?? "—"}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-mute">
                      {formatValue(metric, r[`${metric}_last`])}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-ink">
                      {formatValue(metric, r[metric])}
                    </td>
                    <td
                      className={cn(
                        "px-4 py-2 text-right tabular-nums",
                        r[`${metric}_delta`] > 0
                          ? "text-good"
                          : r[`${metric}_delta`] < 0
                            ? "text-bad"
                            : "text-mute",
                      )}
                    >
                      {r[`${metric}_delta`] == null
                        ? "—"
                        : `${r[`${metric}_delta`] > 0 ? "+" : ""}${formatValue(
                            metric,
                            r[`${metric}_delta`],
                          )}`}
                    </td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-4 py-6 text-mute">
                      No projections yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>

      {selected && <PlayerDetail projection={selected} meta={meta} />}
    </div>
  );
}

/** The arithmetic behind one player's projection, so it can be checked. */
function PlayerDetail({ projection, meta }: { projection: any; meta: Meta }) {
  const fmt = (s: string) => formatSeason(s, projection.season_format ?? meta.season_format);
  const metrics = Object.keys(projection.projected ?? {});
  return (
    <Card>
      <CardHeader
        lead={
          <Avatar
            name={projection.player_name}
            id={projection.player_id}
            league={meta.league}
            size={40}
          />
        }
        title={`${projection.player_name} · ${fmt(projection.target_season)}`}
        right={
          <span className="text-xs text-mute">
            from {projection.based_on.map(fmt).join(", ")}
            {projection.age_at_target ? ` · age ${projection.age_at_target}` : ""}
          </span>
        }
      />
      <CardBody className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-mute">
              <tr className="border-b border-border">
                <th className="px-4 py-2 text-left font-medium">Metric</th>
                {projection.based_on.map((s: string) => (
                  <th key={s} className="px-4 py-2 text-right font-medium">
                    {fmt(s)}
                  </th>
                ))}
                <th className="px-4 py-2 text-right font-medium">Projected</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => {
                const inputs: any[] = projection.inputs?.[m] ?? [];
                return (
                  <tr key={m} className="border-t border-border/60">
                    <td className="px-4 py-2 text-ink">{metricLabel(m)}</td>
                    {projection.based_on.map((s: string) => {
                      const hit = inputs.find((i) => i.season === s);
                      return (
                        <td key={s} className="px-4 py-2 text-right tabular-nums text-mute">
                          {hit ? formatValue(m, hit.value) : "—"}
                        </td>
                      );
                    })}
                    <td className="px-4 py-2 text-right tabular-nums text-ink">
                      {formatValue(m, projection.projected[m])}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="px-4 py-3 text-xs text-mute">
          Recent seasons weighted {`${12}/${3}/${1}`} and by games played, regressed toward the
          league mean, then adjusted ×{projection.age_multiplier} for age.
        </p>
      </CardBody>
    </Card>
  );
}
