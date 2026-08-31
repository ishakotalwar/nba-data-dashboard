import { useEffect, useState } from "react";

import { Avatar, playerAvatar } from "@/components/ui/Avatar";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { PlayerCombobox } from "@/components/ui/PlayerCombobox";
import { api, type Meta } from "@/lib/api";
import { formatValue, label as metricLabel } from "@/lib/metrics";
import { formatSeason } from "@/lib/season";

/**
 * Player forecasting, one player at a time: search for someone and see next
 * season's projected line with the arithmetic that produced it.
 */
export function PredictPlayers({ meta }: { meta: Meta }) {
  const [search, setSearch] = useState("");
  const [projection, setProjection] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const avatar = playerAvatar(meta);

  // A player picked in one league has no meaning in the other.
  useEffect(() => {
    setSearch("");
    setProjection(null);
    setErr(null);
  }, [meta.league]);

  const pick = (name: string) => {
    setSearch(name);
    setErr(null);
    setLoading(true);
    api
      .predictPlayer(meta.player_ids[name], meta.league)
      .then(setProjection)
      .catch((e) => {
        setProjection(null);
        setErr(e.message);
      })
      .finally(() => setLoading(false));
  };

  const accuracy = projection?.accuracy;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Projected next season" />
        <CardBody>
          <div className="label mb-1.5">Find a player</div>
          <PlayerCombobox
            options={meta.players}
            value={search}
            onChange={pick}
            placeholder="Search any player"
            renderAvatar={avatar}
          />
          {err && <p className="mt-2 text-xs text-bad">{err}</p>}
          {accuracy?.projection_mae != null && (
            <p className="mt-4 text-xs text-mute">
              Over the last {accuracy.seasons?.length ?? 3} seasons, points projections were off by{" "}
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

      {loading ? (
        <Card>
          <CardBody>
            <div className="text-sm text-mute">Projecting {search}…</div>
          </CardBody>
        </Card>
      ) : projection ? (
        <PlayerDetail projection={projection} meta={meta} />
      ) : (
        !err && (
          <Card>
            <CardBody>
              <div className="text-sm text-mute">
                Search for a player to see their projected line.
              </div>
            </CardBody>
          </Card>
        )
      )}
    </div>
  );
}

/** The arithmetic behind one player's projection, so it can be checked. */
function PlayerDetail({ projection, meta }: { projection: any; meta: Meta }) {
  const fmt = (s: string) => formatSeason(s, projection.season_format ?? meta.season_format);
  const metrics = Object.keys(projection.projected ?? {});
  const leagueTarget: string | undefined = projection.league_target_season;
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
        {leagueTarget && projection.target_season !== leagueTarget && (
          <p className="px-4 pt-3 text-xs text-mute">
            {projection.player_name} last played in {fmt(projection.based_on[0])}, so this is the
            season that would have followed — not {fmt(leagueTarget)}.
          </p>
        )}
        <p className="px-4 py-3 text-xs text-mute">
          Recent seasons weighted {`${12}/${3}/${1}`} and by games played, regressed toward the
          league mean, then adjusted ×{projection.age_multiplier} for age.
        </p>
      </CardBody>
    </Card>
  );
}
