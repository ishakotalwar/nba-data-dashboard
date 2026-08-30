import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";

export function Teams({ meta }: { meta: Meta }) {
  const [team, setTeam] = useState(meta.teams[0] ?? "");
  const [series, setSeries] = useState<any>(null);
  const [season, setSeason] = useState<string>("");
  const [factors, setFactors] = useState<any>(null);

  useEffect(() => {
    if (!team) return;
    api.teamSeries(team, meta.league).then((d) => {
      setSeries(d);
      const last = d.rows.at(-1)?.season ?? "";
      setSeason(last);
    });
  }, [team]);

  useEffect(() => {
    if (!team || !season) return;
    api.teamFactors(team, season, meta.league).then(setFactors).catch(() => setFactors(null));
  }, [team, season]);

  const winsTrace = useMemo(() => {
    if (!series?.rows) return [];
    const seasons = series.rows.map((r: any) => r.season);
    return [
      { type: "scatter", mode: "lines+markers", name: "Wins", x: seasons, y: series.rows.map((r: any) => r.wins) },
      { type: "scatter", mode: "lines+markers", name: "Losses", x: seasons, y: series.rows.map((r: any) => r.losses) },
    ];
  }, [series]);

  const shootingTrace = useMemo(() => {
    if (!series?.rows) return [];
    const seasons = series.rows.map((r: any) => r.season);
    const keys = ["FG_PCT", "FG3_PCT", "FT_PCT"];
    return keys
      .filter((k) => series.rows.some((r: any) => r[k] != null))
      .map((k) => ({
        type: "scatter",
        mode: "lines+markers",
        name: k,
        x: seasons,
        y: series.rows.map((r: any) => r[k]),
      }));
  }, [series]);

  const factorsTrace = useMemo(() => {
    if (!factors) return [];
    const labels = ["eFG%", "TOV%", "ORB%", "FT rate"];
    return [
      {
        type: "bar",
        name: team,
        x: labels,
        y: labels.map((k) => factors.team_ff[k]),
        marker: { color: "#ff6a3d" },
        text: labels.map((k) => factors.team_ff[k]?.toFixed(3) ?? ""),
        textposition: "outside",
      },
      {
        type: "bar",
        name: "League avg",
        x: labels,
        y: labels.map((k) => factors.league_avg[k]),
        marker: { color: "#4a5568" },
        text: labels.map((k) => factors.league_avg[k]?.toFixed(3) ?? ""),
        textposition: "outside",
      },
    ];
  }, [factors, team]);

  const seasonOptions = series?.rows?.map((r: any) => r.season) ?? [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Teams" subtitle="Historical wins/losses, shooting, and Dean Oliver's Four Factors" />
        <CardBody>
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className="label mb-1.5">Team</div>
              <Select value={team} onChange={setTeam} options={meta.teams} />
            </div>
            <div>
              <div className="label mb-1.5">Season (for Four Factors)</div>
              <Select value={season} onChange={setSeason} options={seasonOptions} />
            </div>
          </div>
        </CardBody>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Wins & losses" />
          <CardBody>
            {winsTrace.length === 0 ? (
              <div className="py-10 text-center text-sm text-mute">No data.</div>
            ) : (
              <Plot data={winsTrace as any} layout={{ margin: { t: 20 } }} height={320} />
            )}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Shooting %" />
          <CardBody>
            {shootingTrace.length === 0 ? (
              <div className="py-10 text-center text-sm text-mute">No data.</div>
            ) : (
              <Plot data={shootingTrace as any} layout={{ margin: { t: 20 } }} height={320} />
            )}
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader
          title={season ? `Four Factors — ${team}, ${season}` : "Four Factors"}
          subtitle="eFG%, TOV%, ORB%, FT rate explain ~95% of winning. ORB% is OREB/(OREB+DREB) as a proxy."
        />
        <CardBody>
          {factorsTrace.length === 0 ? (
            <div className="py-10 text-center text-sm text-mute">No data.</div>
          ) : (
            <Plot
              data={factorsTrace as any}
              layout={{ barmode: "group", margin: { t: 10 } }}
              height={360}
            />
          )}
        </CardBody>
      </Card>
    </div>
  );
}
