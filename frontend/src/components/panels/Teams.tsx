import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";

export function Teams({ meta }: { meta: Meta }) {
  const [team, setTeam] = useState("");
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

  const ratingTrace = useMemo(() => {
    if (!series?.rows) return [];
    const seasons = series.rows.map((r: any) => r.season);
    return [
      { type: "scatter", mode: "lines+markers", name: "Offensive rating",
        x: seasons, y: series.rows.map((r: any) => r.ortg) },
      { type: "scatter", mode: "lines+markers", name: "Defensive rating",
        x: seasons, y: series.rows.map((r: any) => r.drtg) },
    ];
  }, [series]);

  // Net is on a different scale (around 0, not around 110), so it gets its own
  // chart rather than a second y-axis.
  const netTrace = useMemo(() => {
    if (!series?.rows) return [];
    return [
      {
        type: "bar",
        name: "Net rating",
        x: series.rows.map((r: any) => r.season),
        y: series.rows.map((r: any) =>
          r.ortg != null && r.drtg != null ? +(r.ortg - r.drtg).toFixed(1) : null
        ),
        marker: {
          color: series.rows.map((r: any) =>
            r.ortg != null && r.drtg != null && r.ortg - r.drtg >= 0 ? "#4dabff" : "#d73027"
          ),
        },
      },
    ];
  }, [series]);

  const paceTrace = useMemo(() => {
    if (!series?.rows) return [];
    return [
      { type: "scatter", mode: "lines+markers", name: "Pace",
        x: series.rows.map((r: any) => r.season),
        y: series.rows.map((r: any) => r.pace) },
    ];
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
        <CardHeader title="Teams" subtitle="Rate stats over time, plus Dean Oliver's Four Factors against the league" />
        <CardBody>
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className="label mb-1.5">Team</div>
              <Select value={team} onChange={setTeam} options={meta.teams} placeholder="Select" />
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
          <CardHeader title="Offensive & defensive rating" subtitle="Points scored / allowed per 100 possessions" />
          <CardBody>
            {ratingTrace.length === 0 ? (
              <div className="py-10 text-center text-sm text-mute">No data.</div>
            ) : (
              <Plot
                data={ratingTrace as any}
                layout={{ margin: { t: 20 }, xaxis: { type: "category", nticks: 10, tickangle: 0 } }}
                height={320}
              />
            )}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Net rating" subtitle="ORtg − DRtg; above zero outscores its opponents" />
          <CardBody>
            {netTrace.length === 0 ? (
              <div className="py-10 text-center text-sm text-mute">No data.</div>
            ) : (
              <Plot
                data={netTrace as any}
                layout={{
                  margin: { t: 20 },
                  showlegend: false,
                  xaxis: { type: "category", nticks: 10, tickangle: 0 },
                  yaxis: { zerolinecolor: "#3a4250", zerolinewidth: 1 },
                }}
                height={320}
              />
            )}
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title="Pace" subtitle="Possessions per game" />
        <CardBody>
          {paceTrace.length === 0 ? (
            <div className="py-10 text-center text-sm text-mute">No data.</div>
          ) : (
            <Plot
              data={paceTrace as any}
              layout={{ margin: { t: 20 }, showlegend: false, xaxis: { type: "category", nticks: 12, tickangle: 0 } }}
              height={280}
            />
          )}
        </CardBody>
      </Card>

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
              layout={{ barmode: "group", margin: { t: 10 }, xaxis: { type: "category" } }}
              height={360}
            />
          )}
        </CardBody>
      </Card>
    </div>
  );
}
