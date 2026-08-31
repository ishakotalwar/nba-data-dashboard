import { useEffect, useMemo, useState } from "react";
import { api, type Meta } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Plot } from "@/components/ui/Plot";
import { formatSeason } from "@/lib/season";

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
    const seasons = series.rows.map((r: any) => formatSeason(r.season, meta.season_format));
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
        x: series.rows.map((r: any) => formatSeason(r.season, meta.season_format)),
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
        x: series.rows.map((r: any) => formatSeason(r.season, meta.season_format)),
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

  const seasonOptions =
    series?.rows?.map((r: any) => ({ value: r.season, label: formatSeason(r.season, meta.season_format) })) ?? [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Teams" />
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
          <CardHeader title="Offensive & defensive rating" />
          <CardBody>
            <Plot
              data={ratingTrace as any}
              layout={{
                margin: { t: 20 },
                xaxis: { type: "category", nticks: 10, tickangle: 0, title: "Season" },
                yaxis: { title: "Points per 100 poss." },
              }}
              height={320}
              placeholder="Select a team"
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Net rating" />
          <CardBody>
            <Plot
              data={netTrace as any}
              layout={{
                margin: { t: 20 },
                showlegend: false,
                xaxis: { type: "category", nticks: 10, tickangle: 0, title: "Season" },
                yaxis: { title: "ORtg − DRtg", zerolinecolor: "#3a4250", zerolinewidth: 1 },
              }}
              height={320}
              placeholder="Select a team"
            />
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title="Pace" />
        <CardBody>
          <Plot
            data={paceTrace as any}
            layout={{
              margin: { t: 20 },
              showlegend: false,
              xaxis: { type: "category", nticks: 12, tickangle: 0, title: "Season" },
              yaxis: { title: "Possessions per game" },
            }}
            height={280}
            placeholder="Select a team"
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={season ? `Four Factors — ${team}, ${formatSeason(season, meta.season_format)}` : "Four Factors"}
        />
        <CardBody>
          <Plot
            data={factorsTrace as any}
            layout={{
              barmode: "group",
              margin: { t: 10 },
              xaxis: { type: "category", categoryorder: "array",
                       categoryarray: ["eFG%", "TOV%", "ORB%", "FT rate"] },
              yaxis: { title: "Rate" },
            }}
            height={360}
            placeholder="Select a team and season"
          />
        </CardBody>
      </Card>
    </div>
  );
}
