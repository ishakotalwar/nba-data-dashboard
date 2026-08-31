import { useEffect, useState } from "react";
import { api, type Meta, type PlayerInfo, type PlayerSeason } from "@/lib/api";
import { PlayerCombobox } from "./PlayerCombobox";
import { Select } from "./Select";
import { playerAvatar } from "./Avatar";
import { formatSeason } from "@/lib/season";

type Props = {
  meta: Meta;
  value: PlayerSeason;
  onChange: (v: PlayerSeason) => void;
  /** Hide the season control for career-wide contexts. */
  seasonless?: boolean;
  playerLabel?: string;
  className?: string;
  /** Bio and season list for the current player, lifted for the caller's use. */
  onInfo?: (info: PlayerInfo | null) => void;
};

/**
 * The app's one player+season control. Season options come from the seasons the
 * selected player actually played, so an impossible pairing can't be chosen.
 */
export function PlayerSeasonSelector({
  meta,
  value,
  onChange,
  seasonless,
  playerLabel = "Player",
  className,
  onInfo,
}: Props) {
  const avatar = playerAvatar(meta);
  const [info, setInfo] = useState<PlayerInfo | null>(null);

  useEffect(() => {
    const id = value.playerId;
    if (!id) {
      setInfo(null);
      onInfo?.(null);
      return;
    }
    let alive = true;
    api
      .player(id, meta.league)
      .then((d) => {
        if (!alive) return;
        setInfo(d);
        onInfo?.(d);
        // Keep the season if the new player has it; otherwise take their latest.
        if (!seasonless && !d.seasons.includes(value.season)) {
          onChange({ ...value, season: d.seasons.at(-1) ?? "" });
        }
      })
      .catch(() => {
        if (!alive) return;
        setInfo(null);
        onInfo?.(null);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value.playerId, meta.league]);

  const pickPlayer = (name: string) =>
    onChange({ playerId: meta.player_ids[name], playerName: name, season: value.season });

  return (
    <div className={className ?? "grid gap-3 md:grid-cols-2"}>
      <div>
        <div className="label mb-1.5">{playerLabel}</div>
        <PlayerCombobox
          options={meta.players}
          value={value.playerName}
          onChange={pickPlayer}
          renderAvatar={avatar}
        />
      </div>
      {!seasonless && (
        <div>
          <div className="label mb-1.5">{playerLabel === "" ? "\u00a0" : "Season"}</div>
          <Select
            value={value.season}
            onChange={(s) => onChange({ ...value, season: s })}
            options={(info?.seasons ?? []).map((s) => ({
              value: s,
              label: formatSeason(s, meta.season_format),
            }))}
            placeholder={value.playerName ? "Select" : "Pick a player first"}
          />
        </div>
      )}
    </div>
  );
}

/** Empty selection, for panel initial state. */
export const emptySelection: PlayerSeason = { playerName: "", season: "" };
