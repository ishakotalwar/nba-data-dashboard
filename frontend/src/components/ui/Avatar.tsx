import { useEffect, useState } from "react";
import { headshotUrl, initials } from "@/lib/headshot";
import type { LeagueKey } from "@/lib/api";
import { cn } from "@/lib/cn";

/**
 * Bind a Meta to an avatar renderer, for the `renderAvatar` prop the pickers
 * take. Kept out of the pickers themselves so metric lists stay text-only.
 */
export function playerAvatar(meta: { player_ids: Record<string, number>; league: LeagueKey }) {
  return (name: string, size = 28) => (
    <Avatar name={name} id={meta.player_ids[name]} league={meta.league} size={size} />
  );
}

type Props = {
  name: string;
  id?: number;
  league: LeagueKey;
  size?: number;
  className?: string;
};

/**
 * Player headshot, degrading to initials when there is no id, the image 404s,
 * or it hasn't loaded yet.
 *
 * ESPN headshots are RGBA with a transparent background, so the initials have
 * to be removed once the image lands — left underneath they show straight
 * through the cut-out and read as a smudge over the face.
 */
export function Avatar({ name, id, league, size = 28, className }: Props) {
  const src = headshotUrl(id, league, size);
  const [status, setStatus] = useState<"pending" | "ok" | "error">("pending");

  // Virtualized rows are recycled, so a new src must start over.
  useEffect(() => setStatus("pending"), [src]);

  const showInitials = !src || status !== "ok";

  return (
    <span
      style={{ width: size, height: size }}
      className={cn(
        "relative inline-grid shrink-0 place-items-center overflow-hidden rounded-full",
        // Opaque ground: the headshot is a transparent cut-out and needs
        // something behind it, or it composites onto whatever is below.
        "bg-[#2a3240] ring-1 ring-border",
        className
      )}
    >
      {showInitials && (
        <span
          style={{ fontSize: Math.max(9, Math.round(size * 0.38)) }}
          className="font-semibold leading-none text-mute"
        >
          {initials(name)}
        </span>
      )}
      {src && status !== "error" && (
        <img
          src={src}
          alt=""
          aria-hidden
          loading="lazy"
          decoding="async"
          onLoad={() => setStatus("ok")}
          onError={() => setStatus("error")}
          className={cn(
            "absolute inset-0 h-full w-full object-cover object-top transition-opacity",
            status === "ok" ? "opacity-100" : "opacity-0"
          )}
        />
      )}
    </span>
  );
}
