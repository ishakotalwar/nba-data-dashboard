import { cn } from "@/lib/cn";

export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("card", className)}>{children}</div>;
}

export function CardHeader({
  title,
  subtitle,
  right,
  lead,
}: {
  title: React.ReactNode;
  subtitle?: string;
  right?: React.ReactNode;
  /** Optional element before the title, e.g. a player headshot. */
  lead?: React.ReactNode;
}) {
  return (
    <div className="card-header">
      <div className="flex min-w-0 items-center gap-3">
        {lead}
        <div className="min-w-0">
          <div className="truncate text-base font-semibold tracking-tight text-ink">{title}</div>
          {subtitle && <div className="mt-0.5 text-xs text-mute">{subtitle}</div>}
        </div>
      </div>
      {right}
    </div>
  );
}

export function CardBody({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("card-body", className)}>{children}</div>;
}
