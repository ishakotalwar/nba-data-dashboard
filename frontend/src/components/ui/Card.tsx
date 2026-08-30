import { cn } from "@/lib/cn";

export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("card", className)}>{children}</div>;
}

export function CardHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: React.ReactNode }) {
  return (
    <div className="card-header">
      <div>
        <div className="text-base font-semibold tracking-tight text-ink">{title}</div>
        {subtitle && <div className="mt-0.5 text-xs text-mute">{subtitle}</div>}
      </div>
      {right}
    </div>
  );
}

export function CardBody({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("card-body", className)}>{children}</div>;
}
