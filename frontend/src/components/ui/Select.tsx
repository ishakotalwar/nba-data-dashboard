import { cn } from "@/lib/cn";

type Props = {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label?: string }[] | string[];
  className?: string;
  placeholder?: string;
};

export function Select({ value, onChange, options, className, placeholder }: Props) {
  const opts = (options as any[]).map((o) =>
    typeof o === "string" ? { value: o, label: o } : { value: o.value, label: o.label ?? o.value }
  );
  return (
    <select
      className={cn("input appearance-none pr-8", className)}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {placeholder && <option value="">{placeholder}</option>}
      {opts.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}
