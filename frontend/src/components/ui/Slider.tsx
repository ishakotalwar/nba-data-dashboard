import * as RSlider from "@radix-ui/react-slider";
import { cn } from "@/lib/cn";

type Props = {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
};

export function Slider({ value, onChange, min = 0, max = 100, step = 1, className }: Props) {
  return (
    <RSlider.Root
      className={cn("relative flex h-5 w-full touch-none select-none items-center", className)}
      min={min}
      max={max}
      step={step}
      value={[value]}
      onValueChange={(v) => onChange(v[0])}
    >
      <RSlider.Track className="relative h-1 grow rounded-full bg-border">
        <RSlider.Range className="absolute h-full rounded-full bg-accent" />
      </RSlider.Track>
      <RSlider.Thumb className="block h-4 w-4 rounded-full bg-accent shadow focus:outline-none focus:ring-2 focus:ring-accent/40" />
    </RSlider.Root>
  );
}
