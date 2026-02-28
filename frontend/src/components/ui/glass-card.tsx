import { cn } from "@/lib/utils";

interface GlassCardProps {
  label?: string;
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: () => void;
}

export function GlassCard({
  label,
  children,
  className,
  hover,
  onClick,
}: GlassCardProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "rounded-2xl p-5 relative",
        "bg-[rgba(26,29,39,.65)] backdrop-blur-xl",
        "border border-[rgba(108,124,255,.12)]",
        "shadow-[0_4px_24px_rgba(0,0,0,.2),inset_0_1px_0_rgba(255,255,255,.04)]",
        hover &&
          "cursor-pointer transition-all duration-150 hover:border-primary hover:-translate-y-0.5",
        onClick && "cursor-pointer",
        className
      )}
    >
      {label && (
        <div className="text-[0.72rem] uppercase tracking-[0.08em] text-muted-foreground font-semibold mb-3">
          {label}
        </div>
      )}
      {children}
    </div>
  );
}

export function GradientValue({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "font-extrabold leading-none bg-gradient-to-br from-primary to-[#a78bfa] bg-clip-text text-transparent",
        className
      )}
    >
      {children}
    </span>
  );
}
