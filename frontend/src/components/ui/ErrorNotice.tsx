type Props = {
  message: string;
  onRetry?: () => void;
};

export function ErrorNotice({ message, onRetry }: Props) {
  return (
    <div className="flex items-start justify-between gap-4 rounded-lg border border-bad/30 bg-bad/10 px-4 py-3 text-sm text-ink">
      <div>
        <div className="font-medium text-bad">Upstream NBA API issue</div>
        <div className="mt-0.5 text-ink/80">{message}</div>
      </div>
      {onRetry && (
        <button className="btn btn-ghost whitespace-nowrap" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
