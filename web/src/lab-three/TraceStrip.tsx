import type { Result } from "./types";
import { statuses } from "./presentation";

export default function TraceStrip({
  result,
  index,
  onSelect,
}: {
  result: Result;
  index: number;
  onSelect: (index: number) => void;
}) {
  const trace = result.result.trace;
  const start = Math.max(0, Math.min(index - 5, trace.length - 12));
  return (
    <div className="lab3-trace-strip" role="group" aria-label="搜索事件时间线">
      {trace.slice(start, start + 12).map((event, offset) => {
        const position = start + offset;
        const name =
          result.graph.nodes.find((n) => n.id === event.node)?.name ||
          event.node;
        return (
          <button
            type="button"
            key={event.id}
            onClick={() => onSelect(position)}
            aria-pressed={position === index}
            aria-label={`跳转到第 ${position + 1} 步：${statuses[event.status] || event.status}，${name}`}
            title={`${statuses[event.status] || event.status} · ${name}`}
            data-status={event.status}
            data-future={position > index}
          >
            <i aria-hidden="true" />
            <span>{position + 1}</span>
          </button>
        );
      })}
    </div>
  );
}
