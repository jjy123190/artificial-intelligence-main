import { useRef } from "react";

/** Keyboard and pointer resizing share the same bounded panel size. */
export default function PanelDivider({
  width,
  onChange,
}: {
  width: number;
  onChange: (width: number) => void;
}) {
  const drag = useRef<{ x: number; width: number } | null>(null);
  const update = (next: number) => onChange(Math.max(260, Math.min(420, next)));
  return (
    <div
      className="lab3-panel-divider"
      role="separator"
      aria-label="调整参数面板宽度"
      aria-orientation="vertical"
      aria-controls="lab3-inspector"
      aria-valuemin={260}
      aria-valuemax={420}
      aria-valuenow={width}
      tabIndex={0}
      onPointerDown={(event) => {
        drag.current = { x: event.clientX, width };
        event.currentTarget.setPointerCapture(event.pointerId);
        event.preventDefault();
      }}
      onPointerMove={(event) => {
        if (drag.current)
          update(drag.current.width + drag.current.x - event.clientX);
      }}
      onPointerUp={() => {
        drag.current = null;
      }}
      onLostPointerCapture={() => {
        drag.current = null;
      }}
      onKeyDown={(event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
          return;
        event.preventDefault();
        update(
          event.key === "Home"
            ? 260
            : event.key === "End"
              ? 420
              : width + (event.key === "ArrowLeft" ? 16 : -16),
        );
      }}
    />
  );
}
