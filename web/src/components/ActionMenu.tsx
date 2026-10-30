import { useEffect, useRef } from "react";
import { CaretDown } from "@phosphor-icons/react";
import type { ReactNode } from "react";

export default function ActionMenu({
  label,
  icon,
  children,
  id,
  className = "",
  active = false,
}: {
  label: string;
  icon?: ReactNode;
  children: ReactNode;
  id?: string;
  className?: string;
  active?: boolean;
}) {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const closeOutside = (event: PointerEvent) => {
      if (
        ref.current?.open &&
        event.target instanceof Node &&
        !ref.current.contains(event.target)
      )
        ref.current.open = false;
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && ref.current?.open) {
        ref.current.open = false;
        ref.current.querySelector("summary")?.focus();
      }
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", escape);
    };
  }, []);
  return (
    <details
      id={id}
      ref={ref}
      className={"action-menu " + className + (active ? " is-active" : "")}
    >
      <summary
        className="button secondary"
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            if (ref.current) {
              ref.current.open = true;
              ref.current
                .querySelector<HTMLElement>(
                  ".action-menu-panel select, .action-menu-panel button:not(:disabled), .action-menu-panel a, .action-menu-panel input",
                )
                ?.focus();
            }
          }
        }}
      >
        {icon}
        <span>{label}</span>
        <CaretDown size={14} className="menu-caret" />
      </summary>
      <div
        className="action-menu-panel"
        onClick={(event) => {
          if (
            event.target instanceof Element &&
            event.target.closest("button:not(:disabled), a")
          ) {
            if (ref.current) ref.current.open = false;
          }
        }}
      >
        {children}
      </div>
    </details>
  );
}
