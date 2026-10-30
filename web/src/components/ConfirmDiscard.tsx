import { useEffect, useId, useRef } from "react";
import { Warning, DownloadSimple } from "@phosphor-icons/react";

export type Confirmation = {
  title: string;
  description: string;
  action: string;
};

/** Native top-layer dialog: Escape/cancel never discards the draft. */
export default function ConfirmDiscard({
  confirmation,
  onResolve,
  onExport,
}: {
  confirmation: Confirmation | null;
  onResolve: (accepted: boolean) => void;
  onExport: () => void;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const element = dialog.current;
    if (confirmation && element && !element.open) {
      element.showModal();
      cancel.current?.focus();
    } else if (!confirmation && element?.open) {
      element.close();
    }
  }, [confirmation]);
  return (
    <dialog
      ref={dialog}
      className="discard-dialog"
      aria-labelledby={titleId}
      aria-describedby={descriptionId}
      onKeyDown={(event) => {
        if (event.key !== "Tab") return;
        const controls = Array.from(
          dialog.current?.querySelectorAll<HTMLElement>(
            "button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), [tabindex='0']",
          ) || [],
        );
        if (!controls.length) return;
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }}
      onCancel={(event) => {
        event.preventDefault();
        onResolve(false);
      }}
    >
      {confirmation && (
        <>
          <div className="discard-symbol">
            <Warning size={25} />
          </div>
          <h2 id={titleId}>{confirmation.title}</h2>
          <p id={descriptionId}>{confirmation.description}</p>
          <button className="draft-export-link" onClick={onExport}>
            <DownloadSimple size={17} />
            先导出当前草稿
          </button>
          <div className="discard-actions">
            <button
              ref={cancel}
              className="button secondary"
              onClick={() => onResolve(false)}
            >
              保留草稿
            </button>
            <button className="button primary" onClick={() => onResolve(true)}>
              {confirmation.action}
            </button>
          </div>
        </>
      )}
    </dialog>
  );
}
