import { useEffect, useId, useRef, useState } from "react";
import type { Bootstrap, CaseBundle } from "../types";
import { fmt } from "../api";
import HoldingEditor from "./HoldingEditor";
import CompletenessFields from "./CompletenessFields";

export type TaskInputMode = "case" | "entity" | "completeness";
export type TaskInputUpdate = Partial<
  Pick<CaseBundle, "case" | "completeness">
>;
const titles: Record<TaskInputMode, string> = {
  case: "选择案例",
  entity: "调整目标企业输入",
  completeness: "资料完整性设置",
};

/** Stage changes inside the current task; applying never runs another experiment. */
export default function TaskInputs({
  mode,
  workspace,
  boot,
  target,
  busy,
  error,
  onChooseCase,
  onApply,
  onClose,
}: {
  mode: TaskInputMode;
  workspace: CaseBundle;
  boot: Bootstrap | null;
  target: string;
  busy: boolean;
  error: string;
  onChooseCase: (id: string) => Promise<boolean>;
  onApply: (update: TaskInputUpdate) => void;
  onClose: () => void;
}) {
  const titleId = useId();
  const heading = useRef<HTMLHeadingElement>(null);
  const [caseId, setCaseId] = useState(workspace.id);
  const [draft, setDraft] = useState(() => structuredClone(workspace.case));
  const [complete, setComplete] = useState({ ...workspace.completeness });
  const [holdingIndex, setHoldingIndex] = useState(() =>
    workspace.case.holdings.findIndex((row) => row.target === target),
  );
  const [valid, setValid] = useState(true);
  const [applying, setApplying] = useState(false);
  const [localError, setLocalError] = useState("");
  const locked = busy || applying;
  const entity = draft.entities.find((row) => row.eid === target);
  const incoming = draft.holdings
    .map((row, index) => ({ ...row, index }))
    .filter((row) => row.target === target);
  const holding = draft.holdings[holdingIndex];
  const others = incoming.reduce(
    (sum, row) => sum + (row.index === holdingIndex ? 0 : row.ratio),
    0,
  );
  const maximum = Math.max(holding?.ratio || 0, 1 - others) * 100;
  useEffect(() => {
    heading.current?.focus({ preventScroll: true });
  }, []);

  async function apply() {
    if (locked || !valid) return;
    if (mode === "case") {
      setApplying(true);
      setLocalError("");
      try {
        if (await onChooseCase(caseId)) onClose();
      } catch (failure) {
        setLocalError(
          failure instanceof Error ? failure.message : "案例加载失败，请重试。",
        );
      } finally {
        setApplying(false);
      }
    } else {
      onApply(mode === "entity" ? { case: draft } : { completeness: complete });
      onClose();
    }
  }

  return (
    <section
      className="task-input-panel"
      aria-labelledby={titleId}
      aria-busy={locked}
    >
      <header>
        <h3 id={titleId} ref={heading} tabIndex={-1}>
          {titles[mode]}
        </h3>
        <p>
          在当前实验内完成设置，应用后点击“运行本节实验”。不会自动运行其他推理。
        </p>
      </header>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void apply();
        }}
      >
        <fieldset disabled={locked}>
          <legend className="sr-only">{titles[mode]}</legend>
          {mode === "case" && (
            <>
              <label className="settings-field">
                本次实验案例
                <select
                  aria-label="本次实验案例"
                  value={caseId}
                  disabled={!boot}
                  onChange={(event) => setCaseId(event.target.value)}
                >
                  {boot?.cases.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                  {workspace.id === "custom" && (
                    <option value="custom">导入的案例</option>
                  )}
                </select>
              </label>
              <p className="helper-text">
                {boot?.cases.find((item) => item.id === caseId)?.detail ||
                  workspace.case.label}
              </p>
              <p className="helper-text">
                切换案例会更新其他实验的共享输入；有尚未导出的修改时会先确认。
              </p>
            </>
          )}
          {mode === "entity" && entity && (
            <>
              <p className="task-input-target">
                当前目标：{entity.eid} · {entity.name}
              </p>
              {incoming.length ? (
                <>
                  <label className="settings-field" htmlFor="task-holding-edge">
                    股东 → 当前企业
                  </label>
                  <select
                    id="task-holding-edge"
                    value={holdingIndex}
                    onChange={(event) =>
                      setHoldingIndex(Number(event.target.value))
                    }
                  >
                    {incoming.map((row) => (
                      <option key={row.index} value={row.index}>
                        {row.holder} ·{" "}
                        {draft.entities.find((item) => item.eid === row.holder)
                          ?.name || row.holder}{" "}
                        → {entity.eid}
                      </option>
                    ))}
                  </select>
                  <label
                    className="field-label range-label"
                    htmlFor="holding-value"
                  >
                    持股比例{" "}
                    <strong>{fmt((holding?.ratio || 0) * 100, 2)}%</strong>
                  </label>
                  <HoldingEditor
                    key={holdingIndex}
                    value={holding?.ratio || 0}
                    maximum={maximum}
                    busy={locked}
                    onValidity={setValid}
                    onChange={(ratio) =>
                      setDraft((old) => ({
                        ...old,
                        holdings: old.holdings.map((row, index) =>
                          index === holdingIndex ? { ...row, ratio } : row,
                        ),
                      }))
                    }
                  />
                  <p id="holding-help" className="helper-text">
                    其他股东合计 {fmt(others * 100, 2)}%，此项上限{" "}
                    {fmt(maximum, 2)}%。不会改写原始 CSV。
                  </p>
                </>
              ) : (
                <p className="empty-inline">该企业暂无直接持股记录。</p>
              )}
              <label
                className="field-label range-label"
                htmlFor="task-opacity-range"
              >
                不透明度代理 <strong>{entity.opacity.toFixed(2)}</strong>
              </label>
              <input
                id="task-opacity-range"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={entity.opacity}
                onChange={(event) => {
                  const opacity = Number(event.target.value);
                  setDraft((old) => ({
                    ...old,
                    entities: old.entities.map((row) =>
                      row.eid === target ? { ...row, opacity } : row,
                    ),
                  }));
                }}
              />
              <p className="helper-text">
                不透明度用于评分，不代替持股结构证明；均为教学设定。
              </p>
            </>
          )}
          {mode === "completeness" && (
            <>
              <p className="helper-text">
                资料缺失按未知处理，不视为零风险；这些开关不删除持股等原始记录。
              </p>
              <CompletenessFields
                value={complete}
                busy={locked}
                onChange={(key, value) =>
                  setComplete((old) => ({ ...old, [key]: value }))
                }
              />
            </>
          )}
        </fieldset>
        {(localError || (mode === "case" && error)) && (
          <p className="field-error" role="alert">
            {localError || error}
          </p>
        )}
        <div className="task-input-actions">
          <button
            type="submit"
            className="button primary"
            disabled={
              locked ||
              !valid ||
              (mode === "case" && !boot) ||
              (mode === "entity" && !entity)
            }
          >
            {applying
              ? "正在加载案例…"
              : mode === "case"
                ? "使用此案例，继续实验"
                : "应用并返回实验"}
          </button>
          <button
            type="button"
            className="button secondary"
            disabled={locked}
            onClick={onClose}
          >
            取消
          </button>
        </div>
      </form>
    </section>
  );
}
