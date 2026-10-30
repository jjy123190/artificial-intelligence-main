import type { Bootstrap, Completeness, Parameters } from "../types";
import CompletenessFields from "./CompletenessFields";
export default function AnalysisSettings({
  boot,
  parameters,
  completeness,
  busy,
  onParameters,
  onCompleteness,
}: {
  boot: Bootstrap | null;
  parameters: Parameters;
  completeness: Completeness;
  busy: boolean;
  onParameters: (p: Partial<Parameters>) => void;
  onCompleteness: (key: keyof Completeness, value: boolean) => void;
}) {
  return (
    <div className="settings-page-layout">
      <section className="simple-panel settings-section">
        <header>
          <h2>推理与模型</h2>
          <p>设置完成后运行推理；切换页面不会自动丢弃已接受的设置。</p>
        </header>
        <label className="settings-field">
          推理方式
          <select
            aria-label="推理方式"
            value={parameters.method}
            disabled={busy || !boot}
            onChange={(e) => onParameters({ method: e.target.value })}
          >
            {boot?.methods.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
        <label className="settings-field">
          参数来源
          <select
            aria-label="参数来源"
            value={parameters.model_source}
            disabled={busy || !boot}
            onChange={(e) => onParameters({ model_source: e.target.value })}
          >
            {boot?.models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
        <label className="settings-field range-label" htmlFor="threshold-range">
          复核阈值<strong>{parameters.threshold.toFixed(2)}</strong>
        </label>
        <input
          id="threshold-range"
          type="range"
          min="0.05"
          max="0.95"
          step="0.05"
          value={parameters.threshold}
          disabled={busy}
          onChange={(e) => onParameters({ threshold: Number(e.target.value) })}
        />
        <p className="helper-text">
          结构锁定可保留低于评分阈值的已证预警；评分和复核行动是不同目标。
        </p>
      </section>
      <section className="simple-panel settings-section">
        <header>
          <h2>资料完整性</h2>
          <p>资料缺失按未知处理，而不是按零风险处理。</p>
        </header>
        <CompletenessFields
          value={completeness}
          busy={busy}
          onChange={onCompleteness}
        />
        <p className="helper-text">
          全部为当前案例的教学设定，不改变原始CSV；训练参数和策略不保证在所有条件中改善。
        </p>
      </section>
    </div>
  );
}
