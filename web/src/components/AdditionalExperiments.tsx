import { useMemo, useState } from "react";
import type { ExperimentsData, ExperimentRow } from "../types";
const modelNames: Record<string, string> = {
  rules: "指标规则",
  bayes_manual: "Bayes · 手工参数",
  bayes_learned: "Bayes · CPT学习",
  fuzzy: "Mamdani · 手工参数",
  fuzzy_learned: "Mamdani · 自动拟合",
  hybrid_fixed: "固定融合权重",
  hybrid_tuned: "验证选择权重",
  hybrid_adaptive: "自动拟合融合",
  hybrid_no_rule: "消融：移除规则",
  hybrid_no_fuzzy: "消融：移除Mamdani",
  hybrid_no_bayes: "消融：移除Bayes",
  adaptive_no_rule: "自动拟合 · 移除规则",
  adaptive_no_fuzzy: "自动拟合 · 移除Mamdani",
  adaptive_no_bayes: "自动拟合 · 移除Bayes",
  ds_classic: "经典D-S",
  ds_yager: "冲突转无知",
  ds_discount: "可靠性折扣",
  ds_discount_yager: "折扣＋冲突转无知",
};
const scenarios: Record<string, string> = {
  clean: "清洁指标",
  noise_010: "数值噪声0.10",
  noise_025: "数值噪声0.25",
  missing_020: "指标缺失20%",
  missing_040: "指标缺失40%",
  missing_all: "全部缺测",
  combined: "噪声＋缺测",
};
const number = (value: unknown, digits = 3) =>
  typeof value === "number" ? value.toFixed(digits) : "未记录";
const percent = (value: unknown) =>
  typeof value === "number" ? (value * 100).toFixed(1) + "%" : "未记录";
const fallbackNote = (row: ExperimentRow) => {
  const count = Number(row.ablation_fallback_runs || 0);
  if (!count) return "";
  const components: Record<string, string> = {
    bayes: "Bayes",
    fuzzy: "Mamdani",
    rule: "指标规则",
  };
  const baseline = String(row.ablation_fallback_component || "")
    .split(";")
    .map((name) => components[name] || name)
    .join("／");
  return `${count}/${row.ablation_runs} 个种子回退为 ${baseline} 基线`;
};
const mean = (rows: ExperimentRow[], field: string) => {
  const values = rows
    .map((row) => row[field])
    .filter((v): v is number => typeof v === "number");
  return values.length
    ? values.reduce((a, b) => a + b, 0) / values.length
    : undefined;
};
export default function AdditionalExperiments({
  data,
  section,
}: {
  data: ExperimentsData;
  section: "deterministic" | "scoring" | "conflict";
}) {
  const [condition, setCondition] = useState("clean");
  const groups = useMemo(
    () =>
      [...new Set((data.goal_slicing || []).map((r) => Number(r.groups)))].sort(
        (a, b) => a - b,
      ),
    [data],
  );
  if (section === "deterministic")
    return (
      <>
        <section className="results-section">
          <header>
            <h2>目标相关规则切片</h2>
            <p>
              只保留目标依赖的规则。与完整规则集核对目标答案一致；下表为同规模重复实测的均值。
            </p>
          </header>
          <div className="table-scroll">
            <table aria-label="目标切片实测">
              <thead>
                <tr>
                  <th>规模／组</th>
                  <th>完整规则</th>
                  <th>保留规则</th>
                  <th>目标答案一致</th>
                  <th>查询耗时／ms</th>
                </tr>
              </thead>
              <tbody>
                {groups.map((group) => {
                  const rows = data.goal_slicing.filter(
                    (r) => r.groups === group,
                  );
                  return (
                    <tr key={group}>
                      <th>{group}</th>
                      <td>{number(mean(rows, "input_rules"), 0)}</td>
                      <td>{number(mean(rows, "retained_rules"), 0)}</td>
                      <td>
                        {rows.every((r) => r.target_equal === true)
                          ? "全部一致"
                          : "存在不一致，需检查"}
                      </td>
                      <td>{number(mean(rows, "elapsed_ms"), 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
        <section className="results-section">
          <header>
            <h2>独立子案例归结搜索优化</h2>
            <p>
              FIFO基线与优化搜索使用同一子句集，包括无关事实干扰；都要完成证明，不能用提前停止伪造速度收益。
            </p>
          </header>
          <div className="table-scroll">
            <table aria-label="归结搜索优化实测">
              <thead>
                <tr>
                  <th>方法</th>
                  <th>证明状态</th>
                  <th>子句对检查</th>
                  <th>选取子句数</th>
                  <th>平均耗时／ms</th>
                </tr>
              </thead>
              <tbody>
                {["fifo_baseline", "optimized"].map((strategy) => {
                  const rows = (data.resolution || []).filter(
                    (r) => r.model === strategy,
                  );
                  return (
                    <tr key={strategy}>
                      <th>
                        {strategy === "optimized" ? "优化搜索" : "FIFO基线"}
                      </th>
                      <td>
                        {rows.length && rows.every((r) => r.status === "proved")
                          ? "全部完成证明"
                          : "未完整记录"}
                      </td>
                      <td>{number(mean(rows, "pairs"), 0)}</td>
                      <td>{number(mean(rows, "selected"), 0)}</td>
                      <td>{number(mean(rows, "elapsed_ms"), 2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </>
    );
  if (section === "conflict")
    return (
      <section className="results-section">
        <header>
          <h2>独立合成来源的D-S冲突与可靠性实验</h2>
          <p>
            这是额外实验，不把读取同一组指标的Bayes与Mamdani假定成独立证据相乘。完全冲突与未知必须明确处理。
          </p>
        </header>
        <div className="table-scroll">
          <table aria-label="证据冲突扩展实测">
            <thead>
              <tr>
                <th>设定冲突率</th>
                <th>融合策略</th>
                <th>准确率</th>
                <th>F1</th>
                <th>平均无知量</th>
                <th>推理／μs</th>
              </tr>
            </thead>
            <tbody>
              {(data.conflict_summary || []).map((row, i) => (
                <tr key={i}>
                  <td>{percent(row.conflict_rate)}</td>
                  <th>{modelNames[String(row.model)] || row.model}</th>
                  <td>{percent(row.accuracy_mean)}</td>
                  <td>{number(row.f1_mean)}</td>
                  <td>{number(row.mean_ignorance_mean)}</td>
                  <td>{number(row.inference_us_mean, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="helper-text">
          读取
          outputs/lab02/experiments/conflict_summary.csv；来源质量与冲突率是合成设定，不是现实凭证真实性认证。
        </p>
      </section>
    );
  return (
    <>
      <section className="results-section">
        <header>
          <h2>评分算法优化与消融</h2>
          <p>
            手工参数、训练参数、验证选择权重与移除组件的对照，使用同一批合成指标样本。这里评估评分分类，不混用图级复核行动指标。
          </p>
        </header>
        <label className="task-inline-field">
          指标观测条件
          <select
            aria-label="评分实验观测条件"
            value={condition}
            onChange={(e) => setCondition(e.target.value)}
          >
            {[
              ...new Set(data.indicator_summary.map((r) => String(r.scenario))),
            ].map((value) => (
              <option key={value} value={value}>
                {scenarios[value] || value}
              </option>
            ))}
          </select>
        </label>
        <div className="table-scroll">
          <table aria-label="评分优化与消融实测">
            <thead>
              <tr>
                <th>方法</th>
                <th>准确率</th>
                <th>F1</th>
                <th>评分损失↓</th>
                <th>推理／μs</th>
              </tr>
            </thead>
            <tbody>
              {data.indicator_summary
                .filter((row) => row.scenario === condition)
                .map((row) => (
                  <tr key={String(row.model)}>
                    <th>
                      {modelNames[String(row.model)] || row.model}
                      {fallbackNote(row) && (
                        <div className="helper-text">{fallbackNote(row)}</div>
                      )}
                    </th>
                    <td>{percent(row.accuracy_mean)}</td>
                    <td>{number(row.f1_mean)}</td>
                    <td>{number(row.brier_mean, 4)}</td>
                    <td>{number(row.inference_us_mean, 2)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <p className="helper-text">
          准确率与F1来自独立测试集，模型选择只使用训练／验证集。结果相同也如实保留，不强迫融合胜出。
          标注“回退为基线”的行表示移除后没有剩余正权重，采用预先指定的单模型对照，不能解释为组件边际贡献。
        </p>
      </section>
      <details className="task-details">
        <summary>查看先验、CPT与模糊参数敏感性原始实测</summary>
        <div className="table-scroll">
          <table aria-label="参数敏感性实测">
            <thead>
              <tr>
                <th>种子</th>
                <th>方法</th>
                <th>被改变的参数</th>
                <th>值</th>
                <th>F1</th>
                <th>评分损失↓</th>
              </tr>
            </thead>
            <tbody>
              {(data.sensitivity || []).map((row, i) => (
                <tr key={i}>
                  <td>{row.seed}</td>
                  <td>{row.model}</td>
                  <td>{row.parameter}</td>
                  <td>{number(row.value, 2)}</td>
                  <td>{number(row.f1)}</td>
                  <td>{number(row.brier, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}
