import type { Comparison, LabFourSnapshot, ModuleContrast } from "./types";
import { PairedPlot } from "./EvidencePlots";
import { percent } from "./charts";
import { signedPoints } from "./presentation";

export default function ModuleContrasts({
  evidence,
  comparisons,
  shots,
}: {
  evidence: NonNullable<LabFourSnapshot["module_contrasts"]>;
  comparisons: Comparison[];
  shots: 1 | 5;
}) {
  const rows = evidence.rows.filter((row) => row.shot === shots);
  const label = (method: string) =>
    comparisons.find((row) => row.shot === shots && row.method === method)
      ?.label || method;
  const comparison = (row: ModuleContrast) =>
    `${label(row.method)} − ${label(row.reference)}`;
  if (!rows.length)
    return (
      <p className="lab4-empty-inline">
        当前版本没有可核验的模块直接对照，不用均值区间推造配对区间。
      </p>
    );
  return (
    <div className="lab4-module-contrasts">
      <p className="lab4-contrast-explainer">
        每行只比较一组明确方法，不把“相对Baseline的总提升”当成模块贡献。正值表示左边方法高于右边方法；详细条件保留在下方。
      </p>
      <PairedPlot
        rows={rows.map((row) => ({
          method: row.id,
          label: row.label,
          paired_delta: row.paired_delta,
          paired_ci95: row.paired_ci95,
          detail: comparison(row),
        }))}
        title="指定方法间的直接配对差"
        subtitle="同一批冻结任务 · 各行参考方法不同"
        note={evidence.interval_note}
      />
      <details className="lab4-data-details lab4-contrast-details">
        <summary>查看六组对照的条件与精确区间</summary>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="模块直接配对差详情，可横向滚动"
        >
          <table>
            <caption className="sr-only">
              逐任务直接重算的模块对照，不是独立区间相减
            </caption>
            <thead>
              <tr>
                <th scope="col">对照条件</th>
                <th scope="col">方法准确率</th>
                <th scope="col">配对差及95%区间</th>
                <th scope="col">改善 / 持平 / 下降</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <th scope="row">
                    <strong>{row.label}</strong>
                    <small>{comparison(row)}</small>
                    <p>{row.scope}</p>
                  </th>
                  <td>
                    {percent(row.method_accuracy, 2)}
                    <br />
                    <span className="lab4-muted">
                      对照 {percent(row.reference_accuracy, 2)}
                    </span>
                  </td>
                  <td>
                    <strong>{signedPoints(row.paired_delta)} pp</strong>
                    <small>
                      [{signedPoints(row.ci95_low, 3)},{" "}
                      {signedPoints(row.ci95_high, 3)}] pp
                    </small>
                  </td>
                  <td>
                    {row.wins} / {row.ties} / {row.losses}
                    <small>{row.episodes}个配对任务</small>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <p className="lab4-chart-note">
        {evidence.scope}{" "}
        多个对照中的边缘正区间只是逐项诊断，未作多重比较校正，不宜写成“所有模块均显著有效”。
      </p>
    </div>
  );
}
