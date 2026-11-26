import type { LabFourSnapshot } from "./types";
import { percent } from "./charts";
import { ReliabilityPlot, TaskVariation } from "./EvidencePlots";

export default function Reliability({
  diagnostics,
  shots,
}: {
  diagnostics: NonNullable<LabFourSnapshot["diagnostics"]>;
  shots: 1 | 5;
}) {
  const row = diagnostics.rows.find((r) => r.shots === shots);
  if (!row) return null;
  return (
    <section
      className="lab4-evaluation-section lab4-reliability"
      aria-labelledby="lab4-reliability-title"
    >
      <header>
        <h2 id="lab4-reliability-title">不只看平均分：任务波动与分数可靠性</h2>
        <p>以下为双视图方案的固定测试预测诊断，不筛掉难任务，也不据此调参。</p>
      </header>
      <div className="lab4-reliability-facts">
        <div>
          <span>中间90%的任务准确率</span>
          <strong>
            {percent(row.task_p05, 1)} – {percent(row.task_p95, 1)}
          </strong>
          <small>任务的5%–95%分位，不是均值置信区间</small>
        </div>
        <div>
          <span>任务中位数</span>
          <strong>{percent(row.task_median, 1)}</strong>
          <small>{row.episodes} 个任务，全部保留</small>
        </div>
        <div>
          <span>相比单视图图传播</span>
          <strong>
            {row.paired_wins} / {row.paired_ties} / {row.paired_losses}
          </strong>
          <small>改善 / 持平 / 下降的任务数</small>
        </div>
      </div>
      <TaskVariation row={row} />
      <ReliabilityPlot row={row} />
      <details className="lab4-score-reliability">
        <summary>
          模型类别分数有多可靠？ <span>特征校准不等于概率校准</span>
        </summary>
        <p>
          十等宽分箱的平均分数与实际正确率之间，加权绝对差为{" "}
          <strong>{(row.ece10 * 100).toFixed(2)} 个百分点</strong>
          （ECE）。它只描述本次预测，不能据此保证未来概率可靠。
        </p>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="类别分数与实际正确率，可横向滚动"
        >
          <table>
            <caption className="sr-only">双视图模型分数校准诊断</caption>
            <thead>
              <tr>
                <th scope="col">预测类别分数区间</th>
                <th scope="col">查询出现次数</th>
                <th scope="col">平均分数</th>
                <th scope="col">实际正确率</th>
              </tr>
            </thead>
            <tbody>
              {row.score_bins
                .filter((b) => b.occurrences > 0)
                .map((b) => (
                  <tr key={b.lower}>
                    <th scope="row">
                      {percent(b.lower, 0)} – {percent(b.upper, 0)}
                    </th>
                    <td>{b.occurrences.toLocaleString("zh-CN")}</td>
                    <td>{percent(b.mean_score!, 1)}</td>
                    <td>{percent(b.accuracy!, 1)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <p className="lab4-footnote">
          统计 {row.query_occurrences.toLocaleString("zh-CN")}{" "}
          次查询预测；同一图片可能在多个任务出现，并非同等数量的独立图片。
          {diagnostics.scope}
        </p>
      </details>
    </section>
  );
}
