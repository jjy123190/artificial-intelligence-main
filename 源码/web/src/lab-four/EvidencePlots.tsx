import type { CSSProperties } from "react";
import { percent } from "./charts";
import { signedPoints } from "./presentation";
import type { Comparison, LabFourSnapshot } from "./types";

type Vars = CSSProperties & Record<`--${string}`, string>;
const pos = (value: number) => `${Math.max(0, Math.min(1, value)) * 100}%`;

export function AccuracyPlot({
  rows,
  selected,
  onSelect,
}: {
  rows: Comparison[];
  selected?: string;
  onSelect: (method: string) => void;
}) {
  return (
    <div
      className="lab4-accuracy-plot"
      role="group"
      aria-label="方法准确率与95%区间，选择一行查看对应结论"
    >
      <div className="lab4-plot-axis" aria-hidden="true">
        <span />
        <div>
          {[0, 20, 40, 60, 80, 100].map((n) => (
            <span key={n} style={{ left: `${n}%` }}>
              {n}%
            </span>
          ))}
        </div>
        <span />
      </div>
      {rows.map((row) => (
        <button
          type="button"
          className="lab4-plot-row"
          key={row.method}
          aria-pressed={selected === row.method}
          onClick={() => onSelect(row.method)}
          style={
            {
              "--estimate": pos(row.accuracy),
              "--lower": pos(row.accuracy - row.ci95),
              "--upper": pos(row.accuracy + row.ci95),
            } as Vars
          }
        >
          <span className="lab4-plot-label">
            {row.label}
            <small>
              {row.method === "wide" ||
              row.method === "wideview" ||
              row.method === "continued"
                ? "独立较宽模型 · 非同预算消融"
                : row.transductive
                  ? row.balanced_queries
                    ? "均衡转导"
                    : "转导"
                  : "归纳"}
            </small>
          </span>
          <span className="lab4-plot-track" aria-hidden="true">
            <span className="lab4-chance-line" />
            <span className="lab4-plot-bar" />
            <span className="lab4-plot-interval" />
          </span>
          <span className="lab4-plot-value">
            {percent(row.accuracy, 2)}
            <small>± {(row.ci95 * 100).toFixed(2)} pp</small>
          </span>
        </button>
      ))}
      <p className="lab4-chart-note">
        条形从0%开始；细线为均值的95%区间。虚线是20%的随机猜测参考，不是最低准确率。
      </p>
    </div>
  );
}

type PairedRow = Pick<
  Comparison,
  "method" | "label" | "paired_delta" | "paired_ci95"
> & { detail?: string };
export function PairedPlot({
  rows,
  title = "相对同版 Baseline 的配对差",
  subtitle = "不是相对完整方案的模块贡献",
  note = "点为平均配对差，细线为95%区间；这里的参考对象始终是Baseline。要比较模块贡献，请切换到模块直接对照。",
}: {
  rows: PairedRow[];
  title?: string;
  subtitle?: string;
  note?: string;
}) {
  const valid = rows.filter(
    (r) => r.paired_delta != null && r.paired_ci95 != null,
  );
  if (!valid.length)
    return <p className="lab4-empty-inline">没有可绘制的逐任务配对差。</p>;
  const extent = Math.max(
    0.05,
    Math.ceil(
      Math.max(
        ...valid.map((r) => Math.abs(r.paired_delta!) + r.paired_ci95!),
      ) * 20,
    ) / 20,
  );
  const coordinate = (value: number) => pos((value + extent) / (2 * extent));
  return (
    <figure className="lab4-paired-plot">
      <figcaption>
        {title} <span>{subtitle}</span>
      </figcaption>
      <div className="lab4-plot-axis" aria-hidden="true">
        <span />
        <div>
          {[-extent, 0, extent].map((n) => (
            <span key={n} style={{ left: coordinate(n) }}>
              {signedPoints(n, 0)} pp
            </span>
          ))}
        </div>
        <span />
      </div>
      {valid.map((row) => {
        const mean = row.paired_delta!,
          ci = row.paired_ci95!,
          state =
            mean - ci > 0 ? "positive" : mean + ci < 0 ? "negative" : "neutral";
        return (
          <div
            key={row.method}
            className={`lab4-plot-row lab4-paired-row ${state}`}
            style={
              {
                "--estimate": coordinate(mean),
                "--lower": coordinate(mean - ci),
                "--upper": coordinate(mean + ci),
              } as Vars
            }
          >
            <span className="lab4-plot-label">
              {row.label}
              {row.detail && (
                <small className="lab4-pair-comparator">{row.detail}</small>
              )}
              <small>
                {state === "positive"
                  ? "区间高于零"
                  : state === "negative"
                    ? "区间低于零"
                    : "区间跨零"}
              </small>
            </span>
            <span className="lab4-plot-track" aria-hidden="true">
              <span className="lab4-zero-line" />
              <span className="lab4-plot-interval" />
              <span className="lab4-estimate-dot" />
            </span>
            <span className="lab4-plot-value">
              {signedPoints(mean)} pp<small>± {(ci * 100).toFixed(2)} pp</small>
            </span>
          </div>
        );
      })}
      <p className="lab4-chart-note">{note}</p>
    </figure>
  );
}

type Timing = NonNullable<LabFourSnapshot["edge_benchmark"]>["rows"];
export function LatencyPlot({
  rows,
  names,
}: {
  rows: Timing;
  names: Record<string, string>;
}) {
  if (!rows.length)
    return <p className="lab4-empty-inline">当前shot没有耗时记录。</p>;
  const maximum = Math.max(
    1,
    ...rows.map((r) => r.encoding_and_classification_p95_ms),
  );
  return (
    <figure className="lab4-latency-plot">
      <figcaption>
        真实编码与分类耗时 <span>实色：中位数　末端标记：P95</span>
      </figcaption>
      {rows.map((row) => (
        <div
          key={row.method}
          className="lab4-plot-row"
          style={
            {
              "--estimate": pos(
                row.encoding_and_classification_median_ms / maximum,
              ),
              "--upper": pos(row.encoding_and_classification_p95_ms / maximum),
            } as Vars
          }
        >
          <span className="lab4-plot-label">
            {names[row.method] || row.method}
            <small>
              {row.repetitions}次重复 · {row.shots}-shot
            </small>
          </span>
          <span className="lab4-plot-track" aria-hidden="true">
            <span className="lab4-latency-tail" />
            <span className="lab4-plot-bar" />
            <span className="lab4-latency-p95" />
          </span>
          <span className="lab4-plot-value">
            {row.encoding_and_classification_median_ms.toFixed(1)} ms
            <small>
              P95 {row.encoding_and_classification_p95_ms.toFixed(1)} ms
            </small>
          </span>
        </div>
      ))}
      <p className="lab4-chart-note">
        横轴从0开始，上限{maximum.toFixed(1)}{" "}
        ms。排除加载、文件校验、HTTP传输和缩略图渲染，不代表手机或ARM的实测性能。
      </p>
    </figure>
  );
}

type Diagnostic = NonNullable<LabFourSnapshot["diagnostics"]>["rows"][number];
export function ReliabilityPlot({ row }: { row: Diagnostic }) {
  const bins = row.score_bins.filter(
    (b) => b.occurrences > 0 && b.mean_score != null && b.accuracy != null,
  );
  const x = (n: number) => 56 + n * 392,
    y = (n: number) => 306 - n * 264;
  return (
    <figure className="lab4-calibration-plot">
      <figcaption>
        分数与实际正确率 <span>每个点代表一个非空分数区间</span>
      </figcaption>
      <svg
        viewBox="0 0 490 357"
        role="img"
        aria-label={`双视图预测可靠性图，${bins.length}个非空分箱；虚线表示平均分数等于正确率，分箱精确值见下方表格`}
      >
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map((tick) => (
          <g key={tick}>
            <path d={`M56 ${y(tick)}H448`} stroke="var(--border)" />
            <text
              x="45"
              y={y(tick) + 4}
              textAnchor="end"
              fill="var(--muted)"
              fontSize="12"
            >
              {percent(tick, 0)}
            </text>
            <text
              x={x(tick)}
              y="328"
              textAnchor="middle"
              fill="var(--muted)"
              fontSize="12"
            >
              {percent(tick, 0)}
            </text>
          </g>
        ))}
        <path
          d={`M56 306L448 42`}
          stroke="var(--muted)"
          strokeDasharray="5 5"
          fill="none"
        />
        {bins.map((bin) => (
          <g key={bin.lower}>
            <path
              d={`M${x(bin.mean_score!)} ${y(bin.mean_score!)}V${y(bin.accuracy!)}`}
              stroke="var(--primary)"
              strokeWidth="2"
              opacity=".5"
            />
            <circle
              cx={x(bin.mean_score!)}
              cy={y(bin.accuracy!)}
              r="5"
              fill="var(--primary)"
            >
              <title>{`${percent(bin.lower, 0)}–${percent(bin.upper, 0)}分箱：${bin.occurrences}次，分数${percent(bin.mean_score!, 2)}，正确率${percent(bin.accuracy!, 2)}`}</title>
            </circle>
          </g>
        ))}
        <text x="56" y="20" fill="var(--ink)" fontSize="13">
          实际正确率
        </text>
        <text x="448" y="351" textAnchor="end" fill="var(--ink)" fontSize="13">
          模型平均分数
        </text>
      </svg>
      <p className="lab4-chart-note">
        点在对角线上方：分数偏保守；下方：分数偏自信。连线显示差距，不代表置信区间。空分箱不补点。
      </p>
    </figure>
  );
}

export function TaskVariation({ row }: { row: Diagnostic }) {
  return (
    <div className="lab4-diagnostic-overview">
      <figure className="lab4-task-range">
        <figcaption>
          任务准确率的分布摘要 <span>不是均值置信区间</span>
        </figcaption>
        <div
          className="lab4-task-range-track"
          style={
            {
              "--lower": pos(row.task_p05),
              "--upper": pos(row.task_p95),
              "--estimate": pos(row.task_median),
            } as Vars
          }
          aria-hidden="true"
        >
          <span className="lab4-plot-interval" />
          <span className="lab4-estimate-dot" />
        </div>
        <div className="lab4-range-labels">
          <span>P05 {percent(row.task_p05)}</span>
          <strong>中位数 {percent(row.task_median)}</strong>
          <span>P95 {percent(row.task_p95)}</span>
        </div>
      </figure>
      <figure className="lab4-paired-outcomes">
        <figcaption>比单视图图传播更好，还是更差？</figcaption>
        <div className="lab4-outcome-track" aria-hidden="true">
          {[
            [row.paired_wins, "win"],
            [row.paired_ties, "tie"],
            [row.paired_losses, "loss"],
          ].map(([n, key]) => (
            <span
              key={key}
              className={`is-${key}`}
              style={{ flex: Number(n) }}
            />
          ))}
        </div>
        <div className="lab4-range-labels">
          <span>改善 {row.paired_wins}</span>
          <span>持平 {row.paired_ties}</span>
          <span>下降 {row.paired_losses}</span>
        </div>
        <p className="lab4-chart-note">
          全部{row.episodes}个配对任务，不只展示改善的任务。
        </p>
      </figure>
    </div>
  );
}
