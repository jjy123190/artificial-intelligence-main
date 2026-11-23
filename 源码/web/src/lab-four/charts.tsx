import type {
  Comparison,
  EpisodeResult,
  LabFourSnapshot,
  Position,
} from "./types";

const colors = ["#954325", "#64784d", "#566f86", "#a1762c", "#826883"];
export const classColor = (index: number) => colors[index % colors.length];
export const percent = (value: number, digits = 1) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "暂无";
export const duration = (ms: number) =>
  Number.isFinite(ms)
    ? ms < 1000
      ? `${ms < 10 ? ms.toFixed(2) : Math.round(ms)} ms`
      : `${(ms / 1000).toFixed(2)} s`
    : "暂无";

const finitePosition = (point: Position) =>
  Array.isArray(point) && point.length === 2 && point.every(Number.isFinite);

export function FeatureComparison({ result }: { result: EpisodeResult }) {
  const adapting = result.feature_motion === "real_support_adaptation";
  const uncalibrated =
    result.method === "baseline" || result.calibration === "none";
  const descriptiveCenters =
    result.prototype_role === "descriptive_class_centers";
  const classNames = new Map(
    result.classes.map((item) => [item.id, item.name]),
  );
  const queryPredictions = new Map(
    result.queries.map((item) => [item.id, item.predicted_id]),
  );
  const stageLabels = adapting
    ? { before: "适应前", after: "适应后" }
    : uncalibrated
      ? { before: "支持集估计", after: "分类时" }
      : { before: "校准前", after: "校准后" };
  const points = result.points.filter(
    (p) => finitePosition(p.before) && finitePosition(p.after),
  );
  const prototypes = result.prototypes.filter(
    (p) => finitePosition(p.before) && finitePosition(p.after),
  );
  const all = [...points, ...prototypes].flatMap((p) => [p.before, p.after]);
  if (!all.length)
    return <p className="lab4-empty-inline">本次推理未返回二维特征投影。</p>;
  const extent = (axis: number) => {
    const values = all.map((p) => p[axis]);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const padding = Math.max((high - low) * 0.12, 0.1);
    return [low - padding, high + padding];
  };
  const [xMin, xMax] = extent(0);
  const [yMin, yMax] = extent(1);
  const x = (value: number) => 28 + ((value - xMin) / (xMax - xMin)) * 400;
  const y = (value: number) => 262 - ((value - yMin) / (yMax - yMin)) * 238;
  const color = (id: string) =>
    classColor(
      Math.max(
        0,
        result.classes.findIndex((c) => c.id === id),
      ),
    );
  return (
    <div className="lab4-feature-comparison">
      <div className="lab4-scatter-pair">
        {(["before", "after"] as const).map((stage) => (
          <figure key={stage}>
            <figcaption>{stageLabels[stage]}</figcaption>
            <svg
              viewBox="0 0 456 288"
              role="img"
              aria-label={`${stageLabels[stage]}的特征分布，共${points.length}个样本`}
            >
              <path d="M28 24V262H428" fill="none" stroke="var(--border)" />
              {[0.25, 0.5, 0.75].map((t) => (
                <path
                  key={t}
                  d={`M28 ${262 - t * 238}H428`}
                  stroke="var(--border)"
                  strokeDasharray="3 6"
                  opacity="0.6"
                />
              ))}
              {points.map((point) => (
                <circle
                  key={point.id}
                  cx={x(point[stage][0])}
                  cy={y(point[stage][1])}
                  r={point.kind === "support" ? 5 : 3.4}
                  fill={
                    point.kind === "support" ? color(point.class_id) : "none"
                  }
                  stroke={color(point.class_id)}
                  strokeWidth={point.kind === "support" ? 1 : 1.4}
                  opacity={point.kind === "support" ? 1 : 0.65}
                >
                  <title>
                    {point.kind === "support"
                      ? `支持样本 ${point.id}：${classNames.get(point.class_id) || point.class_id}`
                      : `查询样本 ${point.id}，真实类别：${classNames.get(point.class_id) || point.class_id}，预测类别：${classNames.get(queryPredictions.get(point.id) || "") || queryPredictions.get(point.id) || "未记录"}`}
                  </title>
                </circle>
              ))}
              {prototypes.map((prototype) => (
                <g key={prototype.class_id}>
                  {stage === "after" && (
                    <path
                      d={`M${x(prototype.before[0])} ${y(prototype.before[1])}L${x(prototype.after[0])} ${y(prototype.after[1])}`}
                      stroke={color(prototype.class_id)}
                      strokeWidth="1.5"
                      strokeDasharray="3 3"
                    />
                  )}
                  <path
                    d={`M${x(prototype[stage][0])} ${y(prototype[stage][1]) - 8}l8 8 -8 8 -8 -8Z`}
                    stroke={color(prototype.class_id)}
                    strokeWidth="2"
                    fill="var(--paper)"
                  >
                    <title>{`${descriptiveCenters ? "描述性类别中心" : "分类原型"} ${classNames.get(prototype.class_id) || prototype.class_id}`}</title>
                  </path>
                </g>
              ))}
            </svg>
          </figure>
        ))}
      </div>
      <div className="lab4-class-legend" aria-label="真实类别图例">
        {result.classes.map((item, index) => (
          <span key={item.id}>
            <i style={{ background: classColor(index) }} aria-hidden="true" />
            {item.name}
          </span>
        ))}
      </div>
      <p className="lab4-chart-note">
        实心点为支持样本，空心点为查询样本，菱形为
        {descriptiveCenters ? "描述性类别中心" : "分类原型"}
        。颜色按真实类别显示，仅用于结果核对，不参与推理。两图使用相同坐标范围。
        {adapting
          ? "MAML 使用支持集适应网络，样本的特征位置会变化；分类使用完整特征。"
          : uncalibrated
            ? "本次未执行校准，样本特征和类别原型均保持不变。"
            : result.calibration === "variance" || result.calibration === "full"
              ? "样本位置不变；方差校准改变类别距离度量，不能仅由二维点位移判断。"
              : "样本位置不变；本次校准调整类别原型，分类使用完整特征。"}
      </p>
    </div>
  );
}

export function ResultBars({ rows }: { rows: Comparison[] }) {
  const width = 840;
  const left = 190;
  const plotWidth = 570;
  const rowHeight = 61;
  const height = rows.length * rowHeight + 60;
  return (
    <div className="lab4-results-plot">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="不同方法的平均准确率与95%置信区间"
      >
        {[0, 20, 40, 60, 80, 100].map((tick) => (
          <g key={tick}>
            <path
              d={`M${left + (plotWidth * tick) / 100} 10V${height - 32}`}
              stroke="var(--border)"
              strokeDasharray={tick === 20 ? "4 4" : undefined}
            />
            <text
              x={left + (plotWidth * tick) / 100}
              y={height - 8}
              textAnchor="middle"
              fill="var(--muted)"
              fontSize="12"
            >
              {tick}%
            </text>
          </g>
        ))}
        {rows.map((row, index) => {
          const center = 28 + index * rowHeight;
          const estimate =
            left + plotWidth * Math.max(0, Math.min(1, row.accuracy));
          const lower = left + plotWidth * Math.max(0, row.accuracy - row.ci95);
          const upper = left + plotWidth * Math.min(1, row.accuracy + row.ci95);
          return (
            <g key={`${row.method}-${row.shot}`}>
              <text
                x={left - 16}
                y={center + 4}
                textAnchor="end"
                fill="var(--ink)"
                fontSize="14"
              >
                {row.label}
              </text>
              <rect
                x={left}
                y={center - 12}
                width={Math.max(0, estimate - left)}
                height="24"
                rx="3"
                fill={
                  row.method === "calibrated" || row.method === "full"
                    ? "var(--primary)"
                    : "var(--chart-secondary)"
                }
              />
              <path
                d={`M${lower} ${center}H${upper}M${lower} ${center - 5}V${center + 5}M${upper} ${center - 5}V${center + 5}`}
                fill="none"
                stroke="var(--ink)"
                strokeWidth="1.3"
              />
              <text
                x={Math.min(upper + 10, width - 54)}
                y={center + 5}
                fill="var(--ink)"
                fontSize="13"
                fontWeight="600"
              >
                {percent(row.accuracy)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function TrainingCurve({ rows }: { rows: LabFourSnapshot["training"] }) {
  const valid = rows.filter(
    (row) => Number.isFinite(row.loss) && Number.isFinite(row.epoch),
  );
  if (!valid.length) return null;
  const min = Math.min(...valid.map((row) => row.loss));
  const max = Math.max(...valid.map((row) => row.loss));
  const range = Math.max(max - min, 0.01);
  const x = (index: number) =>
    50 + (index / Math.max(valid.length - 1, 1)) * 710;
  const y = (loss: number) => 164 - ((loss - min) / range) * 124;
  return (
    <svg
      viewBox="0 0 810 205"
      className="lab4-training-curve"
      role="img"
      aria-label={`训练损失由${valid[0].loss.toFixed(3)}变化至${valid.at(-1)!.loss.toFixed(3)}`}
    >
      <path d="M50 24V168H760" stroke="var(--border)" fill="none" />
      <text x="40" y="44" textAnchor="end" fill="var(--muted)" fontSize="11">
        {max.toFixed(2)}
      </text>
      <text x="40" y="168" textAnchor="end" fill="var(--muted)" fontSize="11">
        {min.toFixed(2)}
      </text>
      <polyline
        points={valid
          .map((row, index) => `${x(index)},${y(row.loss)}`)
          .join(" ")}
        stroke="var(--primary)"
        strokeWidth="2.5"
        fill="none"
      />
      {valid.map((row, index) => (
        <circle
          key={row.epoch}
          cx={x(index)}
          cy={y(row.loss)}
          r="3"
          fill="var(--primary)"
        >
          <title>{`第${row.epoch}轮：${row.loss.toFixed(4)}`}</title>
        </circle>
      ))}
      <text x="50" y="194" fill="var(--muted)" fontSize="12">
        第 {valid[0].epoch} 轮
      </text>
      <text x="760" y="194" textAnchor="end" fill="var(--muted)" fontSize="12">
        第 {valid.at(-1)!.epoch} 轮
      </text>
    </svg>
  );
}
