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
  const xs = all.map((p) => p[0]),
    ys = all.map((p) => p[1]);
  const minX = Math.min(...xs),
    maxX = Math.max(...xs),
    minY = Math.min(...ys),
    maxY = Math.max(...ys);
  const centerX = (minX + maxX) / 2,
    centerY = (minY + maxY) / 2;
  const scale = Math.min(
    360 / Math.max(maxX - minX, 0.05),
    214 / Math.max(maxY - minY, 0.05),
  );
  // Equal pixels per coordinate unit on both axes, shared before/after plots.
  const x = (value: number) => 228 + (value - centerX) * scale;
  const y = (value: number) => 143 - (value - centerY) * scale;
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
      {result.projection && (
        <p className="lab4-projection-note">
          从 {result.projection.feature_dimensions} 维投影至2维 · 前两主成分保留{" "}
          {percent(
            result.projection.explained_variance.reduce(
              (sum, value) => sum + value,
              0,
            ),
          )}{" "}
          的方差信息。图中重叠不等于高维空间中无法区分。
        </p>
      )}
      <p className="lab4-chart-note">
        实心点为支持样本，空心点为查询样本，菱形为
        {descriptiveCenters ? "描述性类别中心" : "分类原型"}
        。颜色按真实类别显示，仅用于结果核对，不参与推理。两图使用同一PCA投影与坐标范围；二维距离不是模型实际分类距离。
        {(result.calibration === "graph" ||
          result.calibration === "multiview" ||
          result.calibration === "wide") &&
          Number(result.config?.graph_weight ?? 0) > 0 &&
          "本方法还使用近邻图传播，原型图只解释部分计算，不是全部分类边界。"}
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

export function ValidationCurve({
  rows,
}: {
  rows: LabFourSnapshot["training"];
}) {
  const valid = rows.filter(
    (r) => r.validation_1shot != null && r.validation_5shot != null,
  );
  if (!valid.length) return null;
  const x = (index: number) =>
    50 + (index / Math.max(valid.length - 1, 1)) * 710;
  const y = (value: number) => 178 - value * 200;
  const lines = [
    {
      key: "validation_1shot" as const,
      label: "验证集 1-shot",
      color: "var(--primary)",
    },
    {
      key: "validation_5shot" as const,
      label: "验证集 5-shot",
      color: "#466e58",
    },
  ];
  return (
    <figure className="lab4-validation-chart">
      <figcaption>
        验证集泛化表现<span>仅用于选型，未读取测试标签</span>
      </figcaption>
      <svg
        viewBox="0 0 810 225"
        role="img"
        aria-label={`验证集准确率曲线，共${valid.length}个训练轮次`}
      >
        {[0.2, 0.4, 0.6, 0.8].map((t) => (
          <g key={t}>
            <path
              d={`M50 ${y(t)}H760`}
              stroke="var(--border)"
              strokeDasharray="3 5"
            />
            <text
              x="40"
              y={y(t) + 4}
              textAnchor="end"
              fill="var(--muted)"
              fontSize="11"
            >
              {percent(t, 0)}
            </text>
          </g>
        ))}
        {lines.map((line) => (
          <g key={line.key}>
            <polyline
              points={valid
                .map((r, i) => `${x(i)},${y(r[line.key]!)}`)
                .join(" ")}
              stroke={line.color}
              strokeWidth="2.5"
              fill="none"
            />
            {valid.map((r, i) => (
              <circle
                key={r.epoch}
                cx={x(i)}
                cy={y(r[line.key]!)}
                r="2.5"
                fill={line.color}
              >
                <title>
                  第{r.epoch}轮 {line.label}：{percent(r[line.key]!, 2)}
                </title>
              </circle>
            ))}
          </g>
        ))}
        <text x="50" y="201" fill="var(--muted)" fontSize="11">
          第 {valid[0].epoch} 轮
        </text>
        <text
          x="760"
          y="201"
          textAnchor="end"
          fill="var(--muted)"
          fontSize="11"
        >
          第 {valid.at(-1)!.epoch} 轮
        </text>
      </svg>
      <div className="lab4-validation-legend">
        {lines.map((line) => (
          <span key={line.key}>
            <i style={{ background: line.color }} />
            {line.label}
          </span>
        ))}
      </div>
    </figure>
  );
}
