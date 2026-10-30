import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ExperimentRow } from "../types";

const methods: Record<string, string> = {
  naive_scan: "全库扫描",
  indexed_naive: "索引朴素",
  semi_naive: "增量连接",
};
const colors = [
  "var(--graph-guarantee)",
  "var(--graph-trade)",
  "var(--primary)",
];
const value = (row: ExperimentRow, key: string, digits = 0) =>
  typeof row[key] === "number"
    ? row[key].toLocaleString("zh-CN", { maximumFractionDigits: digits })
    : "未记录";

export default function PerformancePanel({ rows }: { rows: ExperimentRow[] }) {
  const groups = useMemo(
    () =>
      [...new Set(rows.map((row) => Number(row.groups)))].sort((a, b) => a - b),
    [rows],
  );
  const [size, setSize] = useState(groups.at(-1) || 1);
  const trend = groups.map((group) =>
    Object.fromEntries([
      ["groups", group],
      ...rows
        .filter((row) => row.groups === group)
        .map((row) => [String(row.model), row.elapsed_ms_mean]),
    ]),
  );
  if (!rows.length)
    return (
      <p className="method-note">
        尚无确定性性能实测。运行 python main.py benchmark 后可在此查看。
      </p>
    );
  return (
    <section className="results-section performance-section">
      <header>
        <h2>确定性优化：相同闭包，不同搜索代价</h2>
        <p>
          原始扫描、索引与增量连接在相同网络上对照。时间单位为毫秒，纵轴使用对数尺度；非当前网页计时。
        </p>
      </header>
      <div
        className="chart-area"
        tabIndex={0}
        role="region"
        aria-label="性能曲线，小屏可横向滚动"
      >
        <ResponsiveContainer width="100%" height="100%" minWidth={480}>
          <LineChart
            data={trend}
            margin={{ top: 16, right: 32, left: 12, bottom: 16 }}
          >
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis
              dataKey="groups"
              tickFormatter={(v) => v + " 组网络"}
              tick={{ fontSize: 12, fill: "var(--ink)" }}
            />
            <YAxis
              scale="log"
              domain={["auto", "auto"]}
              tick={{ fontSize: 12, fill: "var(--muted)" }}
              tickFormatter={(v) => Number(v).toLocaleString()}
            />
            <Tooltip
              formatter={(v) =>
                typeof v === "number" ? v.toFixed(2) + " ms" : "未记录"
              }
              contentStyle={{
                background: "var(--bg)",
                border: "1px solid var(--border)",
                borderRadius: 8,
              }}
            />
            <Legend />
            {Object.entries(methods).map(([key, name], index) => (
              <Line
                key={key}
                dataKey={key}
                name={name}
                stroke={colors[index]}
                strokeWidth={2}
                dot={{ r: 4 }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <label className="performance-filter">
        <span>核对网络规模</span>
        <select value={size} onChange={(e) => setSize(Number(e.target.value))}>
          {groups.map((group) => (
            <option key={group} value={group}>
              {group} 组网络
            </option>
          ))}
        </select>
      </label>
      <div
        className="table-scroll"
        role="region"
        tabIndex={0}
        aria-label="确定性优化实测"
      >
        <table>
          <thead>
            <tr>
              <th>策略</th>
              <th>运行时间 / ms</th>
              <th>候选匹配检查</th>
              <th>规则实例</th>
              <th>迭代轮次</th>
              <th>闭包事实数</th>
            </tr>
          </thead>
          <tbody>
            {rows
              .filter((row) => row.groups === size)
              .map((row) => (
                <tr key={String(row.model)}>
                  <th>{methods[String(row.model)] || String(row.model)}</th>
                  <td>{value(row, "elapsed_ms_mean", 2)}</td>
                  <td>{value(row, "candidate_checks_mean")}</td>
                  <td>{value(row, "rule_instances_mean")}</td>
                  <td>{value(row, "cycles_mean")}</td>
                  <td>{value(row, "facts_mean")}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <p className="helper-text">
        候选检查次数衡量搜索工作量，不等同于内存占用。三次重复均值来自
        output/benchmarks/forward_summary.csv；机器与案例规模会影响计时。
      </p>
    </section>
  );
}
