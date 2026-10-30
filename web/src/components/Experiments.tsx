import ExperimentBreadcrumb from "./ExperimentBreadcrumb";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from "recharts";
import {
  ChartBar,
  DownloadSimple,
  ArrowClockwise,
  Info,
} from "@phosphor-icons/react";
import PerformancePanel from "./PerformancePanel";
import AdditionalExperiments from "./AdditionalExperiments";
import type { ExperimentsData } from "../types";
import { downloadJson, request } from "../api";
const names: Record<string, string> = {
  rules_only: "结构规则",
  bayes_only: "Bayes评分器",
  fuzzy_only: "Mamdani评分器",
  hybrid_fixed_unlocked: "固定混合 · 不锁定",
  hybrid_fixed_locked: "固定混合 · 锁定",
  hybrid_tuned_unlocked: "验证权重 · 不锁定",
  hybrid_tuned_locked: "验证权重 · 锁定",
  fuzzy_learned: "Mamdani · 自动拟合",
  hybrid_adaptive_unlocked: "自动拟合 · 不锁定",
  hybrid_adaptive_locked: "自动拟合 · 硬锁定",
  hybrid_adaptive_calibrated: "自动拟合 · 校准复核",
};
const conditions: Record<string, string> = {
  clean: "清洁观测",
  value_noise: "数值噪声",
  edge_missing_020: "关系缺失20%",
  false_loop_020: "错误闭环20%",
  combined: "混合扰动",
  severe: "重度扰动",
};
const number = (v: unknown) => (typeof v === "number" ? v : null);
const percent = (v: unknown) =>
  typeof v === "number" ? (v * 100).toFixed(1) + "%" : "未记录";
export default function Experiments({
  embedded = false,
}: {
  embedded?: boolean;
}) {
  const Shell = embedded ? "section" : "main";
  const [data, setData] = useState<ExperimentsData | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [condition, setCondition] = useState("clean");
  const [section, setSection] = useState<
    "graph" | "deterministic" | "scoring" | "conflict"
  >("graph");
  const pending = useRef<AbortController | null>(null);
  const load = async () => {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true);
    setError("");
    try {
      const value = await request<ExperimentsData>("/api/experiments", {
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setData(value);
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : "结果读取失败");
    } finally {
      if (pending.current === controller) setBusy(false);
    }
  };
  useEffect(() => {
    void load();
    return () => pending.current?.abort();
  }, []);
  const rows = useMemo(
    () => data?.summary.filter((r) => r.scenario === condition) || [],
    [data, condition],
  );
  const chart = rows.map((r) => ({
    name: names[String(r.model)] || String(r.model),
    f1: number(r.f1_mean),
    recall: number(r.recall_mean),
    fpr: number(r.false_positive_rate_mean),
  }));
  const trend = useMemo(
    () =>
      Object.keys(conditions).map((c) => {
        const selected = data?.summary.filter((r) => r.scenario === c) || [];
        return {
          name: conditions[c],
          unlocked: number(
            selected.find((r) => r.model === "hybrid_fixed_unlocked")
              ?.constraint_violation_rate_mean,
          ),
          locked: number(
            selected.find((r) => r.model === "hybrid_fixed_locked")
              ?.constraint_violation_rate_mean,
          ),
        };
      }),
    [data],
  );
  const paired = data?.paired.find(
    (r) => r.scenario === condition && r.family === "fixed",
  );
  return (
    <Shell
      id={embedded ? undefined : "main-content"}
      tabIndex={-1}
      className="page-shell experiments-page"
    >
      {!embedded && (
        <ExperimentBreadcrumb moduleId="optimization" title="完整保存结果" />
      )}
      <header className="page-header">
        <div>
          {embedded ? <h2>正式实验对比结果</h2> : <h1>实验对比</h1>}
          <p>看效果，也看代价。这里读取已保存的图级实测结果。</p>
        </div>
        <div className="page-actions">
          <button
            className="button secondary"
            onClick={() => void load()}
            disabled={busy}
          >
            <ArrowClockwise size={17} />
            刷新结果
          </button>
          <button
            className="button secondary"
            onClick={() =>
              data && downloadJson(data, "equity-experiments.json")
            }
            disabled={!data}
          >
            <DownloadSimple size={17} />
            导出数据
          </button>
        </div>
      </header>
      {error && (
        <div role="alert" className="error-banner">
          {error}
          <button className="text-button" onClick={() => void load()}>
            重试
          </button>
        </div>
      )}
      {!data && busy ? (
        <div
          className="skeleton experiment-skeleton"
          aria-label="正在读取实验结果"
        />
      ) : !data?.available ? (
        <div className="empty-state large">
          <ChartBar size={42} />
          <h2>还没有图级实验结果</h2>
          <p>
            完整推理工作台仍可使用。生成实验后，网页会读取本地CSV，不会用虚构数字填充图表。
          </p>
          <code>python main.py e2e</code>
        </div>
      ) : (
        <>
          <nav className="method-tabs" aria-label="对比实验类型">
            {(
              [
                ["graph", "图级行动对比"],
                ["deterministic", "确定性优化"],
                ["scoring", "评分与参数优化"],
                ["conflict", "扩展：证据冲突"],
              ] as const
            ).map(([id, title]) => (
              <button
                key={id}
                className={section === id ? "active" : ""}
                aria-pressed={section === id}
                onClick={() => setSection(id)}
              >
                {title}
              </button>
            ))}
          </nav>
          {section === "graph" && (
            <>
              <div className="experiment-toolbar">
                <label>
                  <span>观测条件</span>
                  <select
                    aria-label="观测条件"
                    value={condition}
                    onChange={(e) => setCondition(e.target.value)}
                  >
                    {[
                      ...new Set(data.summary.map((r) => String(r.scenario))),
                    ].map((c) => (
                      <option key={c} value={c}>
                        {conditions[c] || c}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="experiment-source">
                  <strong>
                    {data.metadata.config?.seeds?.length || "多"} 个随机种子
                  </strong>
                  <span>
                    每个种子 {data.metadata.config?.test_size || "未记录"}{" "}
                    张测试图 · 同图配对对照
                  </span>
                </div>
                <span className="data-badge">本地实测 · 合成数据</span>
              </div>
              <div className="chart-grid">
                <section className="chart-section">
                  <header>
                    <h2>最终复核行动的效果</h2>
                    <p>F1与召回率来自最终行动，不是只对评分做分类。</p>
                  </header>
                  <div
                    className="chart-area"
                    tabIndex={0}
                    role="region"
                    aria-label="实验图表，小屏可横向滚动"
                    style={{ height: Math.max(340, chart.length * 48 + 80) }}
                  >
                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                      minWidth={480}
                    >
                      <BarChart
                        data={chart}
                        layout="vertical"
                        margin={{ left: 4, right: 28, top: 12, bottom: 4 }}
                      >
                        <CartesianGrid
                          horizontal={false}
                          stroke="var(--border)"
                        />
                        <XAxis
                          type="number"
                          domain={[0, 1]}
                          tickFormatter={(v) =>
                            (Number(v) * 100).toFixed(0) + "%"
                          }
                          tick={{ fontSize: 12, fill: "var(--muted)" }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          type="category"
                          dataKey="name"
                          width={145}
                          tick={{ fontSize: 12, fill: "var(--ink)" }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip
                          formatter={(v) => percent(v)}
                          contentStyle={{
                            background: "var(--bg)",
                            border: "1px solid var(--border)",
                            borderRadius: 8,
                            fontSize: 14,
                          }}
                        />
                        <Legend />
                        <Bar
                          isAnimationActive={false}
                          dataKey="f1"
                          name="F1"
                          fill="var(--primary)"
                          radius={[0, 3, 3, 0]}
                        />
                        <Bar
                          isAnimationActive={false}
                          dataKey="recall"
                          name="召回率"
                          fill="var(--chart-secondary)"
                          radius={[0, 3, 3, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
                <section className="chart-section">
                  <header>
                    <h2>结构约束有没有被忽略</h2>
                    <p>违反率为0是策略保证，不代表风险判断100%正确。</p>
                  </header>
                  <div
                    className="chart-area"
                    tabIndex={0}
                    role="region"
                    aria-label="实验图表，小屏可横向滚动"
                  >
                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                      minWidth={480}
                    >
                      <LineChart
                        data={trend}
                        margin={{ left: 0, right: 18, top: 15, bottom: 14 }}
                      >
                        <CartesianGrid
                          stroke="var(--border)"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="name"
                          tick={{ fontSize: 12, fill: "var(--muted)" }}
                          interval={0}
                          angle={-15}
                          height={48}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          domain={[0, 1]}
                          tickFormatter={(v) =>
                            (Number(v) * 100).toFixed(0) + "%"
                          }
                          tick={{ fontSize: 12, fill: "var(--muted)" }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip
                          formatter={(v) => percent(v)}
                          contentStyle={{
                            background: "var(--bg)",
                            border: "1px solid var(--border)",
                            borderRadius: 8,
                            fontSize: 14,
                          }}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="unlocked"
                          name="固定混合 · 不锁定"
                          stroke="var(--graph-guarantee)"
                          strokeWidth={2}
                          dot={{ r: 4 }}
                          isAnimationActive={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="locked"
                          name="固定混合 · 锁定"
                          stroke="var(--primary)"
                          strokeWidth={2}
                          dot={{ r: 4 }}
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>
              {paired && (
                <div className="comparison-note">
                  <Info size={20} />
                  <div>
                    <strong>锁定相对不锁定，在该条件下改变了什么？</strong>
                    <p>
                      每个种子平均增加 {Number(paired.delta_tp_mean).toFixed(2)}{" "}
                      个真阳性，同时增加{" "}
                      {Number(paired.delta_fp_mean).toFixed(2)}{" "}
                      个假阳性。F1变化为{" "}
                      {(Number(paired.delta_f1_mean) * 100).toFixed(2)}{" "}
                      个百分点。原始评分不被规则改写。
                    </p>
                  </div>
                </div>
              )}
              <section className="results-section">
                <header>
                  <h2>逐方法指标</h2>
                  <p>
                    均值跨种子汇总。指标为不同目标提供信息，不能只挑最好看的一列。
                  </p>
                </header>
                <div
                  className="table-scroll"
                  role="region"
                  aria-label="各方法的实测指标"
                  tabIndex={0}
                >
                  <table>
                    <thead>
                      <tr>
                        <th>方法</th>
                        <th>准确率</th>
                        <th>F1</th>
                        <th>召回率</th>
                        <th>误报率</th>
                        <th>高预警违反率</th>
                        <th>评分损失↓</th>
                        <th>端到端 / ms</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr
                          key={String(r.model)}
                          className={
                            r.model === "hybrid_fixed_locked"
                              ? "highlight-row"
                              : ""
                          }
                        >
                          <th>{names[String(r.model)] || String(r.model)}</th>
                          <td>{percent(r.accuracy_mean)}</td>
                          <td>
                            {typeof r.f1_mean === "number"
                              ? r.f1_mean.toFixed(3)
                              : "未记录"}
                          </td>
                          <td>{percent(r.recall_mean)}</td>
                          <td>{percent(r.false_positive_rate_mean)}</td>
                          <td>{percent(r.constraint_violation_rate_mean)}</td>
                          <td>
                            {typeof r.score_brier_mean === "number"
                              ? r.score_brier_mean.toFixed(4)
                              : "未记录"}
                          </td>
                          <td>
                            {typeof r.end_to_end_ms_mean === "number"
                              ? r.end_to_end_ms_mean.toFixed(2)
                              : "未记录"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}
          {section === "deterministic" && (
            <PerformancePanel rows={data.benchmark} />
          )}
          {section !== "graph" && (
            <AdditionalExperiments data={data} section={section} />
          )}
          <p className="method-note">
            {data.note}{" "}
            Bayes与Mamdani评分器也共用规则提取的结构指标；此处不声称比较了完全不使用符号预处理的系统。
          </p>
        </>
      )}
    </Shell>
  );
}
