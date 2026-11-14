import { displayText } from "../display-text";
import { useEffect, useRef, useState } from "react";
import { ChartBar, DownloadSimple, Stop } from "@phosphor-icons/react";
import { request, downloadJson } from "../api";
import type { QueryInfo, Result } from "./types";
import { resultTitle, comparisonCompletion } from "./presentation";

const methods = [
  { id: "bfs", name: "BFS", note: "逐层扩展" },
  { id: "astar_fixed", name: "A* 固定", note: "人工启发函数" },
  { id: "astar_ga", name: "A* 遗传优化", note: "验证集优化参数" },
  { id: "mcts", name: "MCTS", note: "UCT 与路径模拟" },
];
type Row = { method: string; result?: Result; error?: string };

// The parent keys this component by question and budget. Unmounting cancels the
// sequence; a late response can never replace results for another configuration.
export default function CompareRuns({
  query,
  budget,
  disabled,
  onBusyChange,
}: {
  query: QueryInfo;
  budget: number;
  disabled: boolean;
  onBusyChange: (busy: boolean) => void;
}) {
  const [rows, setRows] = useState<Row[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const pending = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      pending.current?.abort();
      pending.current = null;
      onBusyChange(false);
    },
    [onBusyChange],
  );

  function cancel() {
    pending.current?.abort();
    pending.current = null;
    setBusy(false);
    onBusyChange(false);
    setMessage("对比已取消，保留已完成结果。后台当前任务可能仍在结束。");
  }

  async function compare() {
    if (pending.current || disabled) return;
    const controller = new AbortController();
    pending.current = controller;
    setRows([]);
    setBusy(true);
    onBusyChange(true);
    const completedRows: Row[] = [];
    try {
      // Respect the backend's bounded worker pool and keep per-method failures visible.
      for (const method of methods) {
        if (controller.signal.aborted) break;
        setMessage(`正在对比 ${method.name}，每种方法扩展预算 ${budget}。`);
        let row: Row;
        const unavailable = query.unavailable_methods?.[method.id];
        if (unavailable) {
          row = { method: method.id, error: unavailable };
        } else {
          try {
            const result = await request<Result>("/api/labs/lab03/run", {
              method: "POST",
              signal: controller.signal,
              timeoutMs: 25_000,
              body: JSON.stringify({
                query_id: query.id,
                method: method.id,
                budget,
                llm_mode: "replay",
              }),
            });
            row = { method: method.id, result };
          } catch (error) {
            if (controller.signal.aborted) return;
            row = {
              method: method.id,
              error:
                error instanceof Error ? error.message : "运行失败，请重试。",
            };
          }
        }
        if (controller.signal.aborted || pending.current !== controller) return;
        completedRows.push(row);
        setRows([...completedRows]);
      }
      if (!controller.signal.aborted)
        setMessage(
          comparisonCompletion(completedRows, methods.map((m) => m.id)).complete
            ? "四种算法均已返回结果。单题结果用于观察搜索行为，整体收益请查看评估分析。"
            : "对比已结束，但部分算法未能运行；错误不记作算法失败答案率，请修复后重试。",
        );
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setBusy(false);
        onBusyChange(false);
      }
    }
  }
  return (
    <section
      className="task-section lab3-comparison"
      aria-labelledby="lab3-comparison-heading"
    >
      <header className="lab3-section-heading">
        <div>
          <h2 id="lab3-comparison-heading">算法对比</h2>
        </div>
        <button
          className="button secondary"
          disabled={disabled || busy}
          onClick={() => void compare()}
        >
          <ChartBar size={18} />
          {busy ? "正在对比…" : rows.length ? "重新对比算法" : "对比四种算法"}
        </button>
      </header>
      <p className="helper-text">
        当前问题 {query.id} · 扩展预算 {budget} · 专家约束均启用
      </p>
      <div className="lab3-comparison-grid">
        {methods.map((method) => {
          const row = rows.find((r) => r.method === method.id);
          const value = row?.result?.result;
          return (
            <article key={method.id} data-valid={value?.valid}>
              <h3>{displayText(method.name)}</h3>
              <p>{displayText(method.note)}</p>
              <strong className="lab3-compare-value">
                {value?.expanded ?? "—"}
                <small>扩展节点</small>
              </strong>
              <dl>
                <div>
                  <dt>节点工作量</dt>
                  <dd>{value?.total_work ?? "—"}</dd>
                </div>
                <div>
                  <dt>启发边检查</dt>
                  <dd>{value?.heuristic_edge_checks ?? "—"}</dd>
                </div>
                <div>
                  <dt>候选检查 / 扫描</dt>
                  <dd>{value ? `${value.total_edge_checks} / ${value.adjacency_edge_scans}` : "—"}</dd>
                </div>
                <div>
                  <dt>复用 / 已证剪枝</dt>
                  <dd>{value ? `${value.successor_cache_hits} / ${value.proof_pruned}` : "—"}</dd>
                </div>
                <div>
                  <dt>模拟步数</dt>
                  <dd>{value?.rollout_steps ?? "—"}</dd>
                </div>
                <div>
                  <dt>有效路径跳数</dt>
                  <dd>{value?.valid ? value.hops : "—"}</dd>
                </div>
              </dl>
              <span className="lab3-compare-status">
                {value
                  ? resultTitle(value)
                  : row?.error
                    ? "运行不可用"
                    : busy
                      ? "等待结果"
                      : "尚未运行"}
              </span>
              {row?.error && <p className="lab3-rejection">{row.error}</p>}
            </article>
          );
        })}
      </div>
      <p className="helper-text">
        BFS / A* 按出队状态计费；MCTS
        按树扩展计费，模拟另计。两跳后缀最多探测2个子前缀；候选证据检查、全部出边扫描另列。相同扩展预算不代表相同总成本。
      </p>
      <p className="lab3-comparison-message" role="status">
        {displayText(message)}
      </p>
      <div className="lab3-run-actions">
        {busy && (
          <button className="button secondary" onClick={cancel}>
            <Stop size={16} />
            取消对比
          </button>
        )}
        {!!rows.length && !busy && (
          <button
            className="button secondary"
            onClick={() =>
              downloadJson(
                {
                  query,
                  budget,
                  methods: methods.map((m) => m.id),
                  ...comparisonCompletion(rows, methods.map((m) => m.id)),
                  note: "单题实测；MCTS模拟另计，不作为整体性能结论。",
                  rows,
                },
                `lab03-comparison-${query.id}-${budget}.json`,
              )
            }
          >
            <DownloadSimple size={16} />
            导出算法对比
          </button>
        )}
      </div>
    </section>
  );
}
