import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Play,
  Pause,
  ArrowClockwise,
  DownloadSimple,
  ShieldCheck,
  TreeStructure,
  CaretLeft,
  CaretRight,
  SkipBack,
  SkipForward,
  CheckCircle,
  WarningCircle,
  Stop,
} from "@phosphor-icons/react";
import { request, downloadJson } from "../api";
import type { CourseLab } from "../types";
import { labHref } from "../routes";
import type { GA, Result, Snapshot } from "../lab-three/types";
import {
  percent,
  relations,
  statuses,
  resultTitle,
  blockedReason,
} from "../lab-three/presentation";
import SearchTree from "../lab-three/SearchTree";

function Curve({ ga }: { ga: GA }) {
  const values = ga.history;
  if (!values.length)
    return (
      <p className="helper-text">尚无可展示的迭代记录，请重新生成实验数据。</p>
    );
  const low = Math.min(...values.map((v) => v.fitness)) - 0.01;
  const high = Math.max(...values.map((v) => v.fitness)) + 0.01;
  const points = values
    .map(
      (v, i) =>
        `${44 + i * (660 / Math.max(1, values.length - 1))},${160 - ((v.fitness - low) / (high - low)) * 120}`,
    )
    .join(" ");
  return (
    <svg
      className="lab3-curve"
      viewBox="0 0 760 210"
      role="img"
      aria-label="遗传算法验证集适应度曲线"
    >
      <path d="M44 25V165H715" fill="none" stroke="var(--border)" />
      <polyline
        points={points}
        stroke="var(--primary)"
        strokeWidth="3"
        fill="none"
      />
      {values.map((v, i) => (
        <g key={i}>
          <circle
            cx={44 + i * (660 / Math.max(1, values.length - 1))}
            cy={160 - ((v.fitness - low) / (high - low)) * 120}
            r="3"
            fill="var(--primary)"
          />
          <text
            x={44 + i * (660 / Math.max(1, values.length - 1))}
            y="187"
            textAnchor="middle"
            fontSize="12"
            fill="var(--muted)"
          >
            {v.generation}
          </text>
        </g>
      ))}
      <text x="44" y="17" fontSize="12" fill="var(--muted)">
        验证适应度 {values[0].fitness.toFixed(4)} →{" "}
        {values.at(-1)!.fitness.toFixed(4)}
      </text>
      <text x="715" y="205" textAnchor="end" fontSize="12" fill="var(--muted)">
        迭代代数
      </text>
    </svg>
  );
}

export default function LabThree({
  lab,
  moduleId,
  active = true,
}: {
  lab: CourseLab;
  moduleId: string;
  active?: boolean;
}) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [loadError, setLoadError] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [queryId, setQueryId] = useState("shared-3");
  const [method, setMethod] = useState("hybrid");
  const [budget, setBudget] = useState(16);
  const [llmMode, setLlmMode] = useState("replay");
  const [result, setResult] = useState<Result | null>(null);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600);
  const [announcement, setAnnouncement] = useState("");
  const [split, setSplit] = useState("test");
  const [tableBudget, setTableBudget] = useState(8);
  const [gaSeed, setGaSeed] = useState("7");
  const pending = useRef<AbortController | null>(null);
  const pendingLoad = useRef<AbortController | null>(null);
  const snapshotRef = useRef<Snapshot | null>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const module = lab.modules.find((m) => m.id === moduleId)!;
  const selected = snapshot?.queries.find((q) => q.id === queryId);
  const blocked = snapshot
    ? blockedReason(snapshot, selected, method, llmMode)
    : "实验数据尚未就绪";

  const load = useCallback(async () => {
    pendingLoad.current?.abort();
    const controller = new AbortController();
    pendingLoad.current = controller;
    setLoading(true);
    setLoadError("");
    try {
      const value = await request<Snapshot>("/api/labs/lab03/snapshot", {
        signal: controller.signal,
        timeoutMs: 15_000,
      });
      if (controller.signal.aborted || pendingLoad.current !== controller)
        return;
      snapshotRef.current = value;
      setSnapshot(value);
      setQueryId((id) =>
        value.queries.some((q) => q.id === id)
          ? id
          : value.queries[0]?.id || "",
      );
      setGaSeed((seed) =>
        value.ga[seed] ? seed : Object.keys(value.ga)[0] || "7",
      );
    } catch (e) {
      if (!controller.signal.aborted)
        setLoadError(
          e instanceof Error ? e.message : "实验数据读取失败，请重试。",
        );
    } finally {
      if (pendingLoad.current === controller) {
        pendingLoad.current = null;
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    if (active && !snapshotRef.current) void load();
    if (!active) {
      setBusy(false);
      setLoading(false);
      setPlaying(false);
    }
    return () => {
      pending.current?.abort();
      pending.current = null;
      pendingLoad.current?.abort();
      pendingLoad.current = null;
    };
  }, [active, load]);

  useEffect(() => {
    if (!playing || !result || !active) return;
    if (step >= result.result.trace.length - 1) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => setStep(step + 1), speed);
    return () => clearTimeout(timer);
  }, [playing, result, active, speed, step]);

  useEffect(() => {
    if (!result || !active) return;
    const frame = requestAnimationFrame(() =>
      resultHeading.current?.focus({ preventScroll: true }),
    );
    return () => cancelAnimationFrame(frame);
  }, [result, active]);

  function reset() {
    pending.current?.abort();
    pending.current = null;
    setBusy(false);
    setResult(null);
    setPlaying(false);
    setStep(0);
    setError("");
    setAnnouncement("");
  }

  function cancel() {
    pending.current?.abort();
    pending.current = null;
    setBusy(false);
    setError("");
    setPlaying(false);
    setAnnouncement(
      "已停止等待本次结果。后台任务可能仍在结束，稍后可重新运行。",
    );
  }

  async function run(nextBudget = budget) {
    if (pending.current || blocked) return;
    const controller = new AbortController();
    pending.current = controller;
    setBudget(nextBudget);
    setBusy(true);
    setError("");
    setResult(null);
    setPlaying(false);
    setStep(0);
    setAnnouncement(
      method === "hybrid_llm" && llmMode === "remote"
        ? "正在调用远程模型，最长等待约3分钟。"
        : "正在运行路径搜索。",
    );
    try {
      const value = await request<Result>("/api/labs/lab03/run", {
        method: "POST",
        body: JSON.stringify({
          query_id: queryId,
          method,
          budget: nextBudget,
          llm_mode: llmMode,
        }),
        signal: controller.signal,
        timeoutMs:
          method === "hybrid_llm" && llmMode === "remote" ? 190_000 : 25_000,
      });
      if (!controller.signal.aborted && pending.current === controller) {
        setResult(value);
        setStep(Math.max(0, value.result.trace.length - 1));
        setAnnouncement(
          resultTitle(value.result) +
            "。扩展 " +
            value.result.expanded +
            " 个节点。",
        );
      }
    } catch (e) {
      if (!controller.signal.aborted) {
        setError(e instanceof Error ? e.message : "搜索未完成，请重试。");
        setAnnouncement("");
      }
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setBusy(false);
      }
    }
  }
  function goStep(index: number) {
    setPlaying(false);
    setStep(index);
  }
  if (!active) return null;
  const metrics =
    snapshot?.summary.filter(
      (r) => r.split === split && r.budget === tableBudget,
    ) || [];
  const paired =
    snapshot?.paired?.filter(
      (r) => r.split === split && r.budget === tableBudget,
    ) || [];
  const current = result?.result.trace[step];
  const lastStep = Math.max(0, (result?.result.trace.length || 1) - 1);
  const currentName = result?.graph.nodes.find(
    (n) => n.id === current?.node,
  )?.name;
  const methodName =
    result?.result.label ||
    snapshot?.methods.find((m) => m.id === method)?.name ||
    method;
  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="page-shell course-shell task-page lab3-page"
    >
      <nav className="course-breadcrumb" aria-label="当前位置">
        <a href={labHref("lab03")}>
          <ArrowLeft size={16} />
          实验三
        </a>
        <span>/</span>
        <span>{module.title}</span>
      </nav>
      <header className="course-header">
        <div>
          <h1>{module.title}</h1>
          <p>{module.description}</p>
        </div>
      </header>
      <div className="task-requirement">
        <strong>对应实验要求</strong>
        <p>{module.requirement}</p>
      </div>
      <nav className="lab3-module-nav" aria-label="实验三模块">
        {lab.modules.map((item) => (
          <a
            key={item.id}
            href={labHref("lab03", item.id)}
            aria-current={item.id === moduleId ? "page" : undefined}
          >
            {item.title}
          </a>
        ))}
      </nav>
      <div
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {announcement}
      </div>
      {loadError && (
        <div className="error-banner" role="alert">
          <span>{loadError}</span>
          <button
            className="button secondary"
            disabled={loading}
            onClick={() => void load()}
          >
            <ArrowClockwise size={16} />
            {loading ? "正在加载…" : "重新加载"}
          </button>
        </div>
      )}
      {!snapshot ? (
        loading ? (
          <div
            className="skeleton course-skeleton"
            role="status"
            aria-label="正在加载实验三"
          />
        ) : (
          <div className="task-empty">
            <p>
              实验数据未能加载。请检查本地服务后重新加载；这不会修改已保存的实验结果。
            </p>
          </div>
        )
      ) : (
        <>
          {snapshot.warnings?.length > 0 && (
            <aside className="lab3-notice warning" aria-label="实验数据提醒">
              <WarningCircle size={20} />
              <div>
                <strong>部分实验数据不可用</strong>
                <ul>
                  {snapshot.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
                <button
                  className="text-button"
                  onClick={() => void load()}
                  disabled={loading}
                >
                  重新检查数据
                </button>
              </div>
            </aside>
          )}
          {moduleId === "search" && (
            <>
              <section className="task-section">
                <header>
                  <h2>从问题出发，逐步找到证据</h2>
                  <p>
                    起点、关系序列和终点属性定义问题。搜索程序只沿图中已有的边前进。
                  </p>
                </header>
                <form
                  className="lab3-search-form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void run();
                  }}
                  aria-busy={busy}
                >
                  <label className="lab3-query-field">
                    多跳问题
                    <select
                      aria-label="实验三问题"
                      value={queryId}
                      onChange={(e) => {
                        reset();
                        setQueryId(e.target.value);
                      }}
                      disabled={busy}
                    >
                      {snapshot.queries.map((q) => (
                        <option key={q.id} value={q.id}>
                          {q.split === "shared" ? "共享场景" : "测试图"} ·{" "}
                          {q.hops}跳{q.llm_cached ? " · LLM实测" : ""} ·{" "}
                          {q.text}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    搜索方法
                    <select
                      aria-label="实验三搜索方法"
                      value={method}
                      onChange={(e) => {
                        reset();
                        setMethod(e.target.value);
                      }}
                      disabled={busy}
                    >
                      {snapshot.methods.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                          {selected?.unavailable_methods?.[m.id]
                            ? "（数据不可用）"
                            : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    扩展预算
                    <select
                      aria-label="实验三扩展预算"
                      value={budget}
                      onChange={(e) => {
                        reset();
                        setBudget(Number(e.target.value));
                      }}
                      disabled={busy}
                    >
                      {[4, 8, 16, 32, 64, 128].map((n) => (
                        <option key={n} value={n}>
                          {n} 个节点
                        </option>
                      ))}
                    </select>
                  </label>
                  {method === "hybrid_llm" && (
                    <label>
                      LLM来源
                      <select
                        aria-label="实验三LLM来源"
                        value={llmMode}
                        disabled={busy}
                        onChange={(e) => {
                          reset();
                          setLlmMode(e.target.value);
                        }}
                      >
                        <option value="replay">重放真实评分记录</option>
                        <option value="remote">实时远程接口（需配置）</option>
                      </select>
                    </label>
                  )}
                  <div className="lab3-run-actions">
                    <button
                      className="button primary"
                      type="submit"
                      disabled={busy || Boolean(blocked)}
                      aria-describedby={
                        blocked ? "lab3-run-blocked" : undefined
                      }
                    >
                      <Play size={17} />
                      {busy ? "正在搜索…" : "运行路径搜索"}
                    </button>
                    {busy && (
                      <button
                        type="button"
                        className="button secondary"
                        onClick={cancel}
                      >
                        <Stop size={16} />
                        停止等待
                      </button>
                    )}
                  </div>
                </form>
                {selected && (
                  <div className="lab3-query-summary">
                    <p>{selected.text}</p>
                    <span>
                      {selected.nodes} 个实体 · {selected.edges} 条关系 ·{" "}
                      {selected.hops} 跳 ·{" "}
                      {selected.split === "shared"
                        ? "共享原始CSV场景"
                        : "图级隔离的教学测试图"}
                    </span>
                  </div>
                )}
                <p className="helper-text lab3-budget-note">
                  {method === "mcts"
                    ? "预算只限制MCTS树扩展；模拟步数另计，不与BFS/A*直接等量。"
                    : "每次从队列取出一个状态计1次扩展，包含目标状态；预算不足不表示无解。"}
                </p>
                {method === "hybrid_no_expert" && (
                  <p className="lab3-notice warning">
                    <WarningCircle size={18} />
                    当前为无专家约束的消融实验，可能返回证据无效的候选路径，不能当作有效答案。
                  </p>
                )}
                {method === "hybrid_llm" && (
                  <p className="helper-text">
                    {llmMode === "replay"
                      ? "重放真实小模型已保存的评分，不会再次调用模型。仅前两层最多8条边有实测评分，其余使用中性值。"
                      : "实时模式只评估起点前6条边，其余使用中性值。与离线实验的评分范围不同，最长等待约3分钟。"}
                  </p>
                )}
                {blocked && (
                  <div className="lab3-notice warning" id="lab3-run-blocked">
                    <WarningCircle size={18} />
                    <div>
                      <p>{blocked}</p>
                      {method === "hybrid_llm" &&
                        llmMode === "replay" &&
                        snapshot.queries.some((q) => q.llm_cached) && (
                          <button
                            className="text-button"
                            onClick={() => {
                              reset();
                              setQueryId(
                                snapshot.queries.find((q) => q.llm_cached)!.id,
                              );
                            }}
                          >
                            切换到有实测记录的问题
                          </button>
                        )}
                    </div>
                  </div>
                )}
                {error && (
                  <div className="error-banner" role="alert">
                    <span>{error}</span>
                    <button
                      className="button secondary"
                      disabled={busy || Boolean(blocked)}
                      onClick={() => void run()}
                    >
                      <ArrowClockwise size={16} />
                      重新运行
                    </button>
                  </div>
                )}
                {!busy && announcement.startsWith("已停止") && (
                  <p className="helper-text">{announcement}</p>
                )}
                {!result && !busy && (
                  <div className="task-empty">
                    <TreeStructure size={24} />
                    <p>
                      先选择问题和算法。运行后可查看答案路径、每一次节点扩展及专家规则拦截。
                    </p>
                  </div>
                )}
                {busy && (
                  <div
                    className="skeleton lab3-result-skeleton"
                    aria-label="正在执行搜索"
                  />
                )}
                {result && (
                  <section
                    className="task-result"
                    aria-labelledby="lab3-result-heading"
                  >
                    <div className="task-result-title">
                      <h2
                        id="lab3-result-heading"
                        ref={resultHeading}
                        tabIndex={-1}
                        className="lab3-result-heading"
                      >
                        {result.result.valid ? (
                          <CheckCircle size={22} />
                        ) : (
                          <WarningCircle size={22} />
                        )}
                        {resultTitle(result.result)}
                      </h2>
                      <button
                        className="button secondary"
                        onClick={() =>
                          downloadJson(
                            result,
                            "lab03-" +
                              result.query.id +
                              "-" +
                              result.result.method +
                              "-" +
                              result.result.budget +
                              ".json",
                          )
                        }
                      >
                        <DownloadSimple size={16} />
                        导出本次结果
                      </button>
                    </div>
                    <p className="lab3-result-meta">
                      {result.result.label} · 扩展预算 {result.result.budget} ·{" "}
                      {result.query.id}
                    </p>
                    <p
                      className={
                        result.result.valid
                          ? "lab3-path-answer"
                          : "lab3-result-explanation"
                      }
                    >
                      {result.explanation}
                    </p>
                    {!result.result.valid && result.result.found && (
                      <p className="lab3-rejection">
                        候选路径：
                        {result.result.path
                          .map(
                            (id) =>
                              result.graph.nodes.find((n) => n.id === id)
                                ?.name || id,
                          )
                          .join(" → ")}
                      </p>
                    )}
                    {result.result.status === "budget_exhausted" &&
                      result.result.budget < 128 && (
                        <button
                          className="button secondary"
                          onClick={() =>
                            void run(Math.min(128, result.result.budget * 2))
                          }
                        >
                          增加到 {Math.min(128, result.result.budget * 2)}{" "}
                          个节点并重试
                          <ArrowRight size={16} />
                        </button>
                      )}
                    <div className="lab3-metrics">
                      <div>
                        <strong>{result.result.expanded}</strong>
                        <span>扩展节点</span>
                      </div>
                      <div>
                        <strong>{result.result.total_work}</strong>
                        <span>总工作量（含模拟）</span>
                      </div>
                      <div>
                        <strong>{result.result.hops ?? "—"}</strong>
                        <span>答案路径跳数</span>
                      </div>
                      <div>
                        <strong>{result.result.expert_rejections}</strong>
                        <span>证据规则拦截</span>
                      </div>
                    </div>
                    {result.llm.mode !== "off" && (
                      <p className="lab3-notice">
                        <ShieldCheck size={18} />
                        {result.llm.mode === "replay"
                          ? "本次重放已保存的真实LLM评分，没有再次调用模型。"
                          : "本次已调用远程LLM评估候选边。"}
                        {result.llm.scored_edges}{" "}
                        条边有评分。评分不替代图谱证据。
                      </p>
                    )}
                    {result.result.trace.length > 0 ? (
                      <>
                        <div className="lab3-playback">
                          <h3>搜索过程</h3>
                          <div
                            className="lab3-play-controls"
                            role="group"
                            aria-label="轨迹播放控制"
                          >
                            <button
                              className="icon-button"
                              aria-label="第一步"
                              disabled={step === 0}
                              onClick={() => goStep(0)}
                            >
                              <SkipBack size={17} />
                            </button>
                            <button
                              className="icon-button"
                              aria-label="上一步"
                              disabled={step === 0}
                              onClick={() => goStep(step - 1)}
                            >
                              <CaretLeft size={17} />
                            </button>
                            <button
                              className="button secondary compact"
                              disabled={lastStep === 0}
                              onClick={() => {
                                if (step === lastStep) setStep(0);
                                setPlaying(!playing);
                              }}
                            >
                              {playing ? (
                                <Pause size={16} />
                              ) : (
                                <Play size={16} />
                              )}
                              {playing ? "暂停" : "播放轨迹"}
                            </button>
                            <button
                              className="icon-button"
                              aria-label="下一步"
                              disabled={step === lastStep}
                              onClick={() => goStep(step + 1)}
                            >
                              <CaretRight size={17} />
                            </button>
                            <button
                              className="icon-button"
                              aria-label="最后一步"
                              disabled={step === lastStep}
                              onClick={() => goStep(lastStep)}
                            >
                              <SkipForward size={17} />
                            </button>
                          </div>
                          <label className="lab3-speed">
                            播放速度
                            <select
                              aria-label="轨迹播放速度"
                              value={speed}
                              onChange={(e) => setSpeed(Number(e.target.value))}
                            >
                              <option value={1200}>0.5×</option>
                              <option value={600}>1×</option>
                              <option value={300}>2×</option>
                            </select>
                          </label>
                        </div>
                        <div className="lab3-timeline">
                          <input
                            aria-label="搜索轨迹步骤"
                            type="range"
                            min={0}
                            max={lastStep}
                            value={step}
                            aria-valuetext={
                              "第" +
                              (step + 1) +
                              "步，" +
                              (statuses[current?.status || ""] || "") +
                              "，" +
                              (currentName || "")
                            }
                            onChange={(e) => goStep(Number(e.target.value))}
                          />
                          <output>
                            {step + 1} / {result.result.trace.length}
                          </output>
                        </div>
                        <div className="lab3-current-step">
                          <span>
                            {statuses[current?.status || ""] || "当前状态"}
                          </span>
                          <strong>{currentName}</strong>
                          <small>
                            深度 g={current?.g}
                            {current?.h != null
                              ? " · h=" +
                                current.h.toFixed(2) +
                                " · f=" +
                                current.f?.toFixed(2)
                              : methodName.includes("MCTS")
                                ? " · UCT选择 / 路径模拟（不使用h）"
                                : " · 按层扩展（不使用h）"}
                            {current?.p != null
                              ? " · 学习概率 " + percent(current.p)
                              : ""}
                          </small>
                          {current?.reason && (
                            <p className="lab3-rejection">{current.reason}</p>
                          )}
                        </div>
                        <SearchTree result={result} index={step} />
                        {result.result.trace_truncated && (
                          <p className="lab3-notice warning">
                            本次产生 {result.result.trace_events}{" "}
                            条事件，受记录上限限制，只保留{" "}
                            {result.result.trace.length}{" "}
                            条（含最终路径）。导出也仅包含已记录事件，指标仍统计完整运行。
                          </p>
                        )}
                      </>
                    ) : (
                      <p className="helper-text">
                        本次运行未记录轨迹，仍可核对下方路径证据与搜索指标。
                      </p>
                    )}
                    {result.path_evidence.length > 0 && (
                      <div
                        className="table-scroll"
                        role="region"
                        aria-label="实验三路径证据，可横向滚动"
                        tabIndex={0}
                      >
                        <table aria-label="实验三答案证据">
                          <caption>
                            {result.result.valid
                              ? "已核验的答案路径证据"
                              : "未通过核验的候选路径证据"}
                          </caption>
                          <thead>
                            <tr>
                              <th scope="col">关系</th>
                              <th scope="col">来源 → 目标</th>
                              <th scope="col">置信度</th>
                              <th scope="col">证据状态</th>
                            </tr>
                          </thead>
                          <tbody>
                            {result.path_evidence.map((e) => (
                              <tr key={e.id}>
                                <th scope="row">{relations[e.relation]}</th>
                                <td>
                                  {
                                    result.graph.nodes.find(
                                      (n) => n.id === e.source,
                                    )?.name
                                  }{" "}
                                  →{" "}
                                  {
                                    result.graph.nodes.find(
                                      (n) => n.id === e.target,
                                    )?.name
                                  }
                                </td>
                                <td>{percent(e.confidence)}</td>
                                <td>
                                  <span
                                    className={
                                      e.eligible
                                        ? "lab3-evidence-ok"
                                        : "lab3-rejection"
                                    }
                                  >
                                    {e.eligible
                                      ? "已核验 · 有效 · 置信度达标"
                                      : e.rejection_reason || "证据未通过"}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    <details className="task-details">
                      <summary>算法语义与解释边界</summary>
                      <p>{result.result.note}</p>
                      <p>
                        这里检索的是关系路径；持股路径不自动等于控制关系，担保或交易线索也不是违法判断。
                      </p>
                    </details>
                  </section>
                )}
              </section>
              <section className="task-section">
                <h2>同一问题，三种搜索方式</h2>
                <div className="lab3-method-notes">
                  <div>
                    <h3>BFS</h3>
                    <p>按深度逐层扩展，作为盲目搜索基线。</p>
                  </div>
                  <div>
                    <h3>A*</h3>
                    <p>
                      以 f=g+h
                      排序。下界版用于校验；固定、GA和学习版比较引导效率。
                    </p>
                  </div>
                  <div>
                    <h3>MCTS</h3>
                    <p>
                      UCT选择、节点扩展、路径模拟和价值回传。额外模拟开销单独计数。
                    </p>
                  </div>
                </div>
              </section>
            </>
          )}
          {moduleId === "optimization" && (
            <>
              <section className="task-section">
                <header>
                  <h2>用验证图学习更好的搜索顺序</h2>
                  <p>
                    适应度 = 有效答案率 − 0.12 ×
                    平均扩展预算占比。测试集不参与参数选择。
                  </p>
                </header>
                <label className="task-inline-field">
                  随机种子
                  <select
                    aria-label="遗传优化随机种子"
                    disabled={Object.keys(snapshot.ga).length === 0}
                    value={gaSeed}
                    onChange={(e) => setGaSeed(e.target.value)}
                  >
                    {Object.keys(snapshot.ga).map((s) => (
                      <option key={s}>{s}</option>
                    ))}
                  </select>
                </label>
                {!snapshot.ga[gaSeed] && (
                  <p className="task-empty">
                    没有可用的GA优化记录。重新生成实验数据后，可在这里查看适应度曲线与参数。
                  </p>
                )}
                {snapshot.ga[gaSeed] && (
                  <>
                    <div
                      className="lab3-curve-scroll"
                      tabIndex={0}
                      role="region"
                      aria-label="GA适应度曲线，可横向滚动"
                    >
                      <Curve ga={snapshot.ga[gaSeed]} />
                    </div>
                    <details className="task-details">
                      <summary>查看每一代的真实测量值</summary>
                      <div
                        className="table-scroll"
                        tabIndex={0}
                        role="region"
                        aria-label="GA各代测量值，可横向滚动"
                      >
                        <table aria-label="GA逐代测量">
                          <thead>
                            <tr>
                              <th scope="col">代数</th>
                              <th scope="col">验证适应度</th>
                              <th scope="col">有效答案率</th>
                              <th scope="col">平均预算占用</th>
                            </tr>
                          </thead>
                          <tbody>
                            {snapshot.ga[gaSeed].history.map((row) => (
                              <tr key={row.generation}>
                                <th scope="row">{row.generation}</th>
                                <td>{row.fitness.toFixed(4)}</td>
                                <td>{percent(row.success_rate)}</td>
                                <td>{percent(row.expanded_fraction)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </details>
                    <p className="helper-text">
                      种群 {snapshot.ga[gaSeed].population} · 迭代{" "}
                      {snapshot.ga[gaSeed].generations} 代 ·{" "}
                      {snapshot.ga[gaSeed].validation_queries} 道验证问题 ·
                      精英保留 / 锦标赛选择 / 算术交叉 / 高斯变异
                    </p>
                    <div className="lab3-weights">
                      {[
                        "主题差异",
                        "后继稀疏",
                        "证据风险",
                        "终点类型",
                        "死路惩罚",
                      ].map((n, i) => (
                        <div key={n}>
                          <span>{n}</span>
                          <strong>
                            {snapshot.ga[gaSeed].weights[i].toFixed(3)}
                          </strong>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </section>
              <section className="task-section">
                <h2>路径价值分类器与专家约束</h2>
                <p>
                  从训练图的成功/失败路径前缀学习逻辑回归分类器。输入是剩余跳数、主题、后继、证据质量等6个特征；不包含答案编号。
                </p>
                {snapshot.learning.length === 0 && (
                  <p className="task-empty">
                    尚无分类器测试记录；未测量不能显示为零分。
                  </p>
                )}
                <div
                  className="table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="分类器测试数据，可横向滚动"
                >
                  <table aria-label="路径分类器测试结果">
                    <thead>
                      <tr>
                        <th scope="col">种子</th>
                        <th scope="col">测试路径前缀</th>
                        <th scope="col">F1</th>
                        <th scope="col">Brier</th>
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.learning.map((r) => (
                        <tr key={r.seed}>
                          <th scope="row">{r.seed}</th>
                          <td>{r.samples}</td>
                          <td>{r.f1.toFixed(4)}</td>
                          <td>{r.brier.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="lab3-notice">
                  <ShieldCheck size={18} />
                  未经核验、已失效或置信度低于0.55的关系会被专家规则拦截。关闭专家约束的消融结果见对比实验。
                </p>
              </section>
            </>
          )}
          {moduleId === "assistant" && (
            <>
              <section className="task-section">
                <h2>模型评估行动，搜索程序产生答案</h2>
                <p>
                  LLM只接收问题的关系序列、目标属性和候选关系的证据字段，输出行动评分。它没有修改图或直接填入答案的权限。
                </p>
                <div className="lab3-method-notes">
                  <div>
                    <h3>① 合法性辅助</h3>
                    <p>模型评价候选边；最终是否可走由确定性证据规则裁决。</p>
                  </div>
                  <div>
                    <h3>② 路径价值</h3>
                    <p>
                      限制为 A / B / C
                      三档下一token概率，转成排序奖励，避免自由文本夹带答案。
                    </p>
                  </div>
                  <div>
                    <h3>③ 路径解释</h3>
                    <p>
                      搜索结束后再解释已经核验的路径，生成文字不参与答案判定。
                    </p>
                  </div>
                </div>
                {snapshot.llm_summary.model ? (
                  <>
                    <p className="lab3-notice">
                      实测模型：{snapshot.llm_summary.model} ·{" "}
                      {snapshot.llm_summary.questions} 道独立测试问题 ·
                      本地推断记录可重放
                    </p>
                    <p>{snapshot.llm_summary.note}</p>
                    {Boolean(snapshot.llm_summary.results?.length) && (
                      <>
                        <div
                          className="table-scroll"
                          tabIndex={0}
                          role="region"
                          aria-label="LLM配对结果，可横向滚动"
                        >
                          <table aria-label="LLM辅助前后配对比较">
                            <thead>
                              <tr>
                                <th scope="col">方法</th>
                                <th scope="col">有效答案率</th>
                                <th scope="col">扩展节点</th>
                                <th scope="col">含模拟工作量</th>
                              </tr>
                            </thead>
                            <tbody>
                              {snapshot.llm_summary.results!.map((row) => (
                                <tr key={row.method}>
                                  <th scope="row">{row.label}</th>
                                  <td>{percent(row.success_rate)}</td>
                                  <td>{row.mean_expanded.toFixed(2)}</td>
                                  <td>{row.mean_total_work.toFixed(2)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        <p className="helper-text">
                          同一批 {snapshot.llm_summary.questions}{" "}
                          题、扩展预算8。LLM评分推断另耗时{" "}
                          {(
                            (snapshot.llm_summary.total_inference_ms || 0) /
                            1000
                          ).toFixed(1)}{" "}
                          秒（合计，不含模型加载），未计入搜索扩展。小样本机制验证，不代表普遍收益。
                        </p>
                      </>
                    )}
                    <p>
                      合法性检查 {snapshot.llm_summary.legal_checks} 条，一致率{" "}
                      {snapshot.llm_summary.legal_check_accuracy == null
                        ? "尚未测量"
                        : percent(snapshot.llm_summary.legal_check_accuracy)}
                      ；{snapshot.llm_summary.explanations} 段解释中，
                      {snapshot.llm_summary.explanation_grounded}{" "}
                      段包含全部路径实体。
                    </p>
                    <a
                      className="button primary"
                      href={labHref("lab03", "search")}
                    >
                      打开搜索工作台
                      <ArrowRight size={17} />
                    </a>
                  </>
                ) : (
                  <p className="helper-text">
                    尚未生成LLM实测记录。核心搜索可运行；请运行实验三llm命令后刷新。
                  </p>
                )}
              </section>
              <section className="task-section">
                <h2>实测解释与失败边界</h2>
                {snapshot.llm_explanations.length === 0 && (
                  <p className="task-empty">
                    尚无模型解释记录。搜索得到的路径证据仍可独立核验。
                  </p>
                )}
                {snapshot.llm_explanations.map((e) => (
                  <details className="task-details" key={e.query_id}>
                    <summary>{e.query_id} · 查看模型原始解释</summary>
                    <p className="lab3-notice">
                      {e.grounded
                        ? "实体名称检查通过；这不等于整段语义已核验。"
                        : "实体核对未通过：" +
                          e.missing_path_entities.join("、") +
                          " 未在解释中完整出现。请以程序证据路径为准。"}
                    </p>
                    <p>{e.text}</p>
                    <small>{e.note}</small>
                  </details>
                ))}
                <p>
                  额外注入错误评分的压力测试中，专家约束下无效答案数：
                  <strong>
                    {snapshot.stress.invalid_answers ?? "尚未测试"}
                  </strong>
                  。这是受控故障注入，与模型真实输出分开记录。
                </p>
                <details className="task-details">
                  <summary>接入自己的远程模型</summary>
                  <p>
                    在启动服务前配置
                    LAB03_LLM_BASE_URL、LAB03_LLM_API_KEY、LAB03_LLM_MODEL，再在工作台选择“实时远程接口”。凭据仅从服务环境读取，不写入结果或提交材料。
                  </p>
                </details>
              </section>
            </>
          )}
          {moduleId === "evaluation" && (
            <>
              <section className="task-section">
                <header>
                  <h2>比较成功率，也比较真实搜索开销</h2>
                  <p>
                    {snapshot.manifest.runs == null
                      ? "尚无基准运行记录"
                      : snapshot.manifest.runs.toLocaleString() +
                        " 次独立运行"}{" "}
                    · 3个随机种子 · 2—4跳 · 图级训练/验证/测试隔离
                  </p>
                </header>
                <div className="lab3-filters">
                  <label>
                    评估场景
                    <select
                      aria-label="实验三评估场景"
                      value={split}
                      onChange={(e) => setSplit(e.target.value)}
                    >
                      <option value="test">独立测试图</option>
                      <option value="ood">分布变化图</option>
                    </select>
                  </label>
                  <label>
                    扩展预算
                    <select
                      aria-label="实验三对比预算"
                      value={tableBudget}
                      onChange={(e) => setTableBudget(Number(e.target.value))}
                    >
                      {[4, 8, 16].map((v) => (
                        <option key={v} value={v}>
                          {v}节点
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                {metrics.length === 0 && (
                  <p className="task-empty">
                    当前场景与预算暂无已保存的测量结果。请重新生成基准数据；未测量不等于成功率为零。
                  </p>
                )}
                <p className="helper-text">
                  有效答案率 · {metrics[0]?.n ?? 0} 道配对问题 / 方法 · 越高越好
                </p>
                <div className="lab3-bars">
                  {metrics.map((r) => (
                    <div key={r.method}>
                      <span>{r.label}</span>
                      <div>
                        <i style={{ width: percent(r.success_rate) }} />
                      </div>
                      <strong>{percent(r.success_rate)}</strong>
                    </div>
                  ))}
                </div>
                <div
                  className="table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="实验数据表，可横向滚动"
                >
                  <table aria-label="实验三算法对比结果">
                    <thead>
                      <tr>
                        <th scope="col">方法</th>
                        <th scope="col">有效答案率</th>
                        <th scope="col">扩展节点</th>
                        <th scope="col">含模拟总工作量</th>
                        <th scope="col">成功路径跳数</th>
                        <th scope="col">后继检查数</th>
                        <th scope="col">冗余路径比例</th>
                        <th scope="col">无效答案率</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.map((r) => (
                        <tr key={r.method}>
                          <th scope="row">{r.label}</th>
                          <td>{percent(r.success_rate)}</td>
                          <td>{r.mean_expanded.toFixed(2)}</td>
                          <td>{r.mean_total_work.toFixed(2)}</td>
                          <td>
                            {r.mean_hops_success == null
                              ? "—"
                              : r.mean_hops_success.toFixed(2)}
                          </td>
                          <td>{r.mean_successor_checks.toFixed(2)}</td>
                          <td>{percent(r.mean_redundant_paths)}</td>
                          <td>{percent(r.invalid_answer_rate)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="helper-text">
                  答案率要求端点正确且每条证据有效。跳数只统计成功路径；后继检查数包含模拟中的检查。MCTS预算只约束树扩展，模拟另列，不能只看树节点数宣称更快。全部结果来自教学合成图。
                </p>
              </section>
              {paired.length > 0 && (
                <section className="task-section">
                  <h2>GA与学习引导是否真的有收益？</h2>
                  <p>
                    与固定启发式 A*
                    在同一批问题上配对。下表为有效答案率的百分点差，按整张图重采样得到95%区间；区间跨0时不能断言收益稳定。
                  </p>
                  <div
                    className="table-scroll"
                    tabIndex={0}
                    role="region"
                    aria-label="配对收益区间，可横向滚动"
                  >
                    <table aria-label="算法配对收益">
                      <thead>
                        <tr>
                          <th scope="col">方法</th>
                          <th scope="col">答案率变化</th>
                          <th scope="col">95%图级重采样区间</th>
                          <th scope="col">独立图数</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paired.map((r) => (
                          <tr key={r.method}>
                            <th scope="row">
                              {snapshot.methods.find((m) => m.id === r.method)
                                ?.name || r.method}
                            </th>
                            <td>
                              {(r.success_delta * 100).toFixed(1)} 个百分点
                            </td>
                            <td>
                              [{(r.graph_bootstrap_low * 100).toFixed(1)},{" "}
                              {(r.graph_bootstrap_high * 100).toFixed(1)}]
                            </td>
                            <td>{r.graph_clusters}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
              <section className="task-section">
                <h2>失败案例与改进边界</h2>
                {snapshot.failures.length === 0 && (
                  <p className="task-empty">
                    尚无已保存的失败案例；不据此认定所有搜索均成功。
                  </p>
                )}
                {snapshot.failures.slice(0, 5).map((f, i) => (
                  <details className="task-details" key={i}>
                    <summary>
                      {snapshot.methods.find((m) => m.id === f.method)?.name ||
                        f.method}{" "}
                      ·{" "}
                      {f.status === "budget_exhausted"
                        ? "预算耗尽"
                        : f.status === "found"
                          ? "证据未通过"
                          : "未找到答案"}
                    </summary>
                    <p>{f.query.text}</p>
                    <p>
                      {f.diagnosis}。独立穷举存在 {f.oracle_paths} 条有效路径。
                    </p>
                  </details>
                ))}
                <p>
                  GA改善验证适应度不保证测试收益；主题关联变化会改变启发式效果。学习概率和LLM评分均不能替代原始证据。
                </p>
              </section>
            </>
          )}
        </>
      )}
      <nav className="task-bottom-nav" aria-label="实验三返回导航">
        <a href={labHref("lab03")}>
          返回实验三任务目录
          <ArrowRight size={16} />
        </a>
      </nav>
    </main>
  );
}
