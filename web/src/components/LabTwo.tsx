import {
  lazy,
  Suspense,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type MouseEvent,
} from "react";
import {
  ArrowLeft,
  ArrowRight,
  Play,
  DownloadSimple,
  Info,
  CaretDown,
} from "@phosphor-icons/react";
import type {
  Bootstrap,
  CaseBundle,
  Parameters,
  CourseLab,
  LabTwoResult,
} from "../types";
import { canonical, downloadJson, request } from "../api";
import {
  entityHref,
  labHref,
  analysisHref,
  reasoningGraphHref,
  recordsHref,
  governanceHref,
} from "../routes";
import TaskInputs from "./TaskInputs";
import type { TaskInputMode, TaskInputUpdate } from "./TaskInputs";
const Experiments = lazy(() => import("./Experiments"));
const RulesPage = lazy(() => import("./RulesPage"));
const featureNames: Record<string, string> = {
  opacity: "股权不透明度代理",
  related_trade: "关联交易额／注册资本",
  guarantee: "对外担保额／注册资本",
  cycle: "持股或担保回路线索",
};
const methods: Record<string, string> = {
  hybrid_unlocked: "不锁定结构",
  hybrid_locked: "保留结构约束",
  hybrid_calibrated: "校准复核（保留证明）",
};
export default function LabTwo({
  lab,
  moduleId,
  workspace,
  boot,
  parameters,
  workspaceBusy,
  workspaceError,
  onReconnect,
  onChooseCase,
  onApplyInputs,
  onParameters,
  onOpenDetails,
  active = true,
}: {
  lab: CourseLab;
  moduleId: string;
  workspace: CaseBundle | null;
  boot: Bootstrap | null;
  parameters: Parameters;
  workspaceBusy: boolean;
  workspaceError: string;
  onReconnect: () => void;
  onChooseCase: (id: string) => Promise<boolean>;
  onApplyInputs: (update: TaskInputUpdate) => void;
  onParameters: (update: Partial<Parameters>) => void;
  onOpenDetails: (target: string, method?: string, source?: string) => void;
  active?: boolean;
}) {
  const module = lab.modules.find((m) => m.id === moduleId)!;
  const [source, setSource] = useState("P1");
  const [target, setTarget] = useState("C3");
  const [algorithm, setAlgorithm] = useState("forward");
  const [inputPanel, setInputPanel] = useState<TaskInputMode | null>(null);
  const inputPanelId = useId();
  const inputTrigger = useRef<HTMLButtonElement | null>(null);
  const restoreInputFocus = useRef(false);
  useEffect(() => {
    if (!restoreInputFocus.current || inputPanel || workspaceBusy || !active)
      return;
    const trigger = inputTrigger.current;
    if (trigger?.isConnected && !trigger.closest("[hidden]")) {
      trigger.focus();
      restoreInputFocus.current = false;
    }
  }, [inputPanel, workspaceBusy, active]);
  const { model_source: model, threshold } = parameters;
  const isResolution =
    moduleId === "deterministic" && algorithm === "resolution";
  const inputTarget = isResolution ? "C3" : target;
  const resolutionReady =
    !isResolution ||
    [
      ["P1", "C1"],
      ["C1", "C2"],
      ["C2", "C3"],
      ["C1", "C3"],
    ].every(([holder, company]) =>
      workspace?.case.holdings.some(
        (holding) => holding.holder === holder && holding.target === company,
      ),
    );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<LabTwoResult | null>(null);
  const [accepted, setAccepted] = useState("");
  const [resultAlgorithm, setResultAlgorithm] = useState("forward");
  const pending = useRef<AbortController | null>(null);
  const locked = useRef(false);
  const signature = useMemo(
    () =>
      canonical({
        case: workspace?.case,
        completeness: isResolution ? undefined : workspace?.completeness,
        source:
          moduleId === "deterministic" && !isResolution ? source : undefined,
        target: moduleId !== "modeling" && !isResolution ? target : undefined,
        algorithm: moduleId === "deterministic" ? algorithm : undefined,
        model: ["uncertain", "hybrid"].includes(moduleId) ? model : undefined,
        threshold: moduleId === "hybrid" ? threshold : undefined,
        moduleId,
      }),
    [
      workspace,
      source,
      target,
      algorithm,
      model,
      threshold,
      moduleId,
      isResolution,
    ],
  );
  const stale = !!result && accepted !== signature;
  useEffect(() => {
    if (!workspace) return;
    const entities = workspace.case.entities;
    if (!entities.some((e) => e.eid === source))
      setSource(
        entities.find((e) => e.kind === "Person")?.eid || entities[0].eid,
      );
    if (!entities.some((e) => e.eid === target && e.kind === "Company"))
      setTarget(entities.find((e) => e.kind === "Company")!.eid);
  }, [workspace, source, target]);
  useEffect(() => {
    if (boot && !boot.models.some((m) => m.id === model))
      onParameters({ model_source: "manual" });
  }, [boot, model, onParameters]);
  useEffect(() => {
    if (!active) {
      setInputPanel(null);
      restoreInputFocus.current = false;
      pending.current?.abort();
      locked.current = false;
      setBusy(false);
    }
    return () => {
      pending.current?.abort();
      locked.current = false;
    };
  }, [active]);
  async function run() {
    if (
      !active ||
      !workspace ||
      locked.current ||
      workspaceBusy ||
      inputPanel ||
      !resolutionReady
    )
      return;
    locked.current = true;
    setBusy(true);
    setError("");
    pending.current?.abort();
    const c = new AbortController();
    pending.current = c;
    const task =
      moduleId === "modeling"
        ? "modeling"
        : moduleId === "deterministic"
          ? algorithm === "resolution"
            ? "resolution"
            : "deterministic"
          : moduleId === "uncertain"
            ? "uncertain"
            : "hybrid";
    try {
      const value = await request<LabTwoResult>("/api/labs/lab02/run", {
        method: "POST",
        signal: c.signal,
        body: JSON.stringify({
          case: workspace.case,
          completeness: workspace.completeness,
          method: "hybrid_locked",
          threshold,
          model_source: model,
          task,
          source,
          target,
        }),
      });
      if (!c.signal.aborted) {
        setResult(value);
        setAccepted(signature);
        setResultAlgorithm(algorithm);
      }
    } catch (e) {
      if (!c.signal.aborted)
        setError(e instanceof Error ? e.message : "实验运行失败");
    } finally {
      if (pending.current === c) {
        locked.current = false;
        setBusy(false);
      }
    }
  }
  function openDetails(event: MouseEvent<HTMLAnchorElement>) {
    if (!workspace || busy || workspaceBusy || inputPanel) {
      event.preventDefault();
      return;
    }
    if (
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    )
      return;
    onOpenDetails(
      target,
      moduleId === "hybrid" ? "hybrid_locked" : undefined,
      moduleId === "deterministic" ? source : undefined,
    );
  }
  const inputControls = (
    <>
      <div className="task-links" role="group" aria-label="实验输入设置">
        {(
          [
            ["case", "选择案例"],
            ["entity", isResolution ? "调整 C3 持股输入" : "调整目标企业输入"],
            ["completeness", "资料完整性设置"],
          ] as const
        )
          .filter(([mode]) => !isResolution || mode !== "completeness")
          .map(([mode, label]) => (
            <button
              type="button"
              key={mode}
              disabled={
                busy ||
                workspaceBusy ||
                !workspace ||
                !!inputPanel ||
                (mode === "entity" && !resolutionReady)
              }
              aria-expanded={inputPanel === mode}
              aria-controls={inputPanelId}
              onClick={(event) => {
                inputTrigger.current = event.currentTarget;
                setInputPanel(mode);
              }}
            >
              {label}
              <CaretDown size={16} aria-hidden="true" />
            </button>
          ))}
      </div>
      <div id={inputPanelId} hidden={!inputPanel}>
        {inputPanel && workspace && (
          <TaskInputs
            key={inputPanel + ":" + workspace.id + ":" + inputTarget}
            mode={inputPanel}
            workspace={workspace}
            boot={boot}
            target={inputTarget}
            busy={workspaceBusy}
            error={workspaceError}
            onChooseCase={onChooseCase}
            onApply={onApplyInputs}
            onClose={() => {
              restoreInputFocus.current = true;
              setInputPanel(null);
            }}
          />
        )}
      </div>
    </>
  );
  const related =
    moduleId === "deterministic" && !isResolution
      ? [
          {
            href: entityHref(target, "evidence"),
            title: "查看目标企业推理证据",
            description: "沿当前查询展开证明树，核对规则与前提。",
          },
          {
            href: reasoningGraphHref(target),
            title: "推导关系与证明演示",
            description: "在关系图中定位当前企业，逐步播放证明。",
          },
          {
            href: recordsHref(),
            title: "验证与完整推理记录",
            description: "保留查询主体和目标，反向验证并查看全部步骤。",
          },
        ]
      : moduleId === "hybrid"
        ? [
            {
              href: entityHref(target),
              title: "查看目标企业分析",
              description: "按「保留结构约束」查看评分、复核行动与依据。",
            },
            {
              href: analysisHref(),
              title: "查看全案例分析与复核结果",
              description: "沿用本页参数，比较案例中全部企业。",
            },
          ]
        : [];
  if (!active) return null;
  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="page-shell course-shell task-page"
    >
      <nav className="course-breadcrumb" aria-label="当前位置">
        <a href={labHref("lab02")}>
          <ArrowLeft size={16} />
          实验二
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
      {!workspace && workspaceError && (
        <div className="error-banner" role="alert">
          <span>{workspaceError}</span>
          <button
            className="button secondary"
            disabled={workspaceBusy}
            onClick={onReconnect}
          >
            重新连接共享案例
          </button>
        </div>
      )}
      {moduleId === "modeling" ? (
        <>
          <section className="task-section">
            <h2>统一场景与输入知识</h2>
            <p>
              自然人、企业、持股、交易、担保和亲属记录构成同一个应用场景。规则推导结构命题，指标映射服务于不确定性评分；二者不是同一种输出。
            </p>
            <div className="task-metrics">
              <span>{workspace?.case.label || "正在载入案例"}</span>
              <span>{workspace?.case.entities.length || "—"} 个实体</span>
              <span>{workspace?.case.holdings.length || "—"} 条持股记录</span>
            </div>
            <form
              className="lab-task-form modeling-task-form"
              onSubmit={(event) => {
                event.preventDefault();
                void run();
              }}
            >
              <label>
                编辑对象
                <select
                  aria-label="实验二建模编辑对象"
                  value={target}
                  disabled={busy || !!inputPanel}
                  onChange={(event) => setTarget(event.target.value)}
                >
                  {workspace?.case.entities
                    .filter((entity) => entity.kind === "Company")
                    .map((entity) => (
                      <option key={entity.eid} value={entity.eid}>
                        {entity.eid} · {entity.name}
                      </option>
                    ))}
                </select>
              </label>
              <button
                className="button primary"
                disabled={busy || workspaceBusy || !workspace || !!inputPanel}
              >
                <Play size={17} />
                {busy ? "正在构建事实…" : "构建当前案例的谓词事实"}
              </button>
            </form>
            {inputControls}
            {error && (
              <p className="field-error" role="alert">
                {error}
              </p>
            )}
            {result?.task === "modeling" && (
              <>
                {stale && (
                  <p className="helper-text">
                    输入已变化，下方是旧快照，请重新构建。
                  </p>
                )}
                <details open className="task-details">
                  <summary>一阶谓词与来源（{result.facts?.length} 条）</summary>
                  <p className="helper-text">
                    这里是已代入实体编号和数值的具体事实，与下方带变量的规则模板不同。
                  </p>
                  <div className="table-scroll">
                    <table aria-label="实验二一阶谓词事实">
                      <thead>
                        <tr>
                          <th>谓词公式</th>
                          <th>来源</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.facts?.map((fact, i) => (
                          <tr key={i}>
                            <td>
                              <code>{fact.formula}</code>
                            </td>
                            <td>{fact.source}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
                <details className="task-details">
                  <summary>结构推导后的新增知识与依据</summary>
                  <pre>{JSON.stringify(result.derived_facts, null, 2)}</pre>
                </details>
              </>
            )}
            <details className="task-details">
              <summary>查看四类指标的映射与未知处理</summary>
              <dl className="limits-list">
                <div>
                  <dt>不透明度</dt>
                  <dd>明确的教学参数，不是从真实企业自动测量。</dd>
                </div>
                <div>
                  <dt>关联交易与对外担保</dt>
                  <dd>金额按注册资本归一化并截断，不是营收比例。</dd>
                </div>
                <div>
                  <dt>回路线索</dt>
                  <dd>
                    来自逻辑可达闭包；数据缺失且未发现正向证据时使用未知，不自动补成0。
                  </dd>
                </div>
              </dl>
            </details>
          </section>
          <section className="task-section">
            <Suspense fallback={<div className="skeleton course-skeleton" />}>
              <RulesPage embedded caseData={workspace?.case ?? null} />
            </Suspense>
          </section>
        </>
      ) : moduleId === "optimization" ? (
        <>
          <section className="task-section">
            <h2>先看对比设计，再看实测</h2>
            <div className="experiment-design">
              <div>
                <h3>确定性优化</h3>
                <p>
                  全扫描、索引朴素、半朴素增量与目标切片，核对闭包和答案一致后比较耗时。
                </p>
              </div>
              <div>
                <h3>不确定性优化</h3>
                <p>
                  CPT学习与平滑、模糊参数拟合、缺测处理；训练／验证／测试分离，避免答案泄漏。
                </p>
              </div>
              <div>
                <h3>抗干扰与负面结果</h3>
                <p>
                  多随机种子、噪声、关系缺失和错误闭环，保留误报与收益，不预设融合一定更好。
                </p>
              </div>
            </div>
            <p className="helper-text">
              这里读取正式结果。重新运行完整实验使用总目录命令：
              <code>python main.py lab02 all</code>
              ，不在网页点击后悄悄覆盖正式数据。
            </p>
          </section>
          <Suspense fallback={<div className="skeleton course-skeleton" />}>
            <Experiments embedded />
          </Suspense>
        </>
      ) : (
        <>
          <section className="task-section">
            <header>
              <h2>
                {moduleId === "deterministic"
                  ? "选择目标并执行真实推理"
                  : moduleId === "uncertain"
                    ? "同一份指标，运行两种方法"
                    : "同一输入，对比三种混合策略"}
              </h2>
              <p>
                {workspace
                  ? "当前案例：" + workspace.case.label
                  : "正在载入共享案例…"}
              </p>
            </header>
            <form
              className="lab-task-form"
              onSubmit={(e) => {
                e.preventDefault();
                void run();
              }}
            >
              {moduleId === "deterministic" ? (
                <>
                  <label>
                    推理方法
                    <select
                      aria-label="实验二推理方法"
                      value={algorithm}
                      disabled={busy || !!inputPanel}
                      onChange={(e) => setAlgorithm(e.target.value)}
                    >
                      <option value="forward">正向推理</option>
                      <option value="backward">表格化反向推理</option>
                      <option value="resolution">独立股权子案例归结</option>
                    </select>
                  </label>
                  {algorithm !== "resolution" && (
                    <label>
                      查询主体
                      <select
                        aria-label="实验二查询主体"
                        value={source}
                        disabled={busy || !!inputPanel}
                        onChange={(e) => setSource(e.target.value)}
                      >
                        {workspace?.case.entities.map((e) => (
                          <option key={e.eid} value={e.eid}>
                            {e.eid} · {e.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                </>
              ) : (
                <label>
                  参数来源
                  <select
                    aria-label="实验二参数来源"
                    value={model}
                    disabled={busy || !boot || !!inputPanel}
                    onChange={(e) =>
                      onParameters({ model_source: e.target.value })
                    }
                  >
                    {boot?.models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {algorithm !== "resolution" || moduleId !== "deterministic" ? (
                <label>
                  目标企业
                  <select
                    aria-label="实验二目标企业"
                    value={target}
                    disabled={busy || !!inputPanel}
                    onChange={(e) => setTarget(e.target.value)}
                  >
                    {workspace?.case.entities
                      .filter((e) => e.kind === "Company")
                      .map((e) => (
                        <option key={e.eid} value={e.eid}>
                          {e.eid} · {e.name}
                        </option>
                      ))}
                  </select>
                </label>
              ) : (
                <p className="resolution-scope">
                  固定目标 Controls(P1, C3)。只编译 P1／C1／C2／C3
                  子案例的原始持股与显式数值比较。
                </p>
              )}
              {moduleId === "hybrid" && (
                <label>
                  复核阈值
                  <select
                    aria-label="实验二复核阈值"
                    value={threshold}
                    disabled={busy || !!inputPanel}
                    onChange={(e) =>
                      onParameters({ threshold: Number(e.target.value) })
                    }
                  >
                    {[
                      ...new Set([
                        ...Array.from({ length: 19 }, (_, i) => (i + 1) / 20),
                        threshold,
                      ]),
                    ]
                      .sort((a, b) => a - b)
                      .map((n) => (
                        <option key={n} value={n}>
                          {n.toFixed(2)}
                        </option>
                      ))}
                  </select>
                </label>
              )}
              <button
                className="button primary"
                disabled={
                  busy ||
                  workspaceBusy ||
                  !workspace ||
                  !!inputPanel ||
                  !resolutionReady
                }
              >
                <Play size={17} />
                {busy ? "正在运行…" : "运行本节实验"}
              </button>
            </form>
            {inputControls}
            {isResolution && !resolutionReady && workspace && (
              <p className="field-error" role="status">
                当前案例缺少 P1→C1、C1→C2、C2→C3 或 C1→C3
                持股记录。请选择包含这四条输入的案例，或使用正向／反向推理。
              </p>
            )}
            {moduleId === "deterministic" && algorithm === "resolution" && (
              <p className="helper-text">
                此演示保留明确数值守卫，不声称全套算术知识库已经转换为纯一阶子句。
              </p>
            )}
            {error && (
              <div className="error-banner" role="alert">
                <span>{error}</span>
                <button className="button secondary" onClick={() => void run()}>
                  重试
                </button>
              </div>
            )}
            {stale && (
              <div className="workspace-change" role="status">
                <Info size={18} />
                <p>
                  输入或方法已变化，下方是上次运行的快照。重新运行后再作为当前结果使用。
                </p>
              </div>
            )}
            {!result ? (
              <div className="task-empty">
                <p>输入就绪。运行后可查看结论、推理步骤和耗时。</p>
              </div>
            ) : (
              <section className="task-result" aria-live="polite">
                <div className="task-result-title">
                  <div>
                    <h2>运行结果</h2>
                    {result.model_source && (
                      <p className="result-parameters">
                        参数：
                        {boot?.models.find(
                          (item) => item.id === result.model_source,
                        )?.name || result.model_source}
                        {result.decision_threshold !== undefined &&
                          " · 复核阈值 " + result.decision_threshold.toFixed(2)}
                      </p>
                    )}
                  </div>
                  <button
                    className="button secondary"
                    disabled={stale || busy}
                    onClick={() =>
                      downloadJson(result, "lab02-" + moduleId + ".json")
                    }
                  >
                    <DownloadSimple size={16} />
                    导出结果
                  </button>
                </div>
                {result.task === "deterministic" ||
                result.task === "resolution" ? (
                  <>
                    <div className="task-verdict">
                      <strong>
                        {result.status === "limit"
                          ? "达到资源预算，证明尚未完成"
                          : result.proved
                            ? "当前知识库可以证明"
                            : "当前知识库暂不能证明"}
                      </strong>
                      <span>
                        {Array.isArray(result.goal)
                          ? result.goal.join(" · ")
                          : result.goal}
                      </span>
                    </div>
                    <div className="task-metrics">
                      <span>
                        {result.task === "resolution"
                          ? "归结状态：" + result.status
                          : "正反向结论一致"}
                      </span>
                      <span>流程总耗时 {result.elapsed_ms.toFixed(2)} ms</span>
                    </div>
                    <details open className="task-details">
                      <summary>
                        {result.task === "resolution"
                          ? "归结证明步骤"
                          : resultAlgorithm === "backward"
                            ? "反向证明路径"
                            : "正向证明依据"}
                      </summary>
                      {result.task === "resolution" && (
                        <p className="helper-text">
                          归结子句和中间公式中的 ?
                          开头符号仍是逻辑变量，不是缺失数据；最终待证目标使用具体实体。
                        </p>
                      )}
                      <pre>
                        {(result.task === "resolution"
                          ? result.proof
                          : resultAlgorithm === "backward"
                            ? result.backward_proof
                            : result.forward_proof
                        )?.join("\n") || "未找到证明；不等于现实中为假。"}
                      </pre>
                    </details>
                    <details className="task-details">
                      <summary>查看完整步骤和优化统计</summary>
                      <pre>
                        {JSON.stringify(
                          {
                            forward: result.forward_stats,
                            backward: result.backward_stats,
                            resolution: result.stats,
                            steps: result.steps,
                          },
                          null,
                          2,
                        )}
                      </pre>
                    </details>
                  </>
                ) : result.task === "uncertain" ? (
                  <>
                    <div className="table-scroll">
                      <table aria-label="实验二输入指标">
                        <thead>
                          <tr>
                            <th>指标</th>
                            <th>本次观测</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(result.observation?.values || {}).map(
                            ([key, value]) => (
                              <tr key={key}>
                                <th>{featureNames[key] || key}</th>
                                <td>
                                  {value === null
                                    ? "未知／未完整采集"
                                    : value.toFixed(3)}
                                </td>
                              </tr>
                            ),
                          )}
                        </tbody>
                      </table>
                    </div>
                    <div className="score-comparison">
                      <section>
                        <h3>Bayes</h3>
                        <strong>{result.bayes?.score.toFixed(4)}</strong>
                        <p>模型内条件风险评分；先验和CPT是明确的模型参数。</p>
                      </section>
                      <section>
                        <h3>Mamdani</h3>
                        <strong>{result.fuzzy?.score.toFixed(4)}</strong>
                        <p>模糊规则和重心解模糊输出，不是风险概率。</p>
                      </section>
                    </div>
                    <details className="task-details">
                      <summary>Bayes先验与条件参数</summary>
                      <pre>
                        {JSON.stringify(result.bayes?.parameters, null, 2)}
                      </pre>
                    </details>
                    <details className="task-details">
                      <summary>Mamdani隶属度、规则激活与参数</summary>
                      <pre>
                        {JSON.stringify(
                          {
                            parameters: result.fuzzy?.parameters,
                            explanation: result.fuzzy?.explanation,
                          },
                          null,
                          2,
                        )}
                      </pre>
                    </details>
                    <p className="helper-text">
                      单个案例没有真实标签，不能在这里编造准确率。准确性、耗时和容错对比请到“算法优化与对比实验”。
                    </p>
                  </>
                ) : (
                  <>
                    <div className="table-scroll">
                      <table aria-label="实验二混合策略对照">
                        <thead>
                          <tr>
                            <th>策略</th>
                            <th>原始评分</th>
                            <th>复核建议</th>
                            <th>结构证明</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.decisions?.map((d) => (
                            <tr key={d.method}>
                              <th>{methods[d.method]}</th>
                              <td>{d.score.toFixed(4)}</td>
                              <td>
                                {d.review_required
                                  ? "优先复核"
                                  : d.review_priority === "中"
                                    ? "持续关注"
                                    : "暂未触发复核"}
                              </td>
                              <td>{d.proofs.length} 个结构命题；保留来源</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <details className="task-details">
                      <summary>查看各策略的完整依据与校准信息</summary>
                      <pre>{JSON.stringify(result.decisions, null, 2)}</pre>
                    </details>
                    <a
                      className="button secondary"
                      href={labHref("lab02", "optimization")}
                    >
                      查看正式实验中的收益与误报
                      <ArrowRight size={17} />
                    </a>
                  </>
                )}
                {result.timing_scope && (
                  <p className="helper-text">{result.timing_scope}</p>
                )}
                {result.task === "deterministic" && (
                  <p className="helper-text">
                    流程总耗时包含事实构建和正反向核对。各引擎耗时见「完整步骤和优化统计」。
                  </p>
                )}
                <p className="helper-text">{result.note}</p>
              </section>
            )}
          </section>
          {moduleId === "uncertain" && (
            <section className="task-section">
              <h2>方法特点与适用范围</h2>
              <div className="table-scroll">
                <table aria-label="不确定性方法比较">
                  <thead>
                    <tr>
                      <th>方法</th>
                      <th>知识形式</th>
                      <th>优势</th>
                      <th>局限</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>Bayes</th>
                      <td>先验、CPT、概率推断</td>
                      <td>显式处理证据与缺测，参数可学习</td>
                      <td>依赖模型假设与概率参数，不自动认证真实性</td>
                    </tr>
                    <tr>
                      <th>Mamdani</th>
                      <td>隶属函数、模糊规则、解模糊</td>
                      <td>表达“高、低、模糊边界”等专家规则</td>
                      <td>参数选择影响结果，输出不能冒称概率</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
      {related.length > 0 && (
        <section className="task-related">
          <header>
            <h2>
              {moduleId === "deterministic"
                ? "继续查看推理详情"
                : "展开分析详情"}
            </h2>
            <p>
              {moduleId === "deterministic"
                ? "使用当前输入继续追溯，始终留在确定性推理模块。"
                : "详情采用「保留结构约束」策略，与本页共享参数来源和复核阈值。"}
            </p>
          </header>
          <nav className="task-detail-links" aria-label="实验详情入口">
            {related.map((item) => (
              <a
                key={item.href}
                href={item.href}
                onClick={openDetails}
                aria-label={item.title}
                aria-disabled={
                  !workspace || busy || workspaceBusy || !!inputPanel
                }
                tabIndex={
                  !workspace || busy || workspaceBusy || !!inputPanel
                    ? -1
                    : undefined
                }
              >
                <span>
                  <strong>{item.title}</strong>
                  <small>{item.description}</small>
                </span>
                <ArrowRight size={18} />
              </a>
            ))}
          </nav>
        </section>
      )}
      {moduleId === "modeling" && (
        <nav className="task-detail-links" aria-label="建模扩展">
          <a href={governanceHref()} aria-label="扩展：证据与版本更新">
            <span>
              <strong>扩展：证据与版本更新</strong>
              <small>
                核对来源、控制权证据和历史版本，观察结论如何随证据更新。
              </small>
            </span>
            <ArrowRight size={18} />
          </a>
        </nav>
      )}
      <nav className="task-bottom-nav" aria-label="实验相关页面">
        <a href={labHref("lab02")}>
          <ArrowLeft size={16} />
          返回实验二任务目录
        </a>
      </nav>
    </main>
  );
}
