import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Play,
  DownloadSimple,
  ArrowClockwise,
} from "@phosphor-icons/react";
import type { CourseLab, LabOneResult } from "../types";
import { request, downloadJson } from "../api";
import { labHref } from "../routes";
import { formatPattern, holdingExample } from "../rule-notation";
import RuleNotation from "./RuleNotation";
import RuleExpression from "./RuleExpression";
const KnowledgeDiagram = lazy(() => import("./KnowledgeDiagram"));
const representations = [
  {
    id: "graph",
    name: "知识图谱",
    logic: "实体、关系与属性组成图结构，模式层限定关系类型。",
    strength: "多跳关联与路径表达直观，便于发现结构和追溯关系。",
    weakness: "图结构本身不自动给出业务推论，复杂约束仍需规则。",
    scene: "关系探索、股权链路、语义检索",
  },
  {
    id: "rules",
    name: "产生式规则",
    logic: "IF 前提成立，THEN 推导结论；事实库、规则库与推理机配合。",
    strength: "规则和依据清晰，可解释推理步骤。",
    weakness: "规则维护与冲突管理有成本，复杂数值路径不易表达。",
    scene: "诊断规则、可解释业务决策",
  },
  {
    id: "frames",
    name: "框架表示",
    logic: "对象以槽和侧面表示，沿类别继承，实例可覆盖缺省值。",
    strength: "对象档案和默认知识组织清晰。",
    weakness: "默认值并非事实，多跳关系推理不是强项。",
    scene: "企业档案、分类体系、继承与缺省知识",
  },
];
export default function LabOne({
  lab,
  moduleId,
  active = true,
}: {
  lab: CourseLab;
  moduleId: string;
  active?: boolean;
}) {
  const module = lab.modules.find((item) => item.id === moduleId)!;
  const [snapshot, setSnapshot] = useState<LabOneResult | null>(null);
  const [result, setResult] = useState<LabOneResult | null>(null);
  const [view, setView] = useState("graph");
  const [task, setTask] = useState("rules");
  const [source, setSource] = useState("P1");
  const [target, setTarget] = useState("C3");
  const [depth, setDepth] = useState(10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pending = useRef<AbortController | null>(null);
  const locked = useRef(false);
  async function load() {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    locked.current = true;
    setBusy(true);
    setError("");
    try {
      const value = await request<LabOneResult>("/api/labs/lab01/snapshot", {
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setSnapshot(value);
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : "读取失败");
    } finally {
      if (pending.current === controller) {
        locked.current = false;
        setBusy(false);
      }
    }
  }
  useEffect(() => {
    if (
      active &&
      !snapshot &&
      (moduleId === "representation" || moduleId === "demo")
    )
      void load();
    if (!active) {
      pending.current?.abort();
      locked.current = false;
      setBusy(false);
    }
    return () => {
      pending.current?.abort();
      locked.current = false;
    };
  }, [moduleId, active]);
  useEffect(() => {
    if (
      task !== "path" &&
      snapshot &&
      !snapshot.entities.some((e) => e.eid === target && e.kind === "Company")
    ) {
      setTarget(snapshot.entities.find((e) => e.kind === "Company")!.eid);
      setResult(null);
    }
  }, [task, target, snapshot]);
  function invalidate() {
    pending.current?.abort();
    locked.current = false;
    setBusy(false);
    setResult(null);
    setError("");
  }
  async function run() {
    if (!active || locked.current) return;
    const controller = new AbortController();
    pending.current = controller;
    locked.current = true;
    setBusy(true);
    setError("");
    try {
      const value = await request<LabOneResult>("/api/labs/lab01/run", {
        method: "POST",
        body: JSON.stringify({ task, source, target, max_depth: depth }),
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setResult(value);
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : "任务失败");
    } finally {
      if (pending.current === controller) {
        locked.current = false;
        setBusy(false);
      }
    }
  }
  const names = Object.fromEntries(
    (snapshot?.entities || []).map((e) => [e.eid, e.name]),
  );
  const holdingRule = snapshot?.rules.find((rule) => rule.name === "R1");
  const example = holdingExample(
    holdingRule && {
      name: holdingRule.name,
      antecedents: holdingRule.conditions.map(formatPattern),
      consequent: formatPattern(holdingRule.conclusions[0]),
      guard_conditions: holdingRule.guard_conditions,
      guard_complete: holdingRule.guard_complete,
    },
    snapshot?.facts || [],
    snapshot?.entities || [],
  );
  if (!active) return null;
  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="page-shell course-shell task-page"
    >
      <nav className="course-breadcrumb" aria-label="当前位置">
        <a href={labHref("lab01")}>
          <ArrowLeft size={16} />
          实验一
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
      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button
            className="button secondary"
            onClick={() => void (snapshot ? run() : load())}
          >
            重试
            <ArrowClockwise size={16} />
          </button>
        </div>
      )}
      {moduleId === "representation" ? (
        <>
          <section className="task-section">
            <header>
              <h2>同一份知识，三种表示</h2>
              <p>
                实验一固定案例版本，读取原始算法与共享
                CSV。切换表示不改变知识或运行实验二评分。
              </p>
            </header>
            <div className="method-tabs" aria-label="知识表示方法">
              {representations.map((item) => (
                <button
                  key={item.id}
                  aria-pressed={view === item.id}
                  className={view === item.id ? "active" : ""}
                  onClick={() => setView(item.id)}
                >
                  {item.name}
                </button>
              ))}
            </div>
            <p className="method-definition">
              {representations.find((item) => item.id === view)?.logic}
            </p>
            {!snapshot ? (
              <div
                className="skeleton course-skeleton"
                aria-label="正在读取实验一知识"
              />
            ) : view === "graph" ? (
              <>
                <div className="task-metrics">
                  <span>{snapshot.stats.entities} 个实体</span>
                  <span>{snapshot.stats.raw_relations} 条原始关系</span>
                  <span>模式层类型约束</span>
                </div>
                <Suspense
                  fallback={<div className="skeleton diagram-skeleton" />}
                >
                  <KnowledgeDiagram data={snapshot} />
                </Suspense>
                <details className="task-details">
                  <summary>查看完整三元组</summary>
                  <div className="table-scroll">
                    <table aria-label="实验一知识图谱三元组">
                      <thead>
                        <tr>
                          <th>头实体</th>
                          <th>关系</th>
                          <th>尾实体</th>
                          <th>属性</th>
                        </tr>
                      </thead>
                      <tbody>
                        {snapshot.triples.map((triple, i) => (
                          <tr key={i}>
                            <td>{names[triple.source]}</td>
                            <td>{triple.name}</td>
                            <td>{names[triple.target]}</td>
                            <td>
                              {triple.relation === "HOLDS"
                                ? (Number(triple.value) * 100).toFixed(1) + "%"
                                : String(triple.value)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
                <details className="task-details">
                  <summary>查看模式层与一阶谓词事实</summary>
                  <pre>
                    {JSON.stringify(
                      { schema: snapshot.schema, facts: snapshot.facts },
                      null,
                      2,
                    )}
                  </pre>
                </details>
              </>
            ) : view === "rules" ? (
              <>
                <RuleNotation example={example} />
                <div className="table-scroll">
                  <table aria-label="实验一产生式规则">
                    <thead>
                      <tr>
                        <th>规则</th>
                        <th>含义</th>
                        <th>前提 → 结论</th>
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.rules.map((rule) => (
                        <tr key={rule.name}>
                          <th>{rule.name}</th>
                          <td>{rule.description}</td>
                          <td>
                            <RuleExpression
                              antecedents={rule.conditions.map(formatPattern)}
                              conclusions={rule.conclusions.map(formatPattern)}
                              guardConditions={rule.guard_conditions}
                              guardComplete={
                                !rule.has_guard || rule.guard_complete === true
                              }
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <>
                <label className="task-inline-field">
                  查看企业框架
                  <select
                    aria-label="实验一框架企业"
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                  >
                    {snapshot.entities
                      .filter((e) => e.kind === "Company")
                      .map((e) => (
                        <option key={e.eid} value={e.eid}>
                          {e.name}
                        </option>
                      ))}
                  </select>
                </label>
                <div className="table-scroll">
                  <table aria-label="实验一框架槽值">
                    <thead>
                      <tr>
                        <th>槽</th>
                        <th>当前值</th>
                        <th>来源</th>
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.frames_before
                        .find((f) => f.id === target)
                        ?.slots.map((slot) => (
                          <tr key={slot.name}>
                            <th>{slot.name}</th>
                            <td>
                              {String(slot.value ?? "未填写")}
                              {slot.unit ? String(slot.unit) : ""}
                            </td>
                            <td>
                              {slot.inherited
                                ? "继承／缺省（不是已核实事实）"
                                : "实例填写"}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
          <section className="task-section">
            <h2>方法优缺点与典型场景</h2>
            <div className="table-scroll">
              <table aria-label="知识表示方法比较">
                <thead>
                  <tr>
                    <th>方法</th>
                    <th>优点</th>
                    <th>局限</th>
                    <th>适用场景</th>
                  </tr>
                </thead>
                <tbody>
                  {representations.map((item) => (
                    <tr key={item.id}>
                      <th>{item.name}</th>
                      <td>{item.strength}</td>
                      <td>{item.weakness}</td>
                      <td>{item.scene}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="helper-text">
              SQLite
              是关系表对照。保存的查询基准使用相同无环数据和多重路径语义，不代表所有数据库的性能。
            </p>
            <details className="task-details">
              <summary>查看图 DFS、递归 CTE 与逐层 JOIN 的原始实测</summary>
              {snapshot?.benchmark ? (
                <div className="table-scroll">
                  <table aria-label="实验一知识表示效率对照">
                    <thead>
                      <tr>
                        <th>深度</th>
                        <th>查询方法</th>
                        <th>中位耗时／ms</th>
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.benchmark.results.map((r, i) => (
                        <tr key={i}>
                          <td>{r.depth}</td>
                          <td>{r.method}</td>
                          <td>{r.median_ms.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p>尚无保存的基准结果。</p>
              )}
            </details>
          </section>
        </>
      ) : moduleId === "industry" ? (
        <section className="task-section editorial-section">
          <h2>真实案例：ICIJ Offshore Leaks 的关系查询</h2>
          <p>
            ICIJ 的公开数据库把 Panama Papers
            等调查项目中的离岸实体、人员与中介记录连接起来。
            企业也可能作为另一家企业的股东，因此查询一个直接股东还不足以解释多层联系。
            这是已经落地的实体关系查询案例，为本项目的股权路径追踪与人工核查流程提供参照。
          </p>
          <a
            className="source-inline-link"
            href="https://offshoreleaks.icij.org/pages/about"
            target="_blank"
            rel="noreferrer"
          >
            查看 ICIJ 数据库公开说明（新窗口）
            <ArrowRight size={16} />
          </a>
          <p>
            本项目没有导入其中的真实人员数据，也不声称复现其内部推理系统。王建国、恒通科技等是为课堂演示构造的实体。
          </p>
          <div className="table-scroll">
            <table aria-label="真实行业案例与本项目的对应">
              <thead>
                <tr>
                  <th>行业中的实际问题</th>
                  <th>本项目的可运行演示</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>分散记录如何关联到同一主体？</td>
                  <td>
                    使用统一实体
                    ID，把持股、交易、担保和亲属记录映射到三元组与企业框架。
                  </td>
                </tr>
                <tr>
                  <td>企业之间的多层联系如何查询？</td>
                  <td>
                    查询股权简单路径与最短关联路径，同时保留关系方向和路径贡献。
                  </td>
                </tr>
                <tr>
                  <td>发现联系后，怎样复核而非直接下结论？</td>
                  <td>
                    用教学规则生成候选控制与复核线索，逐条展示前提及来源，再由人工核查。
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="industry-process">
            {[
              "明确业务问题与授权数据范围",
              "整理企业、人员和关系记录",
              "建立模式层与多种知识表示",
              "执行路径查询与规则推理",
              "追溯来源并由人工核查结论",
            ].map((text, i) => (
              <div key={text}>
                <span>{i + 1}</span>
                <p>{text}</p>
              </div>
            ))}
          </div>
          <h3>本项目怎样落地这一流程？</h3>
          <p>
            CSV
            提供教学输入；知识图谱保存关系，框架组织企业档案，产生式规则生成控制和结构线索；推导结果只在内存回写，原始数据保持不变。
          </p>
          <a className="button secondary" href={labHref("lab01", "demo")}>
            运行小型推理演示
            <ArrowRight size={17} />
          </a>
        </section>
      ) : moduleId === "demo" ? (
        <section className="task-section">
          <header>
            <h2>选择一个小型推理任务</h2>
            <p>
              运行的是实验一原方法。更换输入后清除旧结果，每次展示都对应实际运行。
            </p>
          </header>
          <form
            className="lab-task-form"
            onSubmit={(e) => {
              e.preventDefault();
              void run();
            }}
          >
            <label>
              演示任务
              <select
                aria-label="实验一演示任务"
                value={task}
                disabled={busy}
                onChange={(e) => {
                  invalidate();
                  setTask(e.target.value);
                }}
              >
                <option value="rules">产生式推理与知识增强</option>
                <option value="penetration">股权简单路径穿透</option>
                <option value="path">最短关联路径查询</option>
              </select>
            </label>
            {task !== "penetration" && (
              <label>
                查询主体
                <select
                  aria-label="实验一查询主体"
                  value={source}
                  disabled={busy}
                  onChange={(e) => {
                    invalidate();
                    setSource(e.target.value);
                  }}
                >
                  {snapshot?.entities.map((e) => (
                    <option key={e.eid} value={e.eid}>
                      {e.eid} · {e.name}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label>
              查询目标
              <select
                aria-label="实验一查询目标"
                value={target}
                disabled={busy}
                onChange={(e) => {
                  invalidate();
                  setTarget(e.target.value);
                }}
              >
                {snapshot?.entities
                  .filter((e) => task === "path" || e.kind === "Company")
                  .map((e) => (
                    <option key={e.eid} value={e.eid}>
                      {e.eid} · {e.name}
                    </option>
                  ))}
              </select>
            </label>
            {task === "penetration" && (
              <label>
                最大深度
                <select
                  aria-label="实验一穿透深度"
                  value={depth}
                  disabled={busy}
                  onChange={(e) => {
                    invalidate();
                    setDepth(Number(e.target.value));
                  }}
                >
                  {[1, 2, 4, 6, 10].map((d) => (
                    <option key={d}>{d}</option>
                  ))}
                </select>
              </label>
            )}
            <button className="button primary" disabled={busy || !snapshot}>
              <Play size={17} />
              {busy ? "正在运行…" : "运行实验一任务"}
            </button>
          </form>
          {!result ? (
            <div className="task-empty">
              <p>选择输入并运行。这里将出现真实计算结果、证明步骤或路径。</p>
            </div>
          ) : (
            <section className="task-result" aria-live="polite">
              <div className="task-result-title">
                <h2>运行结果</h2>
                <button
                  className="button secondary"
                  onClick={() => downloadJson(result, "lab01-task.json")}
                >
                  <DownloadSimple size={16} />
                  导出结果
                </button>
              </div>
              {task === "rules" ? (
                <>
                  <div className="task-verdict">
                    <strong>
                      {result.proved
                        ? "当前规则下可以证明"
                        : "当前知识库暂不能证明"}
                    </strong>
                    <span>
                      {source} 控制 {target}；未证明不等于现实为假。
                    </span>
                  </div>
                  <div className="task-metrics">
                    <span>
                      {result.stats.input_facts} → {result.stats.total_facts}{" "}
                      条事实
                    </span>
                    <span>{result.stats.cycles} 轮规则周期</span>
                    <span>新增 {result.stats.added_relations} 条控制关系</span>
                  </div>
                  <details open className="task-details">
                    <summary>查看目标证明</summary>
                    <pre>
                      {result.proof?.length
                        ? result.proof.join("\n")
                        : "没有可用证明；未把查询目标伪装成已知事实。"}
                    </pre>
                  </details>
                  <details className="task-details">
                    <summary>完整规则执行与知识增强记录</summary>
                    <pre>
                      {JSON.stringify(
                        {
                          steps: result.steps,
                          enhanced_relations: result.enhanced_relations,
                          frames_after: result.frames_after,
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                </>
              ) : task === "penetration" ? (
                <>
                  <div className="table-scroll">
                    <table aria-label="实验一股权穿透结果">
                      <thead>
                        <tr>
                          <th>路径端点</th>
                          <th>简单路径累计权益</th>
                          <th>教学筛查候选</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.owners?.map((owner) => (
                          <tr key={owner.owner}>
                            <th>{owner.name}</th>
                            <td>{(owner.ratio * 100).toFixed(4)}%</td>
                            <td>
                              {owner.beneficial ? "达到教学阈值" : "未标记"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <details open className="task-details">
                    <summary>逐条路径和乘积</summary>
                    <pre>
                      {JSON.stringify(
                        result.owners?.map((o) => ({
                          name: o.name,
                          paths: o.paths,
                        })),
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                </>
              ) : (
                <div className="task-path">
                  {result.path === null ? (
                    <p>当前图中没有关联路径。</p>
                  ) : result.path?.length === 0 ? (
                    <p>起点与终点相同，没有需要经过的边。</p>
                  ) : (
                    result.path?.map(([u, rel, v, d], i) => (
                      <p key={i}>
                        {names[u] || u} {d} [
                        {result.schema.find((s) => s.id === rel)?.name || rel}]{" "}
                        {d} {names[v] || v}
                      </p>
                    ))
                  )}
                </div>
              )}
              <p className="helper-text">{result.note}</p>
            </section>
          )}
        </section>
      ) : moduleId === "limitations" ? (
        <section className="task-section editorial-section">
          <h2>必须保留的边界</h2>
          <dl className="limits-list">
            <div>
              <dt>表示不是数据真实性认证</dt>
              <dd>知识图谱能组织关系，但输入来源和证据仍需核查。</dd>
            </div>
            <div>
              <dt>教学控制不是法律认定</dt>
              <dd>
                配偶一致行动、过半持股控制传递等是简化规则；控制、经济权益和风险是不同概念。
              </dd>
            </div>
            <div>
              <dt>缺省值不是已知事实</dt>
              <dd>
                框架默认“低”不能解释成企业安全。查询无法证明不能解释成事实为假。
              </dd>
            </div>
            <div>
              <dt>简单路径有局限</dt>
              <dd>
                实验一对持股环路截断；实验二扩展才引入联立方程的循环经济权益求解，不能用新结果替换旧方法。
              </dd>
            </div>
            <div>
              <dt>性能结论有范围</dt>
              <dd>
                图查询与 SQL
                只在所测数据、实现和机器上比较，不代表所有图数据库或关系数据库。
              </dd>
            </div>
          </dl>
          <h3>后续演进方向</h3>
          <p>
            证据有效期与身份核验、冲突处理、规模优化、数据质量建模和混合推理。实验二已实现其中部分能力，仍没有接入权威身份或工商来源。
          </p>
          <a className="button secondary" href={labHref("lab02")}>
            查看实验二怎样进一步改进
            <ArrowRight size={17} />
          </a>
        </section>
      ) : null}
      <nav className="task-bottom-nav" aria-label="实验相关页面">
        <a href={labHref("lab01")}>
          返回实验一任务目录
          <ArrowRight size={16} />
        </a>
      </nav>
    </main>
  );
}
