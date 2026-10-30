import { useEffect, useState } from "react";
import { reasoningGraphHref as graphHref } from "../routes";
import {
  ArrowRight,
  CaretRight,
  CheckCircle,
  Info,
  SlidersHorizontal,
  TreeStructure,
  WarningDiamond,
  ArrowSquareOut,
} from "@phosphor-icons/react";
import type { Analysis, CaseData, Decision, FactNode } from "../types";
import { fmt, shortName } from "../api";
import HoldingEditor from "./HoldingEditor";

type Props = {
  view?: "result" | "evidence" | "tune";
  standalone?: boolean;
  data: CaseData;
  analysis: Analysis | null;
  selected: string;
  dirty: boolean;
  busy: boolean;
  claim: string | null;
  activeFact: string | null;
  onClaim: (id: string) => void;
  onActiveFact: (id: string | null) => void;
  onSelect: (id: string) => void;
  onHolding: (index: number, value: number) => void;
  onOpacity: (id: string, value: number) => void;
  onRun: () => void;
  inputValid: boolean;
  onValidity: (valid: boolean) => void;
};
function TreeNode({
  id,
  provenance,
  depth = 0,
  active,
  onSelect,
}: {
  id: string;
  provenance: Record<string, FactNode>;
  depth?: number;
  active: string | null;
  onSelect: (id: string) => void;
}) {
  const node = provenance[id];
  const [expanded, setExpanded] = useState(depth < 2);
  if (!node) return null;
  if (depth > 25) return <p className="muted">后续证据请下载完整分析JSON。</p>;
  return (
    <div className="proof-node">
      <div className={"proof-node-heading " + (active === id ? "active" : "")}>
        <button
          className={"tree-toggle " + (expanded ? "open" : "")}
          aria-label={
            node.premises.length
              ? expanded
                ? "折叠前提"
                : "展开前提"
              : "输入事实"
          }
          aria-expanded={node.premises.length ? expanded : undefined}
          disabled={!node.premises.length}
          onClick={() => setExpanded(!expanded)}
        >
          {node.premises.length ? (
            <CaretRight size={14} />
          ) : (
            <span className="leaf-point" />
          )}
        </button>
        <button className="proof-node-label" onClick={() => onSelect(id)}>
          <span className="rule-code">{node.rule || "输入"}</span>
          <span>{node.label}</span>
        </button>
      </div>
      <p className="proof-note">
        {node.rule ? node.rule_description : node.source}
      </p>
      {expanded && node.premises.length > 0 && (
        <div className="proof-children">
          {node.premises.map((p) => (
            <TreeNode
              key={p}
              id={p}
              provenance={provenance}
              depth={depth + 1}
              active={active}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}
function Result({
  decision,
  analysis,
  names,
}: {
  decision: Decision;
  analysis: Analysis;
  names: Record<string, string>;
}) {
  const status = decision.review_required
    ? "优先复核"
    : decision.review_priority === "中"
      ? "持续关注"
      : "暂未触发复核";
  const components = analysis.parameters.method.startsWith("hybrid")
    ? [
        ["bayes", "Bayes"],
        ["fuzzy", "Mamdani"],
        ["rule", "指标规则"],
      ]
    : [
        [
          analysis.parameters.method,
          analysis.parameters.method === "bayes_only"
            ? "Bayes"
            : analysis.parameters.method === "fuzzy_only"
              ? "Mamdani"
              : "结构规则",
        ],
      ];
  return (
    <>
      <div
        className={
          "decision-status " + (decision.review_required ? "review" : "")
        }
      >
        <div>
          {decision.review_required ? (
            <WarningDiamond size={24} />
          ) : (
            <CheckCircle size={24} />
          )}
          <h3>{status}</h3>
        </div>
        <p>
          {decision.deferred_structural_alert
            ? "校准策略暂缓了结构预警的优先复核；证明仍保留，这不等于线索消失或已经排除风险。"
            : decision.lock_changed_action
              ? "评分未过阈值，结构预警仍保留复核行动。"
              : decision.review_required
                ? "当前方法触发了复核，请结合下面的事实依据。"
                : "当前输入未触发优先复核，不代表不存在实际风险。"}
        </p>
      </div>
      <section className="inspector-section">
        <div className="section-heading">
          <h3>当前评分</h3>
          <span className="subtle-label">0–1 归一化评分</span>
        </div>
        <div className="score-row">
          <strong>{decision.score.toFixed(3)}</strong>
          <span>阈值 {analysis.parameters.threshold.toFixed(2)}</span>
        </div>
        <div className="score-components">
          {components.map(([key, label]) => {
            const value =
              key in decision.components
                ? decision.components[key]
                : decision.score;
            return (
              <div className="component-row" key={key}>
                <span>{label}</span>
                <div className="component-track">
                  <span style={{ width: value * 100 + "%" }} />
                </div>
                <strong>{value.toFixed(3)}</strong>
              </div>
            );
          })}
        </div>
        <p className="helper-text">
          模糊与混合评分不是校准概率，也不是监管结论。
        </p>
      </section>
      {analysis.parameters.method.startsWith("hybrid") && (
        <section className="inspector-section">
          <div className="section-heading">
            <h3>结构约束对照</h3>
            <Info size={16} />
          </div>
          <div className="decision-compare">
            <div>
              <span>不锁定结构</span>
              <strong>
                {decision.hybrid_comparison.unlocked.review_required
                  ? "优先复核"
                  : "暂未触发复核"}
              </strong>
              <small>
                {decision.hybrid_comparison.unlocked.score.toFixed(3)}
              </small>
            </div>
            <ArrowRight size={20} />
            <div
              className={
                decision.hybrid_comparison.locked.review_required
                  ? "emphasis"
                  : ""
              }
            >
              <span>保留结构约束</span>
              <strong>
                {decision.hybrid_comparison.locked.review_required
                  ? "优先复核"
                  : "未触发"}
              </strong>
              <small>
                {decision.hybrid_comparison.locked.score.toFixed(3)}
              </small>
            </div>
          </div>
          <p className="helper-text">
            规则不改写分数，只影响最终行动。锁定也可能增加误报。
          </p>
        </section>
      )}
      <section className="inspector-section">
        <div className="section-heading">
          <h3>已知线索</h3>
          <span className="count-label">{decision.alerts.length}</span>
        </div>
        {decision.alerts.length ? (
          <div className="alert-list">
            {decision.alerts.map((a) => (
              <div key={a}>
                <WarningDiamond size={16} />
                <span>{a}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="empty-inline">本次输入尚未推出高预警线索。</p>
        )}
        {decision.controllers.length > 0 && (
          <div className="controller-line">
            <span>实控人候选</span>
            <strong>
              {decision.controllers.map((p) => names[p] || p).join("、")}
            </strong>
          </div>
        )}
      </section>
    </>
  );
}
export default function Inspector(props: Props) {
  const {
    data,
    analysis,
    selected,
    dirty,
    busy,
    claim,
    activeFact,
    onClaim,
    onActiveFact,
    onSelect,
    onHolding,
    onOpacity,
    onRun,
    inputValid,
    onValidity,
    view = "result",
    standalone = false,
  } = props;
  const tab = view;
  const [holdingIndex, setHoldingIndex] = useState(-1);
  const entity = data.entities.find((e) => e.eid === selected);
  const decision = analysis?.decisions[selected];
  const names = Object.fromEntries(data.entities.map((e) => [e.eid, e.name]));
  const incoming = data.holdings
    .map((h, i) => ({ ...h, index: i }))
    .filter((h) => h.target === selected);
  useEffect(() => {
    setHoldingIndex(data.holdings.findIndex((h) => h.target === selected));
  }, [selected, data.entities.length]);
  if (!entity)
    return (
      <aside className="inspector">
        <div className="empty-state">选择一个实体查看结论与证据。</div>
      </aside>
    );
  const holding =
    data.holdings[holdingIndex]?.target === selected
      ? data.holdings[holdingIndex]
      : incoming[0];
  const index = holding
    ? "index" in holding
      ? Number(holding.index)
      : holdingIndex
    : -1;
  const others = holding
    ? data.holdings
        .filter((h, i) => h.target === selected && i !== index)
        .reduce((sum, h) => sum + h.ratio, 0)
    : 0;
  const max = Math.max(holding?.ratio || 0, 1 - others) * 100;
  return (
    <aside
      className={
        "inspector " +
        (standalone ? "standalone-inspector " : "") +
        (busy ? "processing" : "")
      }
      aria-label="所选实体详情"
      aria-busy={busy}
    >
      <header className="entity-heading">
        <div className="entity-kicker">
          <span>{entity.eid}</span>
          <span>
            {entity.kind === "Person" ? "自然人" : entity.category || "企业"}
          </span>
        </div>
        <h2>{entity.name}</h2>
        {entity.kind === "Company" && (
          <p>注册资本 {fmt(entity.capital, 0)} 万元</p>
        )}
      </header>
      {entity.kind === "Person" ? (
        <div className="inspector-body">
          <section className="inspector-section">
            <h3>直接持股记录</h3>
            <div className="person-holdings">
              {!data.holdings.some((h) => h.holder === selected) && (
                <p className="empty-inline">
                  当前案例没有该自然人的直接持股记录。这不代表现实中没有持股。
                </p>
              )}
              {data.holdings
                .filter((h) => h.holder === selected)
                .map((h, i) => (
                  <button key={i} onClick={() => onSelect(h.target)}>
                    <span>{names[h.target] || h.target}</span>
                    <strong>{fmt(h.ratio * 100, 1)}%</strong>
                    <ArrowSquareOut size={16} />
                  </button>
                ))}
            </div>
            <p className="helper-text">
              自然人节点不计算企业风险评分。选择被投企业查看具体推理。
            </p>
          </section>
        </div>
      ) : (
        <>
          <div className="inspector-body" id={"inspector-panel-" + tab}>
            {tab === "result" &&
              (decision && analysis ? (
                <Result decision={decision} analysis={analysis} names={names} />
              ) : (
                <div className="empty-state">
                  <TreeStructure size={28} />
                  <h3>还没有分析结果</h3>
                  <p>运行推理后，这里会出现评分、复核行动和事实依据。</p>
                  <button
                    className="button primary"
                    onClick={onRun}
                    disabled={busy}
                  >
                    运行推理
                  </button>
                </div>
              ))}
            {tab === "evidence" && (
              <>
                <label className="field-label" htmlFor="claim-select">
                  选择要追溯的结论
                </label>
                <select
                  id="claim-select"
                  value={claim || ""}
                  onChange={(e) => onClaim(e.target.value)}
                  disabled={!decision || dirty || busy}
                >
                  <option value="" disabled>
                    选择一条结论
                  </option>
                  {decision?.claim_ids.map((id) => (
                    <option key={id} value={id}>
                      {analysis?.provenance[id]?.label || id}
                    </option>
                  ))}
                </select>
                <p className="helper-text">
                  展开查看规则与前提；点击证据节点后，可在关系图谱中查看对应高亮。
                </p>
                <a className="evidence-graph-link" href={graphHref(selected)}>
                  <ArrowSquareOut size={16} />
                  在关系图谱中查看证明
                </a>
                {analysis && claim ? (
                  <div className="proof-tree">
                    <TreeNode
                      key={claim}
                      id={claim}
                      provenance={analysis.provenance}
                      active={activeFact}
                      onSelect={(id) => onActiveFact(id)}
                    />
                  </div>
                ) : (
                  <div className="empty-state compact">
                    <TreeStructure size={28} />
                    <p>当前没有可追溯结论。未证明不等于现实中为假。</p>
                  </div>
                )}
              </>
            )}
            {tab === "tune" && (
              <>
                <section className="inspector-section first">
                  <div className="section-heading">
                    <h3>调整一条持股关系</h3>
                    <SlidersHorizontal size={16} />
                  </div>
                  {incoming.length > 0 ? (
                    <>
                      <label className="field-label" htmlFor="holding-edge">
                        股东 → 当前企业
                      </label>
                      <select
                        id="holding-edge"
                        value={index}
                        onChange={(e) =>
                          setHoldingIndex(Number(e.target.value))
                        }
                        disabled={busy}
                      >
                        {incoming.map((h) => (
                          <option key={h.index} value={h.index}>
                            {shortName(names[h.holder] || h.holder)} →{" "}
                            {shortName(entity.name)}
                          </option>
                        ))}
                      </select>
                      <label
                        className="field-label range-label"
                        htmlFor="holding-value"
                      >
                        持股比例{" "}
                        <strong>
                          {holding ? fmt(holding.ratio * 100, 2) : 0}%
                        </strong>
                      </label>
                      <HoldingEditor
                        key={selected + ":" + index + ":" + analysis?.id}
                        value={holding?.ratio || 0}
                        maximum={max}
                        busy={busy}
                        onChange={(value) => onHolding(index, value)}
                        onValidity={onValidity}
                      />
                      <p id="holding-help" className="helper-text">
                        其他股东合计 {fmt(others * 100, 2)}%，此项上限{" "}
                        {fmt(max, 2)}%。修改仅作用于网页草案，不会改写原始CSV。
                      </p>
                    </>
                  ) : (
                    <p className="empty-inline">
                      该企业暂无直接持股记录，可导入完整JSON案例。
                    </p>
                  )}
                </section>
                <section className="inspector-section">
                  <label
                    className="field-label range-label"
                    htmlFor="opacity-range"
                  >
                    不透明度代理 <strong>{entity.opacity.toFixed(2)}</strong>
                  </label>
                  <input
                    id="opacity-range"
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={entity.opacity}
                    disabled={busy}
                    onChange={(e) =>
                      onOpacity(entity.eid, Number(e.target.value))
                    }
                  />
                  <p className="helper-text">
                    教学设定，不是对真实企业的测量。
                  </p>
                </section>
                <p className="helper-text">
                  有效修改已保留在当前案例草稿中。返回建模任务后重新构建事实；各推理任务会提示重新运行。
                </p>
                {!inputValid && (
                  <p className="field-error">请先修正持股比例，再继续实验。</p>
                )}
              </>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
