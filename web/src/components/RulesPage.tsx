import ExperimentBreadcrumb from "./ExperimentBreadcrumb";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  CaretDown,
  CaretUp,
  MagnifyingGlass,
  TreeStructure,
} from "@phosphor-icons/react";
import type { CaseData, RuleItem } from "../types";
import { holdingExample } from "../rule-notation";
import RuleNotation from "./RuleNotation";
import RuleExpression from "./RuleExpression";
import { request } from "../api";
export default function RulesPage({
  embedded = false,
  caseData = null,
}: {
  embedded?: boolean;
  caseData?: CaseData | null;
}) {
  const Shell = embedded ? "section" : "main";
  const contentId = useId();
  const [expanded, setExpanded] = useState(!embedded);
  const [rules, setRules] = useState<RuleItem[]>([]);
  const [search, setSearch] = useState("");
  const [group, setGroup] = useState("全部");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  const pending = useRef<AbortController | null>(null);
  const load = async () => {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true);
    setError("");
    try {
      const value = await request<{ items: RuleItem[] }>("/api/rules", {
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setRules(value.items);
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : "规则读取失败");
    } finally {
      if (pending.current === controller) setBusy(false);
    }
  };
  useEffect(() => {
    void load();
    return () => pending.current?.abort();
  }, []);
  const groups = ["全部", ...new Set(rules.map((r) => r.group))];
  const filtered = useMemo(
    () =>
      rules.filter(
        (r) =>
          (group === "全部" || r.group === group) &&
          [
            r.name,
            r.description,
            r.consequent,
            ...r.antecedents,
            ...(r.guard_conditions || []),
          ]
            .join(" ")
            .toLowerCase()
            .includes(search.toLowerCase().trim()),
      ),
    [rules, search, group],
  );
  const example = useMemo(
    () =>
      holdingExample(
        rules.find((rule) => rule.name === "R01"),
        caseData?.holdings.map((row) => [
          "Holds",
          row.holder,
          row.target,
          row.ratio,
        ]) || [],
        caseData?.entities || [],
      ),
    [rules, caseData],
  );
  return (
    <Shell
      id={embedded ? undefined : "main-content"}
      tabIndex={-1}
      className={`page-shell rules-page${embedded && !expanded ? " rules-page-collapsed" : ""}`}
    >
      {!embedded && (
        <ExperimentBreadcrumb moduleId="modeling" title="规则知识库" />
      )}
      <header className="page-header">
        <div>
          {embedded ? <h2>一阶规则知识库</h2> : <h1>规则知识库</h1>}
          <p>
            读取当前 Python
            规则的前提、完整触发条件与结论，和具体案例事实分开展示。
          </p>
        </div>
        <div className="rules-header-actions">
          <span className="data-badge">{rules.length} 条结构规则</span>
          {embedded && (
            <button
              type="button"
              className="button secondary rules-toggle"
              aria-expanded={expanded}
              aria-controls={contentId}
              aria-label={`${expanded ? "收起" : "展开"}一阶规则知识库`}
              onClick={() => setExpanded((value) => !value)}
            >
              {expanded ? "收起" : "展开"}
              {expanded ? (
                <CaretUp size={16} aria-hidden="true" />
              ) : (
                <CaretDown size={16} aria-hidden="true" />
              )}
            </button>
          )}
        </div>
      </header>
      <div
        id={contentId}
        className="rules-content"
        hidden={embedded && !expanded}
      >
        <RuleNotation example={example} dataReady={caseData !== null} />
        <div className="rules-toolbar">
          <label className="search-box">
            <MagnifyingGlass size={18} />
            <span className="sr-only">搜索规则编号、说明或谓词</span>
            <input
              placeholder="搜索规则编号、说明或谓词"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </label>
          <label className="group-select">
            <span>规则类别</span>
            <select value={group} onChange={(e) => setGroup(e.target.value)}>
              {groups.map((g) => (
                <option key={g}>{g}</option>
              ))}
            </select>
          </label>
        </div>
        {error && (
          <div role="alert" className="error-banner">
            {error}
            <button className="text-button" onClick={() => void load()}>
              重新读取
            </button>
          </div>
        )}
        {busy ? (
          <div className="skeleton rule-skeleton" aria-label="正在读取规则" />
        ) : filtered.length ? (
          <div className="rules-list">
            {filtered.map((rule) => (
              <article className="rule-row" key={rule.name}>
                <div className="rule-row-code">{rule.name}</div>
                <div className="rule-content">
                  <header>
                    <h2>{rule.description}</h2>
                    <span>{rule.group}</span>
                  </header>
                  <RuleExpression
                    antecedents={rule.antecedents}
                    conclusions={[rule.consequent]}
                    guardConditions={rule.guard_conditions}
                    guardComplete={
                      !rule.has_guard || rule.guard_complete === true
                    }
                  />
                  {rule.has_guard && !rule.guard_complete && (
                    <p className="guard-note">
                      附加条件未能展开，不能将上式理解为无条件规则；请核对
                      Python 规则实现。
                    </p>
                  )}
                  {rule.note && (
                    <p className="rule-boundary">
                      <strong>适用边界</strong>
                      {rule.note}
                    </p>
                  )}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-state large">
            <TreeStructure size={36} />
            <h2>没有匹配规则</h2>
            <p>换个关键词，或查看全部类别。</p>
            <button
              className="button secondary"
              onClick={() => {
                setGroup("全部");
                setSearch("");
              }}
            >
              清除筛选
            </button>
          </div>
        )}
      </div>
    </Shell>
  );
}
