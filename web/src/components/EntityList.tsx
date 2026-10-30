import { useState } from "react";
import { MagnifyingGlass, Buildings, User } from "@phosphor-icons/react";
import type { Analysis, CaseData } from "../types";
import { shortName } from "../api";
export default function EntityList({
  data,
  analysis,
  selected,
  onSelect,
  dirty,
}: {
  data: CaseData;
  analysis: Analysis | null;
  selected: string;
  onSelect: (id: string) => void;
  dirty: boolean;
}) {
  const [kind, setKind] = useState<"Company" | "Person">("Company");
  const [search, setSearch] = useState("");
  const nodes = data.entities.filter(
    (e) =>
      e.kind === kind &&
      (!search ||
        [e.name, e.eid].some((s) =>
          s.toLowerCase().includes(search.toLowerCase().trim()),
        )),
  );
  return (
    <aside className="entity-list" aria-label="实体列表">
      <div className="entity-list-header">
        <div className="entity-kind" aria-label="实体类型">
          {(["Company", "Person"] as const).map((k) => (
            <button
              key={k}
              onClick={() => setKind(k)}
              aria-pressed={kind === k}
              className={kind === k ? "active" : ""}
            >
              {k === "Company" ? "企业" : "自然人"}
              <span>{data.entities.filter((e) => e.kind === k).length}</span>
            </button>
          ))}
        </div>
        <label className="search-box">
          <MagnifyingGlass size={17} />
          <span className="sr-only">搜索实体名称或编号</span>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索名称或编号"
          />
        </label>
      </div>
      <div className="entity-scroll">
        {nodes.map((entity) => {
          const decision = analysis?.decisions[entity.eid];
          return (
            <button
              key={entity.eid}
              className={
                "entity-row " + (selected === entity.eid ? "selected" : "")
              }
              aria-pressed={selected === entity.eid}
              onClick={() => onSelect(entity.eid)}
              title={entity.name}
            >
              <span className="entity-icon">
                {kind === "Company" ? (
                  <Buildings size={19} />
                ) : (
                  <User size={19} />
                )}
              </span>
              <span className="entity-row-text">
                <strong>{shortName(entity.name)}</strong>
                <small>
                  {entity.eid}
                  {entity.category === "上市公司" ? " · 上市公司" : ""}
                </small>
              </span>
              {decision && !dirty && decision.review_required && (
                <span
                  className="review-mark"
                  title="触发优先复核"
                  aria-label="触发优先复核"
                />
              )}
            </button>
          );
        })}
        {!nodes.length && (
          <div className="empty-state compact">
            <MagnifyingGlass size={24} />
            <p>没有匹配的实体</p>
            <button className="text-button" onClick={() => setSearch("")}>
              清除搜索
            </button>
          </div>
        )}
      </div>
      <div className="entity-list-footer">实体即入口，点击查看依据。</div>
    </aside>
  );
}
