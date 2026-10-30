import { useMemo, useState } from "react";
import {
  ArrowRight,
  Buildings,
  User,
  MagnifyingGlass,
  Graph,
  Funnel,
  X,
} from "@phosphor-icons/react";
import type { Analysis, CaseData } from "../types";
import { entityHref, graphHref } from "../routes";
export default function CaseOverview({
  data,
  analysis,
  dirty,
}: {
  data: CaseData;
  analysis: Analysis | null;
  dirty: boolean;
}) {
  const [kind, setKind] = useState<"Company" | "Person">("Company");
  const [search, setSearch] = useState("");
  const [reviewOnly, setReviewOnly] = useState(false);
  const entities = useMemo(
    () =>
      data.entities.filter(
        (e) =>
          e.kind === kind &&
          (!reviewOnly ||
            dirty ||
            kind === "Person" ||
            analysis?.decisions[e.eid]?.review_required) &&
          [e.eid, e.name].some((value) =>
            value.toLowerCase().includes(search.toLowerCase().trim()),
          ),
      ),
    [data, kind, search, reviewOnly, analysis, dirty],
  );
  const companies = data.entities.filter((e) => e.kind === "Company");
  const alerts = companies.filter(
    (e) => analysis?.decisions[e.eid]?.review_required,
  ).length;
  return (
    <>
      <section className="case-summary" aria-label="当前案例概览">
        <div className="case-summary-main">
          <span className="case-summary-symbol">
            <Buildings size={26} />
          </span>
          <div>
            <h2>{data.label}</h2>
            <p>虚构教学案例 · 从企业目录开始，逐项查看结论和依据。</p>
          </div>
        </div>
        <dl>
          <div>
            <dt>企业</dt>
            <dd>{companies.length}</dd>
          </div>
          <div>
            <dt>自然人</dt>
            <dd>{data.entities.length - companies.length}</dd>
          </div>
          <div>
            <dt>持股关系</dt>
            <dd>{data.holdings.length}</dd>
          </div>
          <div>
            <dt>优先复核</dt>
            <dd>{dirty ? "待更新" : alerts}</dd>
          </div>
        </dl>
      </section>
      <section
        className={"directory" + (kind === "Person" ? " person-directory" : "")}
        aria-label="案例实体目录"
      >
        <header className="directory-toolbar">
          <div className="directory-kind" aria-label="实体类型">
            <button
              className={kind === "Company" ? "active" : ""}
              aria-pressed={kind === "Company"}
              onClick={() => setKind("Company")}
            >
              <Buildings size={17} />
              企业目录<span>{companies.length}</span>
            </button>
            <button
              className={kind === "Person" ? "active" : ""}
              aria-pressed={kind === "Person"}
              onClick={() => setKind("Person")}
            >
              <User size={17} />
              自然人<span>{data.entities.length - companies.length}</span>
            </button>
          </div>
          <div className="directory-search-tools">
            {kind === "Company" && (
              <button
                className={"directory-filter " + (reviewOnly ? "active" : "")}
                aria-pressed={reviewOnly}
                disabled={dirty || !analysis}
                onClick={() => setReviewOnly((old) => !old)}
              >
                <Funnel size={16} />
                只看优先复核
              </button>
            )}
            <label className="search-box">
              <MagnifyingGlass size={18} />
              <span className="sr-only">搜索企业或自然人</span>
              <input
                aria-label="搜索企业或自然人"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="搜索名称或编号"
              />
            </label>
            {search && (
              <button
                className="icon-button search-clear"
                aria-label="清空目录搜索"
                onClick={() => setSearch("")}
              >
                <X size={16} />
              </button>
            )}
          </div>
        </header>
        <div className="directory-columns" aria-hidden="true">
          <span>{kind === "Company" ? "企业名称" : "姓名"}</span>
          <span>类型</span>
          {kind === "Company" && <span>复核建议</span>}
          <span>操作</span>
        </div>
        <div className="directory-rows">
          {entities.map((entity) => {
            const decision = analysis?.decisions[entity.eid];
            return (
              <a
                className="directory-row"
                key={entity.eid}
                href={entityHref(entity.eid)}
                aria-label={
                  "查看" +
                  entity.eid +
                  (entity.kind === "Person" ? "持股" : "分析")
                }
                aria-description={
                  entity.name +
                  "，" +
                  (entity.kind === "Person"
                    ? "直接持股记录"
                    : dirty
                      ? "待重新推理"
                      : decision?.review_required
                        ? "优先复核"
                        : "查看详情")
                }
              >
                <div className="directory-entity">
                  <span className="directory-avatar">
                    {entity.kind === "Company" ? (
                      <Buildings size={20} />
                    ) : (
                      <User size={20} />
                    )}
                  </span>
                  <div>
                    <strong>{entity.name}</strong>
                    <small>{entity.eid}</small>
                  </div>
                </div>
                <span className="directory-category">
                  {entity.kind === "Person"
                    ? "自然人"
                    : entity.category || "企业"}
                </span>
                {entity.kind === "Company" && (
                  <span
                    className={
                      "directory-status " +
                      (decision?.review_required && !dirty ? "review" : "")
                    }
                  >
                    {dirty
                      ? "待重新推理"
                      : decision?.review_required
                        ? "优先复核"
                        : decision?.review_priority === "中"
                          ? "持续关注"
                          : "暂未触发复核"}
                  </span>
                )}
                <span className="directory-open">
                  {entity.kind === "Person" ? "查看持股" : "查看分析"}
                  <ArrowRight size={17} />
                </span>
              </a>
            );
          })}
        </div>
        {entities.length === 0 && (
          <div className="directory-empty">
            <MagnifyingGlass size={28} />
            <h3>没有匹配的实体</h3>
            <p>
              {reviewOnly
                ? "当前筛选下没有匹配项，可清除搜索和筛选查看完整目录。"
                : "换一个名称或编号试试。"}
            </p>
            <button
              className="button secondary"
              onClick={() => {
                setSearch("");
                setReviewOnly(false);
              }}
            >
              清除搜索
            </button>
          </div>
        )}
        <footer className="directory-footer">
          <span role="status" aria-live="polite">
            共 {entities.length} 个{kind === "Company" ? "企业" : "自然人"} ·
            点击条目查看详情；完整关系可在图谱页查看。
          </span>
          <a href={graphHref()}>
            <Graph size={16} />
            打开完整关系图谱
            <ArrowRight size={15} />
          </a>
        </footer>
      </section>
    </>
  );
}
