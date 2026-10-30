import { useMemo, useState } from "react";
import {
  ArrowRight,
  Buildings,
  User,
  MagnifyingGlass,
  X,
} from "@phosphor-icons/react";
import type { CaseData } from "../types";
import { caseHref } from "../routes";
export default function CaseOverview({ data }: { data: CaseData }) {
  const [kind, setKind] = useState<"Company" | "Person">("Company");
  const [search, setSearch] = useState("");
  const entities = useMemo(
    () =>
      data.entities.filter(
        (e) =>
          e.kind === kind &&
          [e.eid, e.name].some((value) =>
            value.toLowerCase().includes(search.toLowerCase().trim()),
          ),
      ),
    [data, kind, search],
  );
  const companies = data.entities.filter((e) => e.kind === "Company").length;
  const relations =
    data.holdings.length +
    data.guarantees.length +
    data.trades.length +
    data.kinships.length;
  return (
    <>
      <section className="case-summary" aria-label="共享案例数据概览">
        <div className="case-summary-main">
          <span className="case-summary-symbol">
            <Buildings size={26} />
          </span>
          <div>
            <h2>{data.label}</h2>
            <p>共享教学数据 · 浏览实体资料与输入关系，不包含实验推理结果。</p>
          </div>
        </div>
        <dl>
          <div>
            <dt>企业</dt>
            <dd>{companies}</dd>
          </div>
          <div>
            <dt>自然人</dt>
            <dd>{data.entities.length - companies}</dd>
          </div>
          <div>
            <dt>持股记录</dt>
            <dd>{data.holdings.length}</dd>
          </div>
          <div>
            <dt>关系记录</dt>
            <dd>{relations}</dd>
          </div>
        </dl>
      </section>
      <section className="directory source-directory" aria-label="案例实体目录">
        <header className="directory-toolbar">
          <div className="directory-kind" aria-label="实体类型">
            <button
              className={kind === "Company" ? "active" : ""}
              aria-pressed={kind === "Company"}
              onClick={() => setKind("Company")}
            >
              <Buildings size={17} />
              企业目录<span>{companies}</span>
            </button>
            <button
              className={kind === "Person" ? "active" : ""}
              aria-pressed={kind === "Person"}
              onClick={() => setKind("Person")}
            >
              <User size={17} />
              自然人<span>{data.entities.length - companies}</span>
            </button>
          </div>
          <div className="directory-search-tools">
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
          <span>操作</span>
        </div>
        <div className="directory-rows">
          {entities.map((entity) => (
            <a
              className="directory-row"
              key={entity.eid}
              href={caseHref(entity.eid)}
              aria-label={"查看" + entity.eid + "资料"}
              aria-description={entity.name + "，原始实体资料与关系记录"}
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
              <span className="directory-open">
                查看资料
                <ArrowRight size={17} />
              </span>
            </a>
          ))}
        </div>
        {entities.length === 0 && (
          <div className="directory-empty">
            <MagnifyingGlass size={28} />
            <h3>没有匹配的实体</h3>
            <p>换一个名称或编号试试。</p>
            <button className="button secondary" onClick={() => setSearch("")}>
              清除搜索
            </button>
          </div>
        )}
        <footer className="directory-footer">
          <span role="status" aria-live="polite">
            共 {entities.length} 个{kind === "Company" ? "企业" : "自然人"} ·
            点击查看资料与输入关系。
          </span>
        </footer>
      </section>
    </>
  );
}
