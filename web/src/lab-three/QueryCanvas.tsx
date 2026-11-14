import { displayText } from "../display-text";
import { ArrowRight, Crosshair } from "@phosphor-icons/react";
import type { QueryInfo } from "./types";
import { relations } from "./presentation";

/** A pre-run view of public query constraints, never a fabricated search result. */
export default function QueryCanvas({ query }: { query: QueryInfo }) {
  return (
    <div className="lab3-query-canvas">
      <div className="lab3-canvas-caption">
        <span>问题结构</span>
        <span>尚未运行</span>
      </div>
      <div
        className="lab3-blueprint-scroll"
        role="region"
        aria-label="待求解的问题结构，可横向滚动"
        tabIndex={0}
      >
        <div className="lab3-blueprint">
          <article className="lab3-blueprint-node start">
            <small>起始实体</small>
            <strong>{displayText(query.start_name) || "指定起点"}</strong>
            <span>深度 0</span>
          </article>
          <div className="lab3-blueprint-connection">
            <ArrowRight size={18} />
          </div>
          <article className="lab3-blueprint-node sequence">
            <small>待搜索的关系序列</small>
            <ol>
              {query.relations?.map((relation, index) => (
                <li key={index}>
                  <em>{String(index + 1).padStart(2, "0")}</em>
                  {relations[relation] || relation}
                </li>
              ))}
            </ol>
          </article>
          <div className="lab3-blueprint-connection">
            <ArrowRight size={18} />
          </div>
          <article className="lab3-blueprint-node target">
            <small>目标约束</small>
            <strong>{query.sector || "指定行业"}公司</strong>
            <span>
              {query.listed == null
                ? "上市状态不限"
                : query.listed
                  ? "已上市"
                  : "未上市"}{" "}
              · {query.hops} 跳
            </span>
          </article>
        </div>
      </div>
      <div className="lab3-canvas-prompt">
        <Crosshair size={16} />
        <span>运行路径搜索后，在此查看候选分支与证据。</span>
      </div>
      <div className="lab3-canvas-status">
        <span>
          {query.nodes} 个实体 · {query.edges} 条关系
        </span>
        <span>仅显示问题约束</span>
      </div>
    </div>
  );
}
