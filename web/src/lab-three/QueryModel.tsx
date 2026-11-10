import { ArrowRight, Target } from "@phosphor-icons/react";
import type { QueryInfo } from "./types";
import { relations } from "./presentation";

export default function QueryModel({ query }: { query: QueryInfo }) {
  return (
    <div className="lab3-query-model">
      <h3>问题定义</h3>
      <p>{query.text}</p>
      {query.relations && (
        <ol className="lab3-relation-chain" aria-label="待搜索的关系序列">
          <li>{query.start_name || "起点"}</li>
          {query.relations.map((relation, index) => (
            <li key={index}>
              <ArrowRight size={14} aria-hidden="true" />
              <span>{relations[relation] || relation}</span>
            </li>
          ))}
        </ol>
      )}
      <div className="lab3-target">
        <Target size={18} aria-hidden="true" />
        <span>
          目标：{query.sector || "指定行业"}公司
          {query.listed == null ? "" : query.listed ? " · 已上市" : " · 未上市"}
          {" · "}
          {query.hops} 跳
        </span>
      </div>
      <details className="lab3-model-details">
        <summary>状态、动作与终止条件</summary>
        <dl>
          <div>
            <dt>状态</dt>
            <dd>当前实体、已访问实体与关系路径。</dd>
          </div>
          <div>
            <dt>动作</dt>
            <dd>沿下一种指定关系扩展，不重复访问实体。</dd>
          </div>
          <div>
            <dt>转移</dt>
            <dd>
              追加实体与边；专家约束启用时，证据须已核验、有效且置信度 ≥ 0.55。
            </dd>
          </div>
          <div>
            <dt>终止</dt>
            <dd>到达指定跳数且终点属性匹配；预算耗尽或候选穷尽也会停止。</dd>
          </div>
        </dl>
      </details>
    </div>
  );
}
