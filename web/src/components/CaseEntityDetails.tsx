import { useMemo } from "react";
import { ArrowRight, Graph } from "@phosphor-icons/react";
import type { CaseData } from "../types";
import { caseHref, graphHref } from "../routes";
const percent = (value: number) =>
  (value * 100).toLocaleString("zh-CN", { maximumFractionDigits: 4 }) + "%";
const money = (value: number) =>
  value.toLocaleString("zh-CN", { maximumFractionDigits: 4 }) + " 万元";
export default function CaseEntityDetails({
  data,
  entityId,
}: {
  data: CaseData;
  entityId: string;
}) {
  const entity = data.entities.find((item) => item.eid === entityId);
  const names = useMemo(
    () =>
      Object.fromEntries(data.entities.map((item) => [item.eid, item.name])),
    [data],
  );
  const rows = useMemo(
    () =>
      [
        ...data.holdings.map((row, index) => ({
          id: "holding-" + index,
          kind: "持股",
          from: row.holder,
          to: row.target,
          value: percent(row.ratio),
          note:
            "披露日期：" +
            (row.since || "未提供") +
            (row.layered ? " · 多层持股输入标记" : ""),
        })),
        ...data.guarantees.map((row, index) => ({
          id: "guarantee-" + index,
          kind: "担保",
          from: row.guarantor,
          to: row.beneficiary,
          value: money(row.amount),
          note: "担保方 → 被担保方",
        })),
        ...data.trades.map((row, index) => ({
          id: "trade-" + index,
          kind: "交易",
          from: row.seller,
          to: row.buyer,
          value: money(row.amount),
          note: "卖方 → 买方",
        })),
        ...data.kinships.map((row, index) => ({
          id: "kinship-" + index,
          kind: "亲属",
          from: row.a,
          to: row.b,
          value: row.relation,
          note: row.evidence
            ? "输入含证据标记，非真实性认证"
            : "输入未标记证据",
        })),
      ].filter((row) => row.from === entityId || row.to === entityId),
    [data, entityId],
  );
  if (!entity) return null;
  return (
    <div className="source-entity-details">
      <section
        className="task-section source-entity-profile"
        aria-label="原始实体资料"
      >
        <header>
          <h2>{entity.name}</h2>
          <p>
            {entity.eid} ·{" "}
            {entity.kind === "Person" ? "自然人" : entity.category || "企业"}
          </p>
        </header>
        <dl className="source-profile-fields">
          <div>
            <dt>实体编号</dt>
            <dd>{entity.eid}</dd>
          </div>
          <div>
            <dt>实体类型</dt>
            <dd>{entity.kind === "Person" ? "自然人" : "企业"}</dd>
          </div>
          {entity.kind === "Company" && (
            <>
              <div>
                <dt>注册资本</dt>
                <dd>{money(entity.capital)}</dd>
              </div>
              <div>
                <dt>披露完整度输入</dt>
                <dd>{percent(entity.disclosure)}</dd>
              </div>
            </>
          )}
        </dl>
        <a className="source-inline-link" href={graphHref(entity.eid)}>
          <Graph size={17} />
          查看该实体的输入关系图
          <ArrowRight size={16} />
        </a>
      </section>
      <section
        className="task-section source-relations"
        aria-label="原始关系记录"
      >
        <header>
          <h2>与该实体相关的输入关系</h2>
          <p>
            {rows.length} 条记录 ·
            持股比例、金额与亲属关系均来自当前案例输入，不含控制认定或推理结论。
          </p>
        </header>
        {rows.length ? (
          <div className="table-scroll">
            <table aria-label="实体原始关系记录">
              <thead>
                <tr>
                  <th>关系</th>
                  <th>主体</th>
                  <th>关联实体</th>
                  <th>比例／金额／关系值</th>
                  <th>输入说明</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id}>
                    <td>{row.kind}</td>
                    <td>
                      <a href={caseHref(row.from)}>
                        {names[row.from]}
                        <small>{row.from}</small>
                      </a>
                    </td>
                    <td>
                      <a href={caseHref(row.to)}>
                        {names[row.to]}
                        <small>{row.to}</small>
                      </a>
                    </td>
                    <td>{row.value}</td>
                    <td>{row.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="source-empty">
            当前案例没有与该实体相连的关系记录，可选择其他实体或查看完整图谱。
          </p>
        )}
      </section>
    </div>
  );
}
