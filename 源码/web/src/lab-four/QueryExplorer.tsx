import { useEffect, useMemo, useRef, useState } from "react";
import {
  CheckCircle,
  XCircle,
  MagnifyingGlass,
  CaretDown,
  CaretUp,
} from "@phosphor-icons/react";
import { classColor, percent } from "./charts";
import type { EpisodeResult } from "./types";

type Filter = "all" | "incorrect" | "correct";
export default function QueryExplorer({ result }: { result: EpisodeResult }) {
  const inspector = useRef<HTMLElement>(null);
  const queryList = useRef<HTMLDivElement>(null);
  function inspectSample(id: string) {
    setSelected(id);
    if (window.matchMedia("(max-width: 960px)").matches) {
      requestAnimationFrame(() => {
        inspector.current?.focus({ preventScroll: true });
        inspector.current?.scrollIntoView({
          block: "start",
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)")
            .matches
            ? "instant"
            : "smooth",
        });
      });
    }
  }
  const [filter, setFilter] = useState<Filter>("all");
  const [classFilter, setClassFilter] = useState("all");
  const [sort, setSort] = useState("original");
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState(result.queries[0]?.id);
  const names = new Map(result.classes.map((c) => [c.id, c.name]));
  useEffect(() => {
    setFilter("all");
    setClassFilter("all");
    setSort("original");
    setExpanded(false);
    setSelected(result.queries[0]?.id);
  }, [result]);
  const filtered = useMemo(() => {
    const rows = result.queries.filter(
      (q) =>
        (filter === "all" ||
          (q.predicted_id === q.target_id) === (filter === "correct")) &&
        (classFilter === "all" || q.target_id === classFilter),
    );
    return sort === "confidence"
      ? rows.sort((a, b) => a.confidence - b.confidence)
      : rows;
  }, [result, filter, classFilter, sort]);
  const visible = expanded ? filtered : filtered.slice(0, 12);
  // Keep the selected detail represented by an actual visible tile after
  // collapsing, changing class filters or sorting. Otherwise mobile back
  // navigation cannot restore a focus target for a hidden selected image.
  const chosen = visible.find((q) => q.id === selected) || visible[0];
  const isCorrect = chosen?.predicted_id === chosen?.target_id;
  const confusion = result.classes.map((actual) =>
    result.classes.map(
      (predicted) =>
        result.queries.filter(
          (q) => q.target_id === actual.id && q.predicted_id === predicted.id,
        ).length,
    ),
  );
  return (
    <section className="lab4-query-section" aria-labelledby="lab4-query-title">
      <header className="lab4-section-title">
        <div>
          <span className="lab4-section-kicker">QUERY · 待分类图片</span>
          <h2 id="lab4-query-title">看看模型认对了什么</h2>
          <p>点击图片检查五个类别的分数。真实标签只在预测完成后用于核对。</p>
        </div>
        <span>{result.total} 张查询图片</span>
      </header>
      <div className="lab4-query-toolbar">
        <div className="lab4-segmented" role="group" aria-label="筛选预测结果">
          {(
            [
              ["all", "全部", result.total],
              ["incorrect", "误判", result.total - result.correct],
              ["correct", "正确", result.correct],
            ] as const
          ).map(([id, label, count]) => (
            <button
              key={id}
              type="button"
              aria-pressed={filter === id}
              onClick={() => {
                setFilter(id);
                setExpanded(false);
              }}
            >
              {label}
              <span>{count}</span>
            </button>
          ))}
        </div>
        <div className="lab4-query-selects">
          <label className="sr-only" htmlFor="lab4-class-filter">
            按真实类别筛选
          </label>
          <select
            id="lab4-class-filter"
            value={classFilter}
            onChange={(e) => {
              setClassFilter(e.target.value);
              setExpanded(false);
            }}
          >
            <option value="all">所有真实类别</option>
            {result.classes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
          <label className="sr-only" htmlFor="lab4-query-sort">
            查询图片排序
          </label>
          <select
            id="lab4-query-sort"
            value={sort}
            onChange={(e) => setSort(e.target.value)}
          >
            <option value="original">原始顺序</option>
            <option value="confidence">低分优先</option>
          </select>
        </div>
      </div>
      <p className="lab4-filter-status" role="status">
        显示 {Math.min(visible.length, filtered.length)} / {filtered.length} 张
        {filter === "incorrect" ? "误判" : ""}图片
      </p>
      <div className="lab4-query-explorer" ref={queryList}>
        <div>
          {visible.length ? (
            <div className="lab4-query-grid">
              {visible.map((sample) => {
                const correct = sample.predicted_id === sample.target_id;
                const index =
                  result.queries.findIndex((q) => q.id === sample.id) + 1;
                return (
                  <button
                    type="button"
                    key={sample.id}
                    className={`lab4-query-tile ${correct ? "correct" : "incorrect"}`}
                    aria-pressed={chosen?.id === sample.id}
                    aria-label={`${String(index).padStart(2, "0")} ${names.get(sample.predicted_id)} ${correct ? "预测正确" : "预测错误"} ${percent(sample.confidence, 0)}，查看类别分数`}
                    onClick={() => inspectSample(sample.id)}
                  >
                    <div className="lab4-query-image">
                      <img
                        src={sample.image}
                        alt={`查询图片 ${index}`}
                        width={84}
                        height={84}
                        loading="lazy"
                      />
                      <span className="lab4-tile-number">
                        {String(index).padStart(2, "0")}
                      </span>
                    </div>
                    <span className="lab4-tile-label">
                      {correct ? (
                        <CheckCircle size={15} weight="fill" />
                      ) : (
                        <XCircle size={15} weight="fill" />
                      )}
                      {names.get(sample.predicted_id)}
                    </span>
                    <span className="lab4-tile-meta">
                      {correct ? "预测正确" : "预测错误"}
                      <b>{percent(sample.confidence, 0)}</b>
                    </span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="lab4-filter-empty">
              <MagnifyingGlass size={28} />
              <h3>没有符合条件的图片</h3>
              <p>试试其他类别，或清除筛选查看全部预测。</p>
              <button
                className="button secondary"
                onClick={() => {
                  setFilter("all");
                  setClassFilter("all");
                }}
              >
                清除筛选
              </button>
            </div>
          )}
          {filtered.length > 12 && (
            <button
              className="lab4-expand"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? <CaretUp size={16} /> : <CaretDown size={16} />}
              {expanded ? "收起图片" : `展开其余 ${filtered.length - 12} 张`}
            </button>
          )}
        </div>
        {chosen && (
          <aside
            className="lab4-inspector"
            ref={inspector}
            tabIndex={-1}
            aria-label="选中图片的预测详情"
          >
            <button
              type="button"
              className="lab4-inspector-back"
              onClick={() => {
                queryList.current?.scrollIntoView({
                  block: "start",
                  behavior: "instant",
                });
                queryList.current
                  ?.querySelector<HTMLButtonElement>(
                    "button[aria-pressed=true]",
                  )
                  ?.focus({ preventScroll: true });
              }}
            >
              返回图片列表
            </button>
            <div className="lab4-inspector-heading">
              <span>样本详情</span>
              <span className={isCorrect ? "lab4-positive" : "lab4-negative"}>
                {isCorrect ? "预测正确" : "预测错误"}
              </span>
            </div>
            <img
              className="lab4-inspector-image"
              src={chosen.image}
              alt="当前选中的查询图片"
              width={168}
              height={168}
            />
            <dl className="lab4-inspector-labels">
              <div>
                <dt>预测</dt>
                <dd>{names.get(chosen.predicted_id)}</dd>
              </div>
              <div>
                <dt>真实标签</dt>
                <dd>{names.get(chosen.target_id)}</dd>
              </div>
            </dl>
            <h3>类别分数</h3>
            <ol className="lab4-probabilities">
              {result.classes
                .map((c, i) => ({
                  ...c,
                  probability: chosen.probabilities[i],
                  index: i,
                }))
                .sort((a, b) => b.probability - a.probability)
                .map((c) => (
                  <li key={c.id}>
                    <div>
                      <span>
                        {c.name}
                        {c.id === chosen.target_id && <small>真实</small>}
                      </span>
                      <b>{percent(c.probability)}</b>
                    </div>
                    <div className="lab4-probability-track">
                      <span
                        style={{
                          width: `${c.probability * 100}%`,
                          background: classColor(c.index),
                        }}
                      />
                    </div>
                  </li>
                ))}
            </ol>
            <p className="lab4-footnote">
              分数反映模型的相对偏好，不是预测正确的保证。
              {result.decision_rule === "balanced_capacity" &&
                "本轮使用整批均衡分配，预测类别可能不是单图分数最高项。"}
            </p>
            <details className="lab4-sample-id">
              <summary>样本来源</summary>
              <code>{chosen.id}</code>
            </details>
          </aside>
        )}
      </div>
      <details className="lab4-confusion">
        <summary>
          按类别查看混淆矩阵 <span>行是真实类别，列是预测类别</span>
        </summary>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="混淆矩阵，可横向滚动"
        >
          <table>
            <caption className="sr-only">
              本次任务混淆矩阵，行是真实类别，列是预测类别
            </caption>
            <thead>
              <tr>
                <th scope="col">真实 / 预测</th>
                {result.classes.map((c) => (
                  <th scope="col" key={c.id}>
                    {c.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.classes.map((c, i) => (
                <tr key={c.id}>
                  <th scope="row">{c.name}</th>
                  {confusion[i].map((n, j) => (
                    <td
                      key={j}
                      className={
                        i === j
                          ? "lab4-confusion-correct"
                          : n
                            ? "lab4-confusion-error"
                            : ""
                      }
                    >
                      {n}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}
