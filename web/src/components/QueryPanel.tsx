import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  MagnifyingGlass,
  CheckCircle,
  Question,
} from "@phosphor-icons/react";
import type { Analysis, CaseData, QueryResult, ControlQuery } from "../types";
import { request } from "../api";
export default function QueryPanel({
  data,
  analysis,
  dirty,
  initialQuery,
}: {
  data: CaseData;
  analysis: Analysis | null;
  dirty: boolean;
  initialQuery?: ControlQuery | null;
}) {
  const pending = useRef<AbortController | null>(null);
  useEffect(() => {
    pending.current?.abort();
    setBusy(false);
    setResult(null);
    setError("");
    return () => pending.current?.abort();
  }, [analysis?.id, dirty]);
  const [source, setSource] = useState(initialQuery?.source || "");
  const [target, setTarget] = useState(initialQuery?.target || "");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    setSource((old) =>
      data.entities.some((e) => e.eid === old)
        ? old
        : data.entities.find((e) => e.eid === initialQuery?.source)?.eid ||
          data.entities.find((e) => e.eid === "P1")?.eid ||
          data.entities[0]?.eid ||
          "",
    );
    setTarget((old) =>
      data.entities.some((e) => e.eid === old && e.kind === "Company")
        ? old
        : data.entities.find(
            (e) => e.eid === initialQuery?.target && e.kind === "Company",
          )?.eid ||
          data.entities.find((e) => e.eid === "C3" && e.kind === "Company")
            ?.eid ||
          data.entities.find((e) => e.kind === "Company")?.eid ||
          "",
    );
    setResult(null);
    setError("");
  }, [data.entities, initialQuery?.source, initialQuery?.target]);
  async function query() {
    if (!analysis || dirty || busy) return;
    const controller = new AbortController();
    pending.current?.abort();
    pending.current = controller;
    setBusy(true);
    setError("");
    try {
      const response = await request<QueryResult>("/api/query", {
        method: "POST",
        signal: controller.signal,
        body: JSON.stringify({
          analysis_id: analysis.id,
          predicate: "Controls",
          arguments: [source, target],
        }),
      });
      if (!controller.signal.aborted) setResult(response);
    } catch (e) {
      if (controller.signal.aborted) return;
      setError(e instanceof Error ? e.message : "查询失败");
    } finally {
      if (pending.current === controller) setBusy(false);
    }
  }
  return (
    <section className="query-section">
      <header>
        <div>
          <h2>验证一个控制结论</h2>
          <p>从目标反向寻找依据，并与正向推理结果核对。</p>
        </div>
        <span className="subtle-label">目标驱动 · 表格化反向推理</span>
      </header>
      <form
        className="query-form"
        onSubmit={(e) => {
          e.preventDefault();
          void query();
        }}
      >
        <label>
          <span>控制主体</span>
          <select
            value={source}
            onChange={(e) => {
              setSource(e.target.value);
              setResult(null);
            }}
            disabled={busy}
          >
            {data.entities.map((e) => (
              <option key={e.eid} value={e.eid}>
                {e.eid} · {e.name}
              </option>
            ))}
          </select>
        </label>
        <span className="query-relation">
          是否控制 <ArrowRight size={18} />
        </span>
        <label>
          <span>目标企业</span>
          <select
            value={target}
            onChange={(e) => {
              setTarget(e.target.value);
              setResult(null);
            }}
            disabled={busy}
          >
            {data.entities
              .filter((e) => e.kind === "Company")
              .map((e) => (
                <option key={e.eid} value={e.eid}>
                  {e.eid} · {e.name}
                </option>
              ))}
          </select>
        </label>
        <button
          className="button secondary"
          disabled={!analysis || dirty || busy || !source || !target}
          type="submit"
        >
          <MagnifyingGlass size={18} />
          {busy ? "正在验证…" : "验证结论"}
        </button>
      </form>
      {dirty && (
        <p className="helper-text">
          参数已变化，请先运行主推理，再对当前数据验证结论。
        </p>
      )}
      {error && (
        <p role="alert" className="field-error">
          {error}
        </p>
      )}
      {result && (
        <div
          className={"query-result " + (result.proved ? "proved" : "unknown")}
          aria-live="polite"
        >
          <div>
            {result.proved ? <CheckCircle size={22} /> : <Question size={22} />}
            <strong>{result.proved ? "证明成立" : "暂不能证明"}</strong>
            <span>{result.message}</span>
          </div>
          {result.lines.length > 0 && (
            <details>
              <summary>
                查看反向证明路径（{result.lines.length} 个节点）
              </summary>
              <pre>{result.lines.join("\n")}</pre>
            </details>
          )}
        </div>
      )}
    </section>
  );
}
