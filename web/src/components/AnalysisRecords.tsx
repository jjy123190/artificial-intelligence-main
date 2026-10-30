import { DownloadSimple } from "@phosphor-icons/react";
import type { Analysis, CaseData, ControlQuery } from "../types";
import { downloadJson } from "../api";
import QueryPanel from "./QueryPanel";
export default function AnalysisRecords({
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
  return (
    <div className="records-page-layout">
      <div className="simple-panel">
        <QueryPanel
          data={data}
          analysis={analysis}
          dirty={dirty}
          initialQuery={initialQuery}
        />
      </div>
      {analysis && (
        <section className="simple-panel">
          <header className="records-header">
            <div>
              <h2>完整推理记录</h2>
              <p>
                {analysis.steps.length}次规则动作 ·
                每一步均对应当前分析的实际前提。
              </p>
            </div>
            <button
              className="button secondary"
              disabled={dirty}
              onClick={() => downloadJson(analysis, "equity-analysis.json")}
            >
              <DownloadSimple size={17} />
              下载分析JSON
            </button>
          </header>
          <details className="all-steps">
            <summary>展开全部推理步骤</summary>
            <div className="steps-list">
              {analysis.steps.map((s, i) => (
                <div key={i}>
                  <span className="cycle-tag">第{s.cycle}轮</span>
                  <span className="rule-code">{s.rule}</span>
                  <span>
                    {s.fact_ids
                      .map((id) => analysis.provenance[id]?.label || id)
                      .join("；")}
                  </span>
                </div>
              ))}
            </div>
          </details>
        </section>
      )}
    </div>
  );
}
