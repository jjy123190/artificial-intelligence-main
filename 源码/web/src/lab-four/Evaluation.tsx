import { useState } from "react";
import Reliability from "./Reliability";
import ModuleContrasts from "./ModuleContrasts";
import {
  ArrowDown,
  ArrowUp,
  DownloadSimple,
  Info,
  ArrowRight,
} from "@phosphor-icons/react";
import { downloadJson } from "../api";
import { percent } from "./charts";
import { AccuracyPlot, PairedPlot, LatencyPlot } from "./EvidencePlots";
import { reportPages, type ReportPage } from "./navigation";
import { signedPoints } from "./presentation";
import type { Comparison, LabFourSnapshot } from "./types";
const timingNames: Record<string, string> = {
  baseline: "Baseline",
  maml: "MAML",
  calibrated: "完整校准",
  graph: "白化＋图传播",
  multiview: "双视图稳健校准",
  wide: "较宽骨干＋度量训练",
  wideview: "较宽双视图",
  continued: "同容量续训双视图",
};

function EvidenceTable({
  rows,
  caption,
}: {
  rows: Comparison[];
  caption: string;
}) {
  return (
    <div
      className="table-scroll lab4-evidence-table"
      tabIndex={0}
      role="region"
      aria-label={`${caption}，可横向滚动`}
    >
      <table>
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            <th scope="col">方法 / 策略</th>
            <th scope="col">平均准确率</th>
            <th scope="col">95% 区间</th>
            <th scope="col">相对 Baseline</th>
            <th scope="col">推理设置</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={`${r.method}-${r.shot}`}
              className={r.method === "full" ? "lab4-featured-row" : ""}
            >
              <th scope="row">{r.label}</th>
              <td className="lab4-number">{percent(r.accuracy, 2)}</td>
              <td>
                {percent(Math.max(0, r.accuracy - r.ci95), 2)} –{" "}
                {percent(Math.min(1, r.accuracy + r.ci95), 2)}
              </td>
              <td>
                {r.method === "baseline" ? (
                  <span className="lab4-muted">参考</span>
                ) : r.paired_delta != null && r.paired_ci95 != null ? (
                  <>
                    <strong
                      className={
                        r.paired_delta - r.paired_ci95 > 0
                          ? "lab4-positive"
                          : r.paired_delta + r.paired_ci95 < 0
                            ? "lab4-negative"
                            : ""
                      }
                    >
                      {signedPoints(r.paired_delta)} pp
                    </strong>
                    <small>
                      ± {(r.paired_ci95 * 100).toFixed(2)} pp · 配对
                    </small>
                  </>
                ) : (
                  "未记录配对差"
                )}
              </td>
              <td>
                {r.transductive
                  ? r.balanced_queries
                    ? "均衡转导"
                    : "转导"
                  : "归纳"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function Conclusion({ row }: { row?: Comparison }) {
  if (
    !row ||
    row.method === "baseline" ||
    row.paired_delta == null ||
    row.paired_ci95 == null
  )
    return null;
  const lower = row.paired_delta - row.paired_ci95,
    upper = row.paired_delta + row.paired_ci95;
  const verdict =
    lower > 0
      ? "配对区间高于零"
      : upper < 0
        ? "配对区间低于零"
        : "配对区间跨零";
  return (
    <div
      className={`lab4-conclusion ${lower > 0 ? "positive" : upper < 0 ? "negative" : "neutral"}`}
    >
      {lower > 0 ? (
        <ArrowUp size={21} />
      ) : upper < 0 ? (
        <ArrowDown size={21} />
      ) : (
        <Info size={21} />
      )}
      <div>
        <strong>
          {verdict}：
          {lower > 0
            ? "本次固定模型下可确认改善"
            : upper < 0
              ? "本次所选方案低于基线"
              : "暂不能确认改善"}
        </strong>
        <p>
          {row.label}相对 Baseline 为 {signedPoints(row.paired_delta)} ±{" "}
          {(row.paired_ci95 * 100).toFixed(2)}{" "}
          个百分点。区间仅覆盖任务抽样，不覆盖重新训练模型的波动。
        </p>
      </div>
    </div>
  );
}
export default function Evaluation({
  snapshot,
  view = "evaluation",
}: {
  snapshot: LabFourSnapshot;
  view?: ReportPage;
}) {
  const [shots, setShots] = useState<1 | 5>(1);
  const [ablationView, setAblationView] = useState<"modules" | "baseline">(
    "modules",
  );
  const [focusMethod, setFocusMethod] = useState<string>(
    snapshot.graph_extension?.available ? "graph" : "full",
  );
  const rows = snapshot.comparisons.filter((r) => r.shot === shots);
  const ablations = snapshot.ablations.filter(
    (r) =>
      r.shot === shots &&
      !["graph", "multiview", "wide", "wideview", "continued"].includes(
        r.method,
      ),
  );
  const full = rows.find(
    (r) => r.method === "full" || r.method === "calibrated",
  );
  const baseline = rows.find((r) => r.method === "baseline");
  const focus = rows.find((r) => r.method === focusMethod) || full || rows[0];
  const best = ablations.reduce<Comparison | undefined>(
    (best, row) => (!best || row.accuracy > best.accuracy ? row : best),
    undefined,
  );
  const cfg = snapshot.calibration?.[String(shots)];
  const page = reportPages.find((item) => item.id === view)!;
  const v2 = snapshot.study?.version === "v2";
  return (
    <div className="lab4-evaluation">
      <header className="lab4-evaluation-header">
        <div>
          <h2>{page.title}</h2>
          <p>{page.description}</p>
        </div>
        <div className="lab4-evaluation-actions">
          <button
            className="button secondary"
            onClick={() =>
              downloadJson(
                {
                  comparisons: snapshot.comparisons,
                  ablations: snapshot.ablations,
                  module_contrasts: snapshot.module_contrasts,
                  history: snapshot.history,
                  paired_history: snapshot.paired_history,
                  provenance: snapshot.provenance,
                  study: snapshot.study,
                  verification: snapshot.verification,
                  diagnostics: snapshot.diagnostics,
                  calibration: snapshot.calibration,
                  edge_benchmark: snapshot.edge_benchmark,
                },
                "lab04-evaluation.json",
              )
            }
          >
            <DownloadSimple size={16} />
            导出评测
          </button>
          <div
            className="lab4-shot-selector"
            role="group"
            aria-label="评测样本数"
          >
            {([1, 5] as const).map((n) => (
              <button
                key={n}
                aria-pressed={shots === n}
                onClick={() => setShots(n)}
              >
                {n}-shot
              </button>
            ))}
          </div>
        </div>
      </header>
      <nav className="lab4-report-nav" aria-label="实验结果页面">
        {reportPages.map((item) => (
          <a
            key={item.id}
            href={`#/labs/lab04/${item.id}`}
            aria-current={view === item.id ? "page" : undefined}
          >
            {item.label}
          </a>
        ))}
      </nav>
      {view === "evaluation" &&
        (rows.length ? (
          <>
            <div className="lab4-evidence-toolbar">
              <label>
                重点查看
                <select
                  value={focus?.method || "full"}
                  onChange={(e) => setFocusMethod(e.target.value)}
                >
                  {rows.map((r) => (
                    <option key={r.method} value={r.method}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </label>
              {snapshot.verification?.verified && (
                <span className="lab4-evidence-status">
                  逐任务核验通过 · {snapshot.verification.core_tasks} 个共同任务
                </span>
              )}
            </div>
            <div className="lab4-evaluation-facts">
              <div>
                <span>任务设置</span>
                <strong>5-way {shots}-shot</strong>
                <small>每类 15 张查询图片</small>
              </div>
              <div>
                <span>配对任务</span>
                <strong>
                  {rows[0].episodes}
                  <small> 个</small>
                </strong>
                <small>同一批类别、支持与查询</small>
              </div>
              <div>
                <span>{focus?.label || "所选方案"}</span>
                <strong>{focus ? percent(focus.accuracy, 2) : "未记录"}</strong>
                <small>
                  ± {focus ? (focus.ci95 * 100).toFixed(2) : "—"} pp · 95%
                  区间半宽
                </small>
              </div>
              <div>
                <span>基线参考</span>
                <strong>
                  {baseline ? percent(baseline.accuracy, 2) : "未记录"}
                </strong>
                <small>监督训练 + 最近原型</small>
              </div>
            </div>
            <section className="lab4-evaluation-section">
              <header>
                <h2>不同学习方式，谁更适合这次任务？</h2>
                <p>
                  误差线为任务级 95% 置信区间。20% 是五分类随机猜测的期望水平。
                </p>
              </header>
              <AccuracyPlot
                rows={rows}
                selected={focus?.method}
                onSelect={setFocusMethod}
              />
              <details className="lab4-data-details">
                <summary>查看精确数值与配对区间</summary>
                <EvidenceTable rows={rows} caption="主要方法对比" />
              </details>
              <p className="lab4-next-page">
                <a href="#/labs/lab04/ablations">
                  哪个模块起作用？打开独立消融页 <ArrowRight size={15} />
                </a>
              </p>
            </section>
            <Conclusion row={focus} />
            {focus?.method === "multiview" && (
              <div className="lab4-incremental-note" role="status">
                <strong>多视图增量实验</strong>
                <p>
                  相对已发布图传播：{signedPoints(focus.paired_vs_graph ?? 0)} ±{" "}
                  {((focus.paired_vs_graph_ci95 ?? 0) * 100).toFixed(2)}{" "}
                  个百分点。
                  {(focus.paired_vs_graph ?? 0) -
                    (focus.paired_vs_graph_ci95 ?? 0) >
                  0
                    ? "配对区间高于零。"
                    : (focus.paired_vs_graph ?? 0) +
                          (focus.paired_vs_graph_ci95 ?? 0) <
                        0
                      ? "配对区间低于零，本设置低于图传播。"
                      : "配对区间跨零，暂不能确认改善。"}
                  同一图像编码两次，增加计算开销；没有增加带标签样本。
                </p>
              </div>
            )}
            {focus?.method === "wide" && (
              <div className="lab4-incremental-note" role="status">
                <strong>独立较宽模型研究 · 不是同预算校准消融</strong>
                <p>
                  {snapshot.wide_extension?.model?.parameters.toLocaleString(
                    "zh-CN",
                  )}
                  参数 · 额外训练
                  {snapshot.wide_extension?.training?.epochs_completed}
                  轮，验证选中第
                  {snapshot.wide_extension?.training?.selected_epoch}
                  轮。较宽骨干不做任务校准的准确率为
                  {percent(focus.raw_accuracy ?? 0, 2)}。 相对双视图方案：
                  {signedPoints(focus.paired_vs_multiview ?? 0)} ±{" "}
                  {((focus.paired_vs_multiview_ci95 ?? 0) * 100).toFixed(2)}
                  个百分点。
                  {(focus.paired_vs_multiview ?? 0) -
                    (focus.paired_vs_multiview_ci95 ?? 0) >
                  0
                    ? "配对区间高于零。"
                    : (focus.paired_vs_multiview ?? 0) +
                          (focus.paired_vs_multiview_ci95 ?? 0) <
                        0
                      ? "配对区间低于零。"
                      : "区间跨零，不能确认改善。"}
                  扩大容量及增加训练预算的效果不能归于校准算法。
                </p>
              </div>
            )}
            {focus?.method === "wideview" && (
              <div className="lab4-incremental-note" role="status">
                <strong>较宽双视图增量 · 同权重，不新增训练</strong>
                <p>
                  相对原较宽方案：
                  {focus.paired_vs_wide != null
                    ? signedPoints(focus.paired_vs_wide)
                    : "未记录"}{" "}
                  ±{" "}
                  {focus.paired_vs_wide_ci95 != null
                    ? (focus.paired_vs_wide_ci95 * 100).toFixed(2)
                    : "未记录"}{" "}
                  个百分点。
                  {focus.paired_vs_wide != null &&
                    focus.paired_vs_wide_ci95 != null &&
                    (focus.paired_vs_wide - focus.paired_vs_wide_ci95 > 0
                      ? "配对区间高于零。"
                      : "区间跨零或低于零，不能确认改善。")}
                  同一图片编码原图与翻转图两次，按验证配置平均表征并进行训练类内白化；不是同计算开销对比，也不能将与轻量骨干的差异全归于校准。
                </p>
              </div>
            )}
            {focus?.method === "continued" && (
              <div className="lab4-incremental-note" role="status">
                <strong>同容量续训研究 · 额外训练预算单独披露</strong>
                <p>
                  与较宽模型相同509,856参数；额外训练
                  {snapshot.continued_extension?.training?.additional_epochs}
                  轮，验证选择第
                  {
                    snapshot.continued_extension?.training
                      ?.selected_additional_epoch
                  }
                  轮。沿用原较宽双视图分类配置，没有新增测试集参数搜索。
                </p>
                <p>
                  相对上一版较宽双视图：
                  {focus.paired_vs_wideview != null
                    ? signedPoints(focus.paired_vs_wideview)
                    : "未记录"}{" "}
                  ±{" "}
                  {focus.paired_vs_wideview_ci95 != null
                    ? (focus.paired_vs_wideview_ci95 * 100).toFixed(2)
                    : "未记录"}{" "}
                  个百分点。
                  {focus.paired_vs_wideview != null &&
                    focus.paired_vs_wideview_ci95 != null &&
                    (focus.paired_vs_wideview - focus.paired_vs_wideview_ci95 >
                    0
                      ? "配对区间高于零。"
                      : "区间跨零或低于零，不能确认改善。")}
                  这是增加训练预算的收益，不归为新的校准模块贡献；单训练种子的区间不覆盖重训练波动。
                </p>
              </div>
            )}
            {focus?.method === "graph" && (
              <p className="lab4-incremental-note">
                同骨干增量改进：训练类统计白化与任务内图传播。使用均衡查询先验，原完整方案保留作为对照。
              </p>
            )}
          </>
        ) : (
          <section className="lab4-not-ready">
            <h2>评测结果尚未就绪</h2>
            <p>完成多任务评测后重新读取；这里不会用演示任务填充平均准确率。</p>
          </section>
        ))}
      {view === "ablations" && (
        <section className="lab4-evaluation-section">
          <header className="lab4-section-title">
            <div>
              <h2>拆开来看，哪个模块真正起作用？</h2>
              <p>
                固定校准骨干逐项消融；去除对齐训练使用配对训练的 Baseline 骨干。
              </p>
            </div>
            {best && (
              <span>
                当前基础消融最高：{best.label} · {percent(best.accuracy, 2)}
              </span>
            )}
          </header>
          <div
            className="lab4-ablation-view"
            role="group"
            aria-label="消融比较口径"
          >
            <button
              type="button"
              aria-pressed={ablationView === "modules"}
              onClick={() => setAblationView("modules")}
            >
              模块直接对照
            </button>
            <button
              type="button"
              aria-pressed={ablationView === "baseline"}
              onClick={() => setAblationView("baseline")}
            >
              相对 Baseline
            </button>
          </div>
          {ablationView === "modules" ? (
            snapshot.module_contrasts ? (
              <ModuleContrasts
                evidence={snapshot.module_contrasts}
                comparisons={[...rows, ...ablations]}
                shots={shots}
              />
            ) : (
              <p className="lab4-empty-inline">
                当前记录未提供模块直接配对统计。可查看相对Baseline的总效果，不据此单独归因。
              </p>
            )
          ) : (
            <PairedPlot rows={ablations} />
          )}
          <details className="lab4-data-details">
            <summary>查看消融实验全部数值</summary>
            <EvidenceTable rows={ablations} caption="校准策略消融" />
          </details>
          {cfg && (
            <details className="lab4-calibration-details">
              <summary>查看验证集锁定的校准参数</summary>
              <dl>
                <div>
                  <dt>均值收缩系数</dt>
                  <dd>{cfg.mean_weight}</dd>
                </div>
                <div>
                  <dt>方差混合系数</dt>
                  <dd>{cfg.variance_blend ?? 1}</dd>
                </div>
                <div>
                  <dt>原型更新</dt>
                  <dd>
                    {cfg.prototype_steps} 次 · 权重 {cfg.prototype_weight}
                  </dd>
                </div>
                <div>
                  <dt>查询类别先验</dt>
                  <dd>
                    {cfg.balanced_queries
                      ? "均衡先验（不读取标签）"
                      : "无均衡约束"}
                  </dd>
                </div>
              </dl>
              <p>
                系数为 0 表示验证集选择关闭该模块，不应将其写成该模块带来提升。
              </p>
            </details>
          )}
        </section>
      )}
      {view === "history" && snapshot.history?.length ? (
        <section className="lab4-evaluation-section">
          <header>
            <h2>保留原始实验，不覆盖负结果</h2>
            <p>
              旧版使用 32 维骨干和 100
              个任务；其原始记录保留。不同协议的均值不能当作配对提升。
            </p>
          </header>
          {snapshot.history.map((h) => (
            <details key={h.version} className="lab4-history">
              <summary>{h.label}</summary>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">方法</th>
                      <th scope="col">1-shot</th>
                      <th scope="col">5-shot</th>
                    </tr>
                  </thead>
                  <tbody>
                    {["baseline", "maml", "full", "prototype_only"].map(
                      (method) => (
                        <tr key={method}>
                          <th scope="row">
                            {h.rows.find((r) => r.method === method)?.label}
                          </th>
                          {[1, 5].map((n) => {
                            const row = h.rows.find(
                              (r) => r.method === method && r.shot === n,
                            );
                            return (
                              <td key={n}>
                                {row
                                  ? `${percent(row.accuracy, 2)} ± ${(row.ci95 * 100).toFixed(2)} pp`
                                  : "未记录"}
                              </td>
                            );
                          })}
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
              </div>
            </details>
          ))}
          {!!snapshot.paired_history?.length && (
            <>
              <h3 className="lab4-history-paired-title">
                旧权重也重新面对本轮相同任务
              </h3>
              <EvidenceTable
                rows={snapshot.paired_history.filter((r) => r.shot === shots)}
                caption="历史权重在新评测协议上的复测"
              />
              <p className="lab4-footnote">
                配对差仍以本轮 Baseline
                为参考。这样可区分骨干训练改善与校准策略改善。
              </p>
            </>
          )}
        </section>
      ) : null}
      {view === "deployment" && snapshot.edge_benchmark && (
        <section className="lab4-evaluation-section">
          <header>
            <h2>在本机CPU上，重新编码每张图</h2>
            <p>
              同一计时口径：图像编码 + 分类 /
              支持集适应。中位数与P95反映延迟分布，不与缓存特征计时混排。
            </p>
          </header>
          <LatencyPlot
            rows={snapshot.edge_benchmark.rows.filter((r) => r.shots === shots)}
            names={timingNames}
          />
          <details className="lab4-data-details">
            <summary>查看逐方法延迟数值与重复次数</summary>
            <div
              className="table-scroll"
              tabIndex={0}
              role="region"
              aria-label="本机CPU延迟对比，可横向滚动"
            >
              <table>
                <caption className="sr-only">
                  本地CPU统一计时范围的推理延迟
                </caption>
                <thead>
                  <tr>
                    <th scope="col">方法</th>
                    <th scope="col">中位数</th>
                    <th scope="col">P95</th>
                    <th scope="col">重复次数</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshot.edge_benchmark.rows
                    .filter((r) => r.shots === shots)
                    .map((r) => (
                      <tr key={r.method}>
                        <th scope="row">{timingNames[r.method] || r.method}</th>
                        <td>
                          {r.encoding_and_classification_median_ms.toFixed(1)}{" "}
                          ms
                        </td>
                        <td>
                          {r.encoding_and_classification_p95_ms.toFixed(1)} ms
                        </td>
                        <td>{r.repetitions}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      )}
      {view === "reliability" && snapshot.diagnostics && (
        <Reliability diagnostics={snapshot.diagnostics} shots={shots} />
      )}
      {view === "reliability" && !snapshot.diagnostics && (
        <p className="lab4-empty-inline">
          当前没有可核验的概率诊断记录，不用准确率推造可靠性。
        </p>
      )}
      {view === "deployment" && !snapshot.edge_benchmark && (
        <p className="lab4-empty-inline">当前没有统一计时口径的部署记录。</p>
      )}
      {view === "history" && !snapshot.history?.length && (
        <p className="lab4-empty-inline">当前版本没有归档的历史实验。</p>
      )}
      {(view === "history" || view === "reliability") && (
        <section className="lab4-evaluation-section lab4-limitations">
          <header>
            <h2>读懂结果，也看清边界</h2>
            <p>
              {v2
                ? "这是原基准上的后续优化实验，不是新数据集的独立外部验证。"
                : "完整组合不一定优于单个模块，单次训练也不能代表所有设置。"}
            </p>
          </header>
          <ul>
            {snapshot.limitations.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </section>
      )}
      {view === "history" && (
        <details className="lab4-provenance">
          <summary>实验来源与可复现信息</summary>
          <dl>
            <div>
              <dt>实验版本</dt>
              <dd>{snapshot.study?.name || "原版实验"}</dd>
            </div>
            <div>
              <dt>随机种子</dt>
              <dd>{snapshot.provenance.seed}</dd>
            </div>
            <div>
              <dt>结果生成时间</dt>
              <dd>{snapshot.provenance.run_id || "未记录"}</dd>
            </div>
            <div>
              <dt>校准模型 SHA-256</dt>
              <dd>
                <code>{snapshot.provenance.checkpoint_sha256}</code>
              </dd>
            </div>
            <div>
              <dt>评测协议</dt>
              <dd>{snapshot.dataset.protocol_note}</dd>
            </div>
          </dl>
          <a href="#/labs/lab04/protocol">
            查看方法与训练过程 <ArrowRight size={14} />
          </a>
        </details>
      )}
    </div>
  );
}
