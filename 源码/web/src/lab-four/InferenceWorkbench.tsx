import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowsClockwise,
  ArrowRight,
  CheckCircle,
  DownloadSimple,
  Flask,
  Play,
  Shuffle,
  SlidersHorizontal,
  WarningCircle,
} from "@phosphor-icons/react";
import { downloadJson, request } from "../api";
import type {
  Calibration,
  EpisodeRequest,
  EpisodeResult,
  LabFourSnapshot,
  Method,
} from "./types";
import { classColor, duration, FeatureComparison, percent } from "./charts";
import {
  calibrationNames,
  methodNames,
  runKey,
  runName,
  sameEpisode,
  signedPoints,
  validEpisode,
} from "./presentation";
import QueryExplorer from "./QueryExplorer";

type Run = { input: EpisodeRequest; result: EpisodeResult };
const methodDetails: Record<Method, string> = {
  baseline: "固定图像特征 · 最近原型分类",
  calibrated: "统计量收缩 · 原型自适应更新",
  maml: "支持集梯度适应 · 全网络元学习",
};

export default function InferenceWorkbench({
  snapshot,
  active,
}: {
  snapshot: LabFourSnapshot;
  active: boolean;
}) {
  const [method, setMethod] = useState<Method>("calibrated");
  const [shots, setShots] = useState<1 | 5>(1);
  const [calibration, setCalibration] = useState<Calibration>(
    snapshot.graph_extension?.available ? "graph" : "full",
  );
  const [seed, setSeed] = useState("20261124");
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const pending = useRef<AbortController | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const runButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (busy) return;
    if (calibration === "continued" && !snapshot.continued_extension?.available)
      setCalibration(snapshot.graph_extension?.available ? "graph" : "full");
    if (calibration === "wideview" && !snapshot.wideview_extension?.available)
      setCalibration(snapshot.graph_extension?.available ? "graph" : "full");
    if (calibration === "wide" && !snapshot.wide_extension?.available)
      setCalibration(snapshot.graph_extension?.available ? "graph" : "full");
    if (calibration === "graph" && !snapshot.graph_extension?.available)
      setCalibration("full");
    if (calibration === "multiview" && !snapshot.multiview_extension?.available)
      setCalibration(snapshot.graph_extension?.available ? "graph" : "full");
  }, [
    busy,
    calibration,
    snapshot.graph_extension?.available,
    snapshot.multiview_extension?.available,
    snapshot.wide_extension?.available,
    snapshot.wideview_extension?.available,
    snapshot.continued_extension?.available,
  ]);
  const current = runs.find((r) => runKey(r.input) === selected) || runs.at(-1);
  const result = current?.result;
  const seedValue = Number(seed);
  const seedValid =
    /^\d+$/.test(seed.trim()) &&
    Number.isSafeInteger(seedValue) &&
    seedValue <= 2_147_483_647;
  useEffect(() => {
    if (!active) {
      pending.current?.abort();
      pending.current = null;
      setBusy(false);
      setProgress("");
    }
    return () => {
      pending.current?.abort();
      pending.current = null;
    };
  }, [active]);
  const input = (): EpisodeRequest => ({
    method,
    shots,
    seed: seedValue,
    calibration: method === "calibrated" ? calibration : "none",
  });
  const stale =
    !!current &&
    (current.input.shots !== shots ||
      current.input.seed !== seedValue ||
      runKey(current.input) !== runKey(input()));
  const paired = runs.length > 1;
  const base = runs.find((r) => r.input.method === "baseline");
  const matchingBenchmark =
    snapshot.comparisons.find(
      (r) =>
        r.shot === shots &&
        r.method ===
          (method === "calibrated"
            ? calibration === "full" ||
              calibration === "graph" ||
              calibration === "multiview" ||
              calibration === "wide" ||
              calibration === "wideview" ||
              calibration === "continued"
              ? calibration
              : `${calibration}_only`
            : method),
    ) ||
    (method === "calibrated"
      ? snapshot.ablations.find(
          (r) =>
            r.shot === shots &&
            r.method ===
              (calibration === "full" ||
              calibration === "graph" ||
              calibration === "multiview" ||
              calibration === "wide" ||
              calibration === "wideview" ||
              calibration === "continued"
                ? calibration
                : calibration === "none"
                  ? "alignment_only"
                  : `${calibration}_only`),
        )
      : undefined);
  const correctByClass = useMemo(
    () =>
      result?.classes.map((c) => ({
        ...c,
        correct: result.queries.filter(
          (q) => q.target_id === c.id && q.predicted_id === c.id,
        ).length,
      })) || [],
    [result],
  );

  async function run(compare = false) {
    if (pending.current || !snapshot.available || !seedValid) return;
    const controller = new AbortController();
    pending.current = controller;
    const chosen = input();
    const requestRevision = snapshot.provenance.artifact_revision;
    const plan: EpisodeRequest[] = compare
      ? [
          { ...chosen, method: "baseline", calibration: "none" },
          { ...chosen, method: "calibrated", calibration },
          { ...chosen, method: "maml", calibration: "none" },
        ]
      : [chosen];
    setBusy(true);
    setError("");
    setNotice("");
    let next = [...runs];
    try {
      for (let i = 0; i < plan.length; i++) {
        const task = plan[i];
        setProgress(
          `${compare ? `${i + 1} / ${plan.length} · ` : ""}${runName(task)} 正在编码并分类`,
        );
        const value = await request<unknown>("/api/labs/lab04/run", {
          method: "POST",
          body: JSON.stringify(task),
          signal: controller.signal,
          timeoutMs: 120_000,
        });
        if (controller.signal.aborted || pending.current !== controller) return;
        if (
          !validEpisode(value) ||
          value.seed !== task.seed ||
          value.shots !== task.shots ||
          value.method !== task.method ||
          value.calibration !== task.calibration
        )
          throw new Error("推理结果与请求或评分不一致，本次结果未纳入对比。");
        if (
          requestRevision &&
          value.artifact_revision &&
          value.artifact_revision !== requestRevision
        ) {
          throw new Error(
            "模型版本在本次请求前已更新，请先重新读取实验状态，再运行配对比较。",
          );
        }
        if (next.length && !sameEpisode(next[0].result, value)) {
          if (i > 0)
            throw new Error("方法间的样本不一致，已停止配对比较，请重新运行。");
          next = [];
        }
        next = [
          ...next.filter((r) => runKey(r.input) !== runKey(task)),
          { input: task, result: value },
        ];
        setRuns(next);
        setSelected(runKey(task));
      }
      setSelected(runKey(chosen));
      setProgress(
        compare ? "三种方法已在同一组样本上完成对照。" : "真实推理完成。",
      );
      requestAnimationFrame(() =>
        heading.current?.focus({ preventScroll: true }),
      );
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(
          failure instanceof Error ? failure.message : "推理未完成，请重试。",
        );
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setBusy(false);
      }
    }
  }
  function cancelRun() {
    const controller = pending.current;
    if (!controller) return;
    pending.current = null;
    controller.abort();
    setBusy(false);
    setProgress("");
    setError("");
    setNotice(
      "已取消等待，保留已完成的结果。服务器当前计算可能仍会完成，但不会再更新本次结果，也不会继续运行后续对比方法。",
    );
    requestAnimationFrame(() =>
      runButton.current?.focus({ preventScroll: true }),
    );
  }
  function choose(run: Run) {
    setSelected(runKey(run.input));
    setMethod(run.input.method);
    if (run.input.method === "calibrated")
      setCalibration(run.input.calibration);
    setShots(run.input.shots);
    setSeed(String(run.input.seed));
  }
  function newSeed() {
    const values = new Uint32Array(1);
    crypto.getRandomValues(values);
    setSeed(String(values[0] % 2_147_483_648));
  }
  function exportRun() {
    if (!result) return;
    downloadJson(
      {
        schema_version: 1,
        scope: "single live episode; not benchmark accuracy",
        exported_at: new Date().toISOString(),
        runs: runs.map(({ input, result: r }) => ({
          input,
          result: {
            ...r,
            support: r.support.map(({ image: _, ...rest }) => rest),
            queries: r.queries.map(({ image: _, ...rest }) => rest),
          },
        })),
      },
      `lab04-${result.shots}shot-seed${result.seed}.json`,
    );
  }
  return (
    <div className="lab4-workbench">
      <aside className="lab4-setup" aria-label="分类任务设置">
        <div className="lab4-setup-heading">
          <SlidersHorizontal size={18} />
          <h2>设计一个小样本任务</h2>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void run();
          }}
        >
          <fieldset disabled={busy}>
            <legend>每类给模型几张示例？</legend>
            <div className="lab4-shot-cards">
              {([1, 5] as const).map((n) => (
                <button
                  key={n}
                  type="button"
                  aria-pressed={shots === n}
                  onClick={() => setShots(n)}
                >
                  <strong>
                    {n}
                    <small> 张</small>
                  </strong>
                  <span>{n}-shot</span>
                </button>
              ))}
            </div>
          </fieldset>
          <fieldset disabled={busy}>
            <legend>推理方法</legend>
            <div className="lab4-methods">
              {(["baseline", "calibrated", "maml"] as const).map((id) => (
                <label key={id} className={method === id ? "selected" : ""}>
                  <input
                    type="radio"
                    name="lab4-method"
                    value={id}
                    checked={method === id}
                    onChange={() => setMethod(id)}
                  />
                  <span>
                    <strong>{methodNames[id]}</strong>
                    <small>{methodDetails[id]}</small>
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
          {method === "calibrated" && (
            <label className="lab4-control-label">
              校准策略
              <select
                value={calibration}
                onChange={(e) => setCalibration(e.target.value as Calibration)}
                disabled={busy}
              >
                {Object.entries(calibrationNames)
                  .filter(
                    ([id]) =>
                      (id !== "graph" || snapshot.graph_extension?.available) &&
                      (id !== "multiview" ||
                        snapshot.multiview_extension?.available) &&
                      (id !== "wide" || snapshot.wide_extension?.available) &&
                      (id !== "wideview" ||
                        snapshot.wideview_extension?.available) &&
                      (id !== "continued" ||
                        snapshot.continued_extension?.available),
                  )
                  .map(([id, name]) => (
                    <option key={id} value={id}>
                      {name}
                    </option>
                  ))}
              </select>
            </label>
          )}
          {method === "calibrated" &&
            (calibration === "wide" ||
              calibration === "wideview" ||
              calibration === "continued") && (
              <p className="lab4-footnote">
                此选项使用独立较宽骨干（
                {snapshot.wide_extension?.model?.parameters.toLocaleString(
                  "zh-CN",
                )}
                参数）与额外训练，不是原64维骨干的同预算消融。
              </p>
            )}
          {method === "calibrated" && calibration === "wideview" && (
            <p className="lab4-footnote">
              相对较宽单视图没有新增训练，但每图编码原图与翻转图两次。两种shot都在验证类锁定为均值聚合与训练类内白化。
            </p>
          )}
          {method === "calibrated" && calibration === "continued" && (
            <p className="lab4-footnote">
              保持509,856参数，额外训练
              {snapshot.continued_extension?.training?.additional_epochs}
              轮，验证选中第
              {
                snapshot.continued_extension?.training
                  ?.selected_additional_epoch
              }
              轮；不是同训练预算比较。每张图片仍编码两次。
            </p>
          )}
          <div className="lab4-control-label">
            <label htmlFor="lab4-task-seed">任务种子</label>
            <span className="lab4-seed-row">
              <input
                id="lab4-task-seed"
                type="text"
                inputMode="numeric"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                disabled={busy}
                aria-invalid={!seedValid}
                aria-describedby="lab4-seed-help"
              />
              <button
                type="button"
                className="icon-button"
                title="随机生成任务种子"
                aria-label="随机生成任务种子"
                onClick={newSeed}
                disabled={busy}
              >
                <Shuffle size={18} />
              </button>
            </span>
          </div>
          <p
            id="lab4-seed-help"
            className={seedValid ? "lab4-footnote" : "lab4-input-error"}
          >
            {seedValid
              ? "相同 shot 与种子使用同一组图片，公平比较方法。"
              : "请输入 0 至 2147483647 之间的整数。"}
          </p>
          <button
            ref={runButton}
            className="button primary lab4-run"
            disabled={busy || !snapshot.available || !seedValid}
            type="submit"
          >
            {busy ? (
              <ArrowsClockwise size={18} className="lab4-spinning" />
            ) : (
              <Play size={18} weight="fill" />
            )}
            {busy ? "正在推理" : "运行分类任务"}
          </button>
          <button
            className="button secondary lab4-compare-button"
            disabled={busy || !snapshot.available || !seedValid}
            type="button"
            onClick={() => void run(true)}
          >
            同任务对比三种方法
            <ArrowRight size={16} />
          </button>
          <p className="lab4-footnote">
            对比将运行 Baseline、
            {method === "calibrated"
              ? calibrationNames[calibration]
              : "所选校准策略"}
            与 MAML。全部使用真实权重，不是结果回放。
          </p>
        </form>
        <div className="lab4-setup-protocol">
          <span>固定协议</span>
          <strong>5 类 · {5 * shots} 张支持图片 · 75 张查询图片</strong>
          <p>支持与查询不重叠；查询标签不参与适应。</p>
        </div>
        {matchingBenchmark && (
          <div className="lab4-benchmark-hint">
            <span>所选方法的多任务参考</span>
            <strong>
              {percent(matchingBenchmark.accuracy, 2)}
              <small> ± {(matchingBenchmark.ci95 * 100).toFixed(2)} pp</small>
            </strong>
            <p>
              {matchingBenchmark.episodes} 个任务 · 95% 置信区间
              <br />
              不代表下一次抽样的准确率。
            </p>
            <a href="#/labs/lab04/evaluation">
              查看完整评测
              <ArrowRight size={13} />
            </a>
          </div>
        )}
      </aside>
      <div className="lab4-workspace">
        {notice && (
          <p className="lab4-operation-note" role="status">
            {notice}
          </p>
        )}
        {error && (
          <div className="error-banner" role="alert">
            <WarningCircle size={20} />
            <span>{error}</span>
          </div>
        )}
        {!snapshot.available && (
          <div className="lab4-not-ready">
            <WarningCircle size={32} />
            <h2>模型尚未就绪</h2>
            <p>请安装实验四依赖并核对权重，然后重新读取状态。</p>
          </div>
        )}
        {busy && (
          <div className="lab4-progress">
            <span className="lab4-progress-message" role="status">
              <ArrowsClockwise
                size={17}
                className="lab4-spinning"
                aria-hidden="true"
              />
              <span>{progress}</span>
            </span>
            <button
              type="button"
              className="button secondary lab4-cancel"
              onClick={cancelRun}
            >
              取消等待
            </button>
            <span className="lab4-progress-track" aria-hidden="true" />
          </div>
        )}
        {!result && snapshot.available && (
          <section className="lab4-onboarding" aria-label="如何开始分类实验">
            <span className="lab4-section-kicker">
              5-WAY CLASSIFICATION · 五分类任务
            </span>
            <h2>
              只看 {shots} 张示例，
              <br />
              能认出一个新类别吗？
            </h2>
            <p>
              让模型从五个未见过的类别中学习，再独立识别 75
              张图片。你可以检查每次误判，也可以让三种方法面对同一份考题。
            </p>
            <button
              type="button"
              className="button primary lab4-quickstart"
              disabled={busy || !seedValid}
              onClick={() => void run()}
            >
              <Play size={17} weight="fill" />
              开始一次真实预测
            </button>
            <div
              className="lab4-episode-diagram"
              aria-label={`${5 * shots}张有标签支持图片经过模型，预测75张无标签查询图片`}
            >
              <div>
                <div className="lab4-example-squares">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <span
                      key={i}
                      style={{
                        borderColor: classColor(i),
                        color: classColor(i),
                      }}
                    >
                      {shots}
                    </span>
                  ))}
                </div>
                <strong>{5 * shots} 张支持图片</strong>
                <small>每类 {shots} 张 · 标签可见</small>
              </div>
              <ArrowRight size={23} />
              <div className="lab4-diagram-network">
                <Flask size={29} />
                <strong>ResNet12</strong>
                <small>提取特征并适应</small>
              </div>
              <ArrowRight size={23} />
              <div>
                <div className="lab4-query-mosaic">
                  {Array.from({ length: 15 }, (_, i) => (
                    <span key={i} />
                  ))}
                </div>
                <strong>75 张查询图片</strong>
                <small>预测时标签不可见</small>
              </div>
            </div>
            <div className="lab4-onboarding-footer">
              <CheckCircle size={18} />
              <span>从左侧开始运行。每一次预测都可复现、核对和导出。</span>
            </div>
          </section>
        )}
        {result && (
          <div className="lab4-run-result" aria-busy={busy}>
            {snapshot.provenance.artifact_revision &&
              result.artifact_revision !==
                snapshot.provenance.artifact_revision && (
                <div className="lab4-stale" role="status">
                  模型结果版本已更新，当前展示的是之前的运行记录。重新运行以使用新版本。
                </div>
              )}
            <section
              className="lab4-result-summary"
              aria-labelledby="lab4-result-title"
            >
              <div className="lab4-result-heading">
                <span className="lab4-section-kicker">
                  LIVE INFERENCE · 真实推理
                </span>
                <h2 ref={heading} tabIndex={-1} id="lab4-result-title">
                  本次分类结果
                </h2>
                <p>
                  {result.way}-way {result.shots}-shot · 种子 {result.seed}
                </p>
              </div>
              <div className="lab4-result-actions">
                <button
                  type="button"
                  className="button secondary"
                  disabled={busy || !seedValid || !snapshot.available}
                  onClick={() => void run(true)}
                >
                  <ArrowsClockwise size={16} />
                  对比三种方法
                </button>
                <button
                  type="button"
                  className="button secondary lab4-export"
                  onClick={exportRun}
                  disabled={busy}
                >
                  <DownloadSimple size={16} />
                  导出记录
                </button>
              </div>
              <dl className="lab4-metrics">
                <div>
                  <dt>本次准确率</dt>
                  <dd>{percent(result.accuracy)}</dd>
                </div>
                <div>
                  <dt>正确 / 总数</dt>
                  <dd>
                    {result.correct}
                    <small> / {result.total}</small>
                  </dd>
                </div>
                <div>
                  <dt>编码 + 分类耗时</dt>
                  <dd className="lab4-time">{duration(result.latency_ms)}</dd>
                </div>
              </dl>
              <p className="lab4-result-note">
                {result.transductive
                  ? "转导推理：联合使用无标签查询图片。"
                  : "归纳推理：不联合使用其他查询图片。"}
                {result.balanced_queries ? "本方法还假设查询类别均衡。" : ""}
                单次任务结果不等于多任务平均准确率。
              </p>
            </section>
            {stale && (
              <div className="lab4-stale" role="status">
                参数已改变，下方仍是上一次的结果。点击运行以应用新设置。
              </div>
            )}
            <div
              className="lab4-run-comparison"
              aria-label="本任务已经运行的方法"
            >
              <div className="lab4-run-comparison-title">
                <h3>{paired ? "同一份考题，不同的方法" : "本任务运行记录"}</h3>
                <span>
                  {paired ? "已核对样本 ID 一致" : "继续运行其他方法可直接对照"}
                </span>
              </div>
              <div
                className="lab4-run-tabs"
                role="group"
                aria-label="切换查看方法结果"
              >
                {runs.map((r) => (
                  <button
                    key={runKey(r.input)}
                    disabled={busy}
                    type="button"
                    aria-pressed={r === current}
                    onClick={() => choose(r)}
                  >
                    <span>{runName(r.input)}</span>
                    <strong>{percent(r.result.accuracy)}</strong>
                    <small>
                      {base && r !== base
                        ? `${signedPoints(r.result.accuracy - base.result.accuracy)} pp vs Baseline`
                        : `${r.result.correct} / ${r.result.total} 正确`}
                    </small>
                  </button>
                ))}
              </div>
            </div>
            <section
              className="lab4-support-section"
              aria-labelledby="lab4-support-title"
            >
              <header className="lab4-section-title">
                <div>
                  <span className="lab4-section-kicker">
                    SUPPORT · 给模型的示例
                  </span>
                  <h2 id="lab4-support-title">这次要认识的五个类别</h2>
                  <p>
                    每类 {result.shots}{" "}
                    张标注图片。下方进度是预测完成后的类别召回率。
                  </p>
                </div>
              </header>
              <div className="lab4-support-grid">
                {correctByClass.map((c, index) => (
                  <div className="lab4-support-class" key={c.id}>
                    <h3>
                      <i style={{ background: classColor(index) }} />
                      {c.name}
                    </h3>
                    <div
                      className={`lab4-support-images ${result.shots === 1 ? "single" : ""}`}
                    >
                      {result.support
                        .filter((s) => s.class_id === c.id)
                        .map((s, i) => (
                          <img
                            key={s.id}
                            src={s.image}
                            alt={`${c.name}的第${i + 1}张支持图片`}
                            width={84}
                            height={84}
                          />
                        ))}
                    </div>
                    <div className="lab4-class-recall">
                      <span
                        style={{
                          width: `${(c.correct / 15) * 100}%`,
                          background: classColor(index),
                        }}
                      />
                    </div>
                    <small>
                      正确 <span>{c.correct} / 15</span>
                    </small>
                  </div>
                ))}
              </div>
            </section>
            <QueryExplorer result={result} />
            <section className="lab4-feature-section">
              <header className="lab4-section-title">
                <div>
                  <span className="lab4-section-kicker">
                    FEATURE SPACE · 特征空间
                  </span>
                  <h2>校准改变了什么？</h2>
                  <p>
                    {result.feature_motion === "real_support_adaptation"
                      ? "两次投影来自网络真实适应前后的特征。"
                      : "图像特征不移动；校准修正类别原型或距离度量。"}
                  </p>
                </div>
              </header>
              <FeatureComparison result={result} />
            </section>
            <details className="lab4-process-details">
              <summary>本次推理过程</summary>
              <ol>
                {result.steps.map((s) => (
                  <li key={s.name}>
                    <strong>{s.name}</strong>
                    <p>{s.detail}</p>
                  </li>
                ))}
              </ol>
            </details>
          </div>
        )}
        <span className="sr-only" role="status">
          {!busy && progress}
          {result && !busy
            ? `准确率${percent(result.accuracy)}，${result.correct}张预测正确。`
            : ""}
        </span>
      </div>
    </div>
  );
}
