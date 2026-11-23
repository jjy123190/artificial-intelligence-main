import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowClockwise,
  Play,
  CheckCircle,
  XCircle,
  ImageSquare,
  CaretDown,
  CaretUp,
} from "@phosphor-icons/react";
import { request } from "../api";
import { labHref } from "../routes";
import type { CourseLab } from "../types";
import type {
  Calibration,
  Comparison,
  EpisodeRequest,
  EpisodeResult,
  LabFourSnapshot,
  Method,
} from "../lab-four/types";
import {
  classColor,
  duration,
  FeatureComparison,
  percent,
  ResultBars,
  TrainingCurve,
} from "../lab-four/charts";
import "../lab-four/lab-four.css";

const sections = [
  { id: "inference", label: "现场推理" },
  { id: "protocol", label: "方法与协议" },
  { id: "evaluation", label: "对比与消融" },
];
const calibrationNames: Record<Calibration, string> = {
  none: "无校准 · 对齐骨干",
  mean: "仅均值校准",
  variance: "仅方差校准",
  prototype: "仅原型更新",
  full: "完整特征校准",
};
const methodNames: Record<Method, string> = {
  baseline: "Baseline",
  maml: "MAML",
  calibrated: "特征校准",
};
const timingLabels: Record<string, string> = {
  inference_from_cached_features: "特征分类",
  image_encoding_and_support_adaptation: "编码+适应",
};

function MetricTable({
  rows,
  ablation = false,
}: {
  rows: Comparison[];
  ablation?: boolean;
}) {
  return (
    <div
      className="table-scroll"
      tabIndex={0}
      aria-label={
        ablation ? "消融实验结果，可横向滚动" : "方法对比结果，可横向滚动"
      }
    >
      <table>
        <caption className="sr-only">
          {ablation ? "不同校准策略的消融实验" : "同一测试协议下的方法对比"}
        </caption>
        <thead>
          <tr>
            <th scope="col">{ablation ? "校准策略" : "方法"}</th>
            <th scope="col">平均准确率</th>
            <th scope="col">95% 置信区间</th>
            <th scope="col">任务数</th>
            {!ablation && <th scope="col">每任务耗时</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.method}-${row.shot}`}>
              <th scope="row">{row.label}</th>
              <td className="lab4-number">{percent(row.accuracy, 2)}</td>
              <td>
                {percent(Math.max(0, row.accuracy - row.ci95), 2)} ~{" "}
                {percent(Math.min(1, row.accuracy + row.ci95), 2)}
              </td>
              <td>{row.episodes.toLocaleString("zh-CN")}</td>
              {!ablation && (
                <td>
                  {row.latency_ms == null ? "未记录" : duration(row.latency_ms)}
                  <small className="lab4-timing-scope">
                    {timingLabels[row.timing_scope] || "范围未注明"}
                  </small>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function LabFour({
  lab,
  moduleId,
  active = true,
}: {
  lab: CourseLab;
  moduleId: string;
  active?: boolean;
}) {
  const [snapshot, setSnapshot] = useState<LabFourSnapshot | null>(null);
  const [loadError, setLoadError] = useState("");
  const [loading, setLoading] = useState(false);
  const [method, setMethod] = useState<Method>("calibrated");
  const [shots, setShots] = useState<1 | 5>(1);
  const [seed, setSeed] = useState("7");
  const [calibration, setCalibration] = useState<Calibration>("full");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<EpisodeResult | null>(null);
  const [lastRequest, setLastRequest] = useState<EpisodeRequest | null>(null);
  const [showAllQueries, setShowAllQueries] = useState(false);
  const [evaluationShots, setEvaluationShots] = useState<1 | 5>(1);
  const loadPending = useRef<AbortController | null>(null);
  const runPending = useRef<AbortController | null>(null);
  const snapshotRef = useRef<LabFourSnapshot | null>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const seedValue = Number(seed);
  const seedValid =
    seed.trim() !== "" &&
    Number.isInteger(seedValue) &&
    seedValue >= 0 &&
    seedValue <= 9999;

  const load = useCallback(async () => {
    loadPending.current?.abort();
    const controller = new AbortController();
    loadPending.current = controller;
    setLoading(true);
    setLoadError("");
    try {
      const value = await request<LabFourSnapshot>("/api/labs/lab04/snapshot", {
        signal: controller.signal,
        timeoutMs: 20_000,
      });
      if (controller.signal.aborted || loadPending.current !== controller)
        return;
      snapshotRef.current = value;
      setSnapshot(value);
      setMethod((current) =>
        value.methods.some((item) => item.id === current)
          ? current
          : value.methods[0]?.id || "calibrated",
      );
    } catch (failure) {
      if (!controller.signal.aborted)
        setLoadError(
          failure instanceof Error
            ? failure.message
            : "实验四数据读取失败，请重试。",
        );
    } finally {
      if (loadPending.current === controller) {
        loadPending.current = null;
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    if (active && !snapshotRef.current) void load();
    if (!active) {
      loadPending.current?.abort();
      runPending.current?.abort();
      runPending.current = null;
      setBusy(false);
      setLoading(false);
    }
  }, [active, load]);
  useEffect(
    () => () => {
      loadPending.current?.abort();
      runPending.current?.abort();
    },
    [],
  );

  async function run() {
    if (runPending.current || !snapshot?.available || !seedValid) return;
    const input: EpisodeRequest = {
      method,
      shots,
      seed: seedValue,
      calibration: method === "calibrated" ? calibration : "none",
    };
    const controller = new AbortController();
    runPending.current = controller;
    setBusy(true);
    setError("");
    try {
      const value = await request<EpisodeResult>("/api/labs/lab04/run", {
        method: "POST",
        body: JSON.stringify(input),
        signal: controller.signal,
        timeoutMs: 120_000,
      });
      if (controller.signal.aborted || runPending.current !== controller)
        return;
      if (
        value.compute !== "live" ||
        !Number.isFinite(value.accuracy) ||
        !Array.isArray(value.queries)
      ) {
        throw new Error("接口未返回有效的现场推理结果，请检查模型状态后重试。");
      }
      setResult(value);
      setLastRequest(input);
      setShowAllQueries(false);
      requestAnimationFrame(() =>
        resultHeading.current?.focus({ preventScroll: true }),
      );
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(
          failure instanceof Error
            ? failure.message
            : "本次推理未完成，请重试。",
        );
    } finally {
      if (runPending.current === controller) {
        runPending.current = null;
        setBusy(false);
      }
    }
  }

  const changed =
    !!lastRequest &&
    (lastRequest.method !== method ||
      lastRequest.shots !== shots ||
      lastRequest.seed !== seedValue ||
      lastRequest.calibration !==
        (method === "calibrated" ? calibration : "none"));
  const classNames = useMemo(
    () => new Map(result?.classes.map((item) => [item.id, item.name]) || []),
    [result],
  );
  const comparisonRows =
    snapshot?.comparisons.filter(
      (row) => row.shot === evaluationShots && Number.isFinite(row.accuracy),
    ) || [];
  const ablationRows =
    snapshot?.ablations.filter(
      (row) => row.shot === evaluationShots && Number.isFinite(row.accuracy),
    ) || [];
  const visibleQueries = result
    ? showAllQueries
      ? result.queries
      : result.queries.slice(0, 12)
    : [];
  const currentSection =
    sections.find((section) => section.id === moduleId) || sections[0];
  const methodName = (id: Method) =>
    snapshot?.methods.find((item) => item.id === id)?.name || methodNames[id];

  return (
    <main
      id={active ? "main-content" : undefined}
      tabIndex={-1}
      className="page-shell course-shell task-page lab4-page"
    >
      <nav className="course-breadcrumb" aria-label="当前位置">
        <a href={labHref("lab04")}>
          <ArrowLeft size={16} />
          实验四
        </a>
        <span>/</span>
        <span>{currentSection.label}</span>
      </nav>
      <header className="course-header lab4-heading">
        <div>
          <h1>小样本图像分类与特征校准</h1>
          <p>用少量已标注样本认识新类别，观察特征估计与原型更新带来的变化。</p>
        </div>
        <button
          className="icon-button lab4-reload"
          type="button"
          onClick={() => void load()}
          disabled={loading || busy}
          title="重新读取实验状态"
          aria-label="重新读取实验状态"
        >
          <ArrowClockwise
            size={20}
            className={loading ? "lab4-spinning" : undefined}
          />
        </button>
      </header>
      <nav className="lab4-section-nav" aria-label="实验四小节">
        {sections.map((section) => (
          <a
            key={section.id}
            href={labHref(lab.id, section.id)}
            aria-current={currentSection.id === section.id ? "page" : undefined}
          >
            {section.label}
          </a>
        ))}
      </nav>
      {loadError && (
        <div className="error-banner" role="alert">
          <span>{loadError}</span>
          <button
            className="button secondary"
            onClick={() => void load()}
            disabled={loading}
          >
            重新加载
          </button>
        </div>
      )}
      {!snapshot && loading && (
        <div
          className="skeleton course-skeleton"
          aria-label="正在读取实验四模型与结果"
        />
      )}

      {snapshot && currentSection.id === "inference" && (
        <>
          <section className="lab4-controls" aria-label="分类任务设置">
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void run();
              }}
            >
              <label>
                推理方法
                <select
                  value={method}
                  onChange={(event) => setMethod(event.target.value as Method)}
                  disabled={busy || !snapshot.methods.length}
                >
                  {snapshot.methods.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                每类标注样本
                <select
                  value={shots}
                  onChange={(event) =>
                    setShots(Number(event.target.value) as 1 | 5)
                  }
                  disabled={busy}
                >
                  <option value={1}>1 张 · 1-shot</option>
                  <option value={5}>5 张 · 5-shot</option>
                </select>
              </label>
              {method === "calibrated" && (
                <label>
                  校准策略
                  <select
                    value={calibration}
                    onChange={(event) =>
                      setCalibration(event.target.value as Calibration)
                    }
                    disabled={busy}
                  >
                    {Object.entries(calibrationNames).map(([id, name]) => (
                      <option key={id} value={id}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label className="lab4-seed">
                抽样种子
                <input
                  type="number"
                  min={0}
                  max={9999}
                  step={1}
                  value={seed}
                  onChange={(event) => setSeed(event.target.value)}
                  disabled={busy}
                  aria-invalid={!seedValid}
                  aria-describedby={!seedValid ? "lab4-seed-error" : undefined}
                />
              </label>
              <button
                type="submit"
                className="button primary lab4-run"
                disabled={busy || !snapshot.available || !seedValid}
              >
                {busy ? (
                  <ArrowClockwise size={18} className="lab4-spinning" />
                ) : (
                  <Play size={18} weight="fill" />
                )}
                {busy ? "模型推理中" : "运行分类任务"}
              </button>
            </form>
            <p className="lab4-control-note">
              固定 5
              个类别。每类样本数与种子相同时，复用同一组样本对照不同方法。
            </p>
            {!seedValid && (
              <p className="lab4-input-error" id="lab4-seed-error">
                抽样种子请输入 0 到 9999 的整数。
              </p>
            )}
          </section>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          {!snapshot.available && (
            <section className="lab4-not-ready" role="status">
              <ImageSquare size={32} />
              <h2>模型与实验数据尚未就绪</h2>
              <p>模型准备完成后，重新读取实验状态即可运行。</p>
            </section>
          )}
          {snapshot.available && !result && (
            <section className="lab4-not-ready" aria-live="polite">
              <ImageSquare size={36} />
              <h2>
                {busy ? "正在用支持样本适应新类别" : "运行一次真实分类任务"}
              </h2>
              <p>
                {busy
                  ? "完成后将显示查询样本的预测结果与特征分布。"
                  : "支持集提供类别示例，查询集用于观察模型能否识别未见过的图像。"}
              </p>
            </section>
          )}
          {result && (
            <div className="lab4-run-result" aria-busy={busy}>
              <section
                className="lab4-result-summary"
                aria-labelledby="lab4-result-title"
              >
                <div>
                  <h2 ref={resultHeading} id="lab4-result-title" tabIndex={-1}>
                    本次分类结果
                  </h2>
                  <p>
                    {methodName(result.method)} · {result.way}-way{" "}
                    {result.shots}-shot · 种子 {result.seed}
                    {result.transductive ? " · 转导推断" : " · 归纳推断"}
                    {result.method === "calibrated" && lastRequest
                      ? ` · ${calibrationNames[lastRequest.calibration]}`
                      : ""}
                  </p>
                  {changed && (
                    <span className="lab4-stale">
                      参数已改变，以下保留上一次运行结果。
                    </span>
                  )}
                </div>
                <dl className="lab4-metrics">
                  <div>
                    <dt>查询准确率</dt>
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
                    <dt>本次耗时</dt>
                    <dd className="lab4-time">{duration(result.latency_ms)}</dd>
                  </div>
                </dl>
              </section>
              <p className="sr-only" role="status">
                推理完成，{result.total}张查询图像中{result.correct}
                张分类正确，准确率{percent(result.accuracy)}。
              </p>

              <section
                className="task-section lab4-sample-section"
                aria-labelledby="lab4-support-title"
              >
                <header className="lab4-section-title">
                  <div>
                    <h2 id="lab4-support-title">支持集</h2>
                    <p>模型可见的标注样本，每类 {result.shots} 张。</p>
                  </div>
                  <span>{result.support.length} 张图像</span>
                </header>
                <div className="lab4-support-grid">
                  {result.classes.map((item, index) => (
                    <div key={item.id} className="lab4-support-class">
                      <h3>
                        <i
                          style={{ background: classColor(index) }}
                          aria-hidden="true"
                        />
                        {item.name}
                      </h3>
                      <div
                        className={
                          result.shots === 1
                            ? "lab4-support-images single"
                            : "lab4-support-images"
                        }
                      >
                        {result.support
                          .filter((sample) => sample.class_id === item.id)
                          .map((sample) => (
                            <img
                              key={sample.id}
                              src={sample.image}
                              alt={`${item.name}的支持样本${sample.id}`}
                              width={84}
                              height={84}
                              loading="lazy"
                            />
                          ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              <section
                className="task-section lab4-sample-section"
                aria-labelledby="lab4-query-title"
              >
                <header className="lab4-section-title">
                  <div>
                    <h2 id="lab4-query-title">查询集预测</h2>
                    <p>真实标签仅用于结果核对，不参与本次适应与校准。</p>
                  </div>
                  <span>{result.queries.length} 张图像</span>
                </header>
                <div className="lab4-query-grid">
                  {visibleQueries.map((sample) => {
                    const correct = sample.predicted_id === sample.target_id;
                    return (
                      <figure
                        key={sample.id}
                        className={`lab4-query ${correct ? "correct" : "incorrect"}`}
                      >
                        <div className="lab4-query-image">
                          <img
                            src={sample.image}
                            alt={`查询样本${sample.id}`}
                            width={84}
                            height={84}
                            loading="lazy"
                          />
                          <span
                            className="lab4-verdict"
                            aria-label={correct ? "预测正确" : "预测错误"}
                          >
                            {correct ? (
                              <CheckCircle size={18} weight="fill" />
                            ) : (
                              <XCircle size={18} weight="fill" />
                            )}
                          </span>
                        </div>
                        <figcaption>
                          <strong>
                            {classNames.get(sample.predicted_id) ||
                              sample.predicted_id}
                          </strong>
                          <span>置信度 {percent(sample.confidence)}</span>
                          {!correct && (
                            <small>
                              实际：
                              {classNames.get(sample.target_id) ||
                                sample.target_id}
                            </small>
                          )}
                        </figcaption>
                      </figure>
                    );
                  })}
                </div>
                {result.queries.length > 12 && (
                  <button
                    className="lab4-expand"
                    onClick={() => setShowAllQueries((value) => !value)}
                    aria-expanded={showAllQueries}
                  >
                    {showAllQueries ? (
                      <CaretUp size={16} />
                    ) : (
                      <CaretDown size={16} />
                    )}
                    {showAllQueries
                      ? "收起查询图像"
                      : `查看全部 ${result.queries.length} 张查询图像`}
                  </button>
                )}
                <p className="lab4-chart-note">
                  置信度是模型的类别分数，不等于预测正确的保证；本轮准确率也不替代多任务评测。
                </p>
              </section>

              <section
                className="task-section"
                aria-labelledby="lab4-features-title"
              >
                <header>
                  <h2 id="lab4-features-title">
                    {result.prototype_role === "descriptive_class_centers"
                      ? "特征分布与类别中心"
                      : "特征分布与类别原型"}
                  </h2>
                  <p>
                    {result.prototype_role === "descriptive_class_centers"
                      ? "类别中心用于观察特征变化；MAML 的预测来自适应后的分类头。"
                      : result.feature_motion === "real_support_adaptation"
                        ? "比较同一组图像在支持集适应前后的真实特征变化。"
                        : result.method === "baseline" ||
                            result.calibration === "none"
                          ? "根据支持样本的特征均值分类，本次不执行校准。"
                          : "图像特征保持不变，比较校准前后的类别原型位置。"}
                  </p>
                </header>
                <FeatureComparison result={result} />
              </section>
              {result.steps.length > 0 && (
                <section
                  className="lab4-process"
                  aria-labelledby="lab4-process-title"
                >
                  <h2 id="lab4-process-title">推理过程</h2>
                  <ol>
                    {result.steps.map((step, index) => (
                      <li key={`${index}-${step.name}`}>
                        <span>{String(index + 1).padStart(2, "0")}</span>
                        <div>
                          <h3>{step.name}</h3>
                          <p>{step.detail}</p>
                        </div>
                      </li>
                    ))}
                  </ol>
                </section>
              )}
            </div>
          )}
        </>
      )}

      {snapshot && currentSection.id === "protocol" && (
        <>
          <section
            className="task-section lab4-protocol"
            aria-labelledby="lab4-protocol-title"
          >
            <header>
              <h2 id="lab4-protocol-title">数据与任务划分</h2>
              <p>{snapshot.dataset.protocol_note}</p>
            </header>
            <dl className="lab4-protocol-facts">
              <div>
                <dt>数据集</dt>
                <dd>{snapshot.dataset.name}</dd>
              </div>
              <div>
                <dt>图像尺寸</dt>
                <dd>
                  {snapshot.dataset.image_size} × {snapshot.dataset.image_size}
                </dd>
              </div>
              <div>
                <dt>训练 / 验证 / 测试类别</dt>
                <dd>
                  {snapshot.dataset.train_classes} /{" "}
                  {snapshot.dataset.val_classes} /{" "}
                  {snapshot.dataset.test_classes}
                </dd>
              </div>
              <div>
                <dt>每个训练类别使用</dt>
                <dd>{snapshot.dataset.training_images_per_class} 张</dd>
              </div>
            </dl>
            <div className="lab4-protocol-rule">
              <strong>支持集与查询集分开</strong>
              <p>
                每个任务从测试类别中选择 5 类。支持集提供每类 1 张或 5
                张带标签图像；查询集使用其他图像。验证集用于选择设置，测试结果用于最终比较。
              </p>
            </div>
          </section>
          <section
            className="lab4-methods"
            aria-labelledby="lab4-methods-title"
          >
            <h2 id="lab4-methods-title">方法设计</h2>
            {snapshot.methods.map((item, index) => (
              <article key={item.id}>
                <span className="lab4-method-index">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div>
                  <h3>{item.name}</h3>
                  <p>{item.description}</p>
                </div>
              </article>
            ))}
          </section>
          <section
            className="task-section"
            aria-labelledby="lab4-calibration-title"
          >
            <header>
              <h2 id="lab4-calibration-title">均值方差校准与原型更新</h2>
              <p>
                极少样本的统计量容易偏移；基础类别提供先验，未标注查询样本提供任务内分布信息。
              </p>
            </header>
            <ol className="lab4-explanation">
              <li>
                <strong>估计特征分布</strong>
                <span>
                  用训练类别的统计量约束支持集均值与方差；1-shot
                  的方差估计需要先验。
                </span>
              </li>
              <li>
                <strong>调整类别距离</strong>
                <span>
                  利用收缩后的类条件方差计算距离，减少不稳定维度的影响。
                </span>
              </li>
              <li>
                <strong>更新类别原型</strong>
                <span>
                  利用未标注查询特征修正原型，属于转导式推断；查询标签不参与更新。
                </span>
              </li>
            </ol>
            <p className="lab4-loss">
              <span>训练目标</span>
              <code>
                L = L<sub>CE</sub> + λ L<sub>align</sub>
              </code>
              <small>
                Baseline
                只使用分类损失；校准骨干额外加入分布对齐正则项。推理策略消融固定校准骨干；去除分布对齐使用单独的训练对照。
                {snapshot.model.alignment_weight !== undefined
                  ? `本轮 λ = ${snapshot.model.alignment_weight}。`
                  : ""}
              </small>
            </p>
          </section>
          <section
            className="task-section"
            aria-labelledby="lab4-backbone-title"
          >
            <header>
              <h2 id="lab4-backbone-title">本轮模型与训练</h2>
              <p>
                {snapshot.model.name} · 通道宽度{" "}
                {snapshot.model.channels.join(" / ")} ·{" "}
                {snapshot.model.parameters.toLocaleString("zh-CN")} 个参数
              </p>
            </header>
            {snapshot.training.length ? (
              <TrainingCurve rows={snapshot.training} />
            ) : (
              <p className="lab4-empty-inline">训练曲线尚未生成。</p>
            )}
          </section>
          <section className="lab4-continuity">
            <h2>与原平台的衔接</h2>
            <p>
              实验一至三研究股权关系与证据推理。实验四使用独立图像数据研究小样本特征估计，复用平台入口、接口和结果查看流程。mini-ImageNet
              的分类结果不能直接作为企业材料识别效果。
            </p>
          </section>
        </>
      )}

      {snapshot && currentSection.id === "evaluation" && (
        <>
          <div className="lab4-evaluation-header">
            <div>
              <h2>多任务评测</h2>
              <p>读取已保存的实测结果，比较相同类别划分与抽样协议。</p>
            </div>
            <div
              className="lab4-shot-selector"
              role="group"
              aria-label="评测样本数"
            >
              {([1, 5] as const).map((value) => (
                <button
                  key={value}
                  onClick={() => setEvaluationShots(value)}
                  aria-pressed={evaluationShots === value}
                >
                  {value}-shot
                </button>
              ))}
            </div>
          </div>
          {comparisonRows.length ? (
            <section
              className="task-section"
              aria-labelledby="lab4-comparison-title"
            >
              <header>
                <h2 id="lab4-comparison-title">
                  方法对比 · 5-way {evaluationShots}-shot
                </h2>
                <p>
                  比较平均准确率与 95% 置信区间。20%
                  为五分类随机猜测的期望水平。
                </p>
              </header>
              <ResultBars rows={comparisonRows} />
              <MetricTable rows={comparisonRows} />
              <p className="lab4-chart-note">
                计时范围不同，不能据此直接排列方法速度：原型方法统计缓存特征上的分类，MAML
                统计图像编码与支持集适应。现场推理的耗时均包含图像编码。
              </p>
            </section>
          ) : (
            <section className="lab4-not-ready">
              <h2>这组对比结果尚未生成</h2>
              <p>完成真实评测后重新读取，页面会显示准确率、置信区间与耗时。</p>
            </section>
          )}
          {ablationRows.length ? (
            <section
              className="task-section"
              aria-labelledby="lab4-ablation-title"
            >
              <header>
                <h2 id="lab4-ablation-title">校准策略消融</h2>
                <p>
                  逐项观察均值、方差与原型更新的贡献，不预设完整方案一定更好。
                </p>
              </header>
              <MetricTable rows={ablationRows} ablation />
            </section>
          ) : (
            <section className="task-section">
              <h2>校准策略消融</h2>
              <p className="lab4-empty-inline">该样本数的消融结果尚未生成。</p>
            </section>
          )}
          {snapshot.limitations.length > 0 && (
            <section
              className="lab4-limitations"
              aria-labelledby="lab4-limitations-title"
            >
              <h2 id="lab4-limitations-title">结果的适用范围</h2>
              <ul>
                {snapshot.limitations.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </section>
          )}
          <details className="lab4-provenance">
            <summary>实验记录</summary>
            <dl>
              <div>
                <dt>随机种子</dt>
                <dd>{snapshot.provenance.seed}</dd>
              </div>
              {snapshot.provenance.run_id && (
                <div>
                  <dt>运行编号</dt>
                  <dd>{snapshot.provenance.run_id}</dd>
                </div>
              )}
              <div>
                <dt>模型指纹 SHA-256</dt>
                <dd>
                  <code>
                    {snapshot.provenance.checkpoint_sha256 || "尚未生成"}
                  </code>
                </dd>
              </div>
            </dl>
          </details>
        </>
      )}
    </main>
  );
}
