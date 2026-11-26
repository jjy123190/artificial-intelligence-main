import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, ArrowClockwise } from "@phosphor-icons/react";
import { request } from "../api";
import { validSnapshot } from "../lab-four/snapshot-contract";
import { labHref } from "../routes";
import type { CourseLab } from "../types";
import type { LabFourSnapshot } from "../lab-four/types";
import { TrainingCurve, ValidationCurve } from "../lab-four/charts";
import Evaluation from "../lab-four/Evaluation";
import { asReportPage, reportPages } from "../lab-four/navigation";
import InferenceWorkbench from "../lab-four/InferenceWorkbench";
import "../lab-four/lab-four.css";
import "../lab-four/workbench.css";
import "../lab-four/evidence-pages.css";

const sections = [
  { id: "inference", label: "现场推理" },
  { id: "protocol", label: "方法与协议" },
  { id: "evaluation", label: "对比与消融" },
];
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
  const loadPending = useRef<AbortController | null>(null);
  const snapshotRef = useRef<LabFourSnapshot | null>(null);
  const load = useCallback(async () => {
    loadPending.current?.abort();
    const controller = new AbortController();
    loadPending.current = controller;
    setLoading(true);
    setLoadError("");
    try {
      const value = await request<unknown>("/api/labs/lab04/snapshot", {
        signal: controller.signal,
        timeoutMs: 20_000,
      });
      if (controller.signal.aborted || loadPending.current !== controller)
        return;
      if (!validSnapshot(value))
        throw new Error(
          "实验状态数据不完整或统计不一致，未更新页面，请重新读取。",
        );
      snapshotRef.current = value;
      setSnapshot(value);
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
      setLoading(false);
    }
  }, [active, load]);
  useEffect(
    () => () => {
      loadPending.current?.abort();
    },
    [],
  );

  const reportView = asReportPage(moduleId);
  const currentSection =
    sections.find(
      (section) => section.id === (reportView ? "evaluation" : moduleId),
    ) || sections[0];
  const currentLabel =
    reportPages.find((item) => item.id === reportView)?.label ||
    currentSection.label;

  return (
    <main
      id={active ? "main-content" : undefined}
      tabIndex={-1}
      className={`page-shell course-shell task-page lab4-page${reportView ? " lab4-report-page" : ""}`}
    >
      <nav className="course-breadcrumb" aria-label="当前位置">
        <a href={labHref("lab04")}>
          <ArrowLeft size={16} />
          实验四
        </a>
        <span>/</span>
        <span>{currentLabel}</span>
      </nav>
      <header className="course-header lab4-heading">
        <div>
          <span className="lab4-section-kicker">
            实验四 / FEW-SHOT LEARNING
          </span>
          <h1>小样本分类实验室</h1>
          <p>小样本图像分类与特征校准 · 从一次真实预测，到一组可复核的实验。</p>
        </div>
        <button
          className="icon-button lab4-reload"
          type="button"
          onClick={() => void load()}
          disabled={loading}
          title="重新读取实验状态"
          aria-label="重新读取实验状态"
        >
          <ArrowClockwise
            size={20}
            className={loading ? "lab4-spinning" : undefined}
          />
        </button>
      </header>
      <div className="lab4-model-strip">
        <span className="lab4-model-status">
          {snapshot?.available
            ? "模型就绪"
            : loading
              ? "连接模型中"
              : "检查模型状态"}
        </span>
        <span>mini-ImageNet</span>
        <span>ResNet12 · {snapshot?.model.channels.at(-1) || 32} 维</span>
        <span>本地 CPU 推理</span>
      </div>
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
          <span>
            {loadError}
            {snapshot && " 仍保留上一次通过校验的结果，不代表本次刷新成功。"}
          </span>
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

      {snapshot && (
        <div hidden={currentSection.id !== "inference"}>
          <InferenceWorkbench
            snapshot={snapshot}
            active={active && currentSection.id === "inference"}
          />
        </div>
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
            className="lab4-protocol-methods"
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
          {snapshot.study?.version === "v2" && (
            <section className="task-section lab4-v2-method">
              <header>
                <h2>后续优化：稳定特征空间，约束不可靠统计量</h2>
                <p>训练预算与算法模块分开记录，不把所有提升归因于校准。</p>
              </header>
              <ol className="lab4-explanation">
                <li>
                  <strong>训练类中心化</strong>
                  <span>
                    只用基础训练类估计中心，按锁定参数变换支持与查询特征。不利用测试类标签。
                  </span>
                </li>
                <li>
                  <strong>抑制方差负迁移</strong>
                  <span>
                    在共享距离尺度与类条件方差之间做保守混合，系数仅在验证类选择。
                  </span>
                </li>
                <li>
                  <strong>均衡转导先验</strong>
                  <span>
                    软分配迭代同时约束类别总质量。本实验每类15张查询符合均衡设定；真实不均衡任务需要另行验证。
                  </span>
                </li>
              </ol>
              <p className="lab4-chart-note">
                “仅原型更新”等消融保留固定的特征预处理；另设“仅基础类中心化”和“去除特征中心化”以区分贡献。校准前后散点显示的是同一套已预处理特征。
              </p>
            </section>
          )}
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
              <>
                <p className="lab4-curve-title">训练分类损失</p>
                <TrainingCurve rows={snapshot.training} />
                <ValidationCurve rows={snapshot.training} />
              </>
            ) : (
              <p className="lab4-empty-inline">训练曲线尚未生成。</p>
            )}
          </section>
        </>
      )}

      {snapshot && (
        <div hidden={!reportView}>
          <Evaluation snapshot={snapshot} view={reportView || "evaluation"} />
        </div>
      )}
    </main>
  );
}
