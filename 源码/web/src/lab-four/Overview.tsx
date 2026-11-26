import { useEffect, useState } from "react";
import {
  ArrowRight,
  ArrowLeft,
  Play,
  Flask,
  ChartBar,
  BookOpen,
} from "@phosphor-icons/react";
import { request } from "../api";
import { validSnapshot } from "./snapshot-contract";
import type { CourseLab } from "../types";
import type { LabFourSnapshot } from "./types";
import { percent } from "./charts";
import "./lab-four.css";
import "./workbench.css";
import "./evidence-pages.css";

export default function Overview({ lab }: { lab: CourseLab }) {
  const [snapshot, setSnapshot] = useState<LabFourSnapshot | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    void request<unknown>("/api/labs/lab04/snapshot", {
      signal: controller.signal,
      timeoutMs: 15000,
    })
      .then((value) => {
        if (controller.signal.aborted) return;
        if (!validSnapshot(value))
          throw new Error("实验状态数据不完整或统计不一致，请重试。");
        setSnapshot(value);
      })
      .catch((e) => {
        if (!controller.signal.aborted)
          setError(e instanceof Error ? e.message : "实验状态暂不可用");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [attempt]);
  const sections = [
    {
      id: "inference",
      icon: Play,
      title: "让模型真正预测一次",
      description:
        "从五类示例开始，检查75张图片的预测。三种方法，同一组样本，逐张核对。",
      action: "打开推理工作台",
    },
    {
      id: "protocol",
      icon: BookOpen,
      title: "理解方法，不只看分数",
      description:
        "ResNet12如何提取特征？校准改变了什么？从数据隔离到训练过程逐项解释。",
      action: "查看方法与协议",
    },
    {
      id: "evaluation",
      icon: ChartBar,
      title: "用配对实验检验改善",
      description:
        "平均准确率、置信区间、模块消融与原始负结果，让结论经得起复核。",
      action: "阅读评测结果",
    },
  ];
  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="page-shell course-shell lab4-page lab4-overview"
    >
      <nav className="course-breadcrumb" aria-label="当前位置">
        <a href="#/home">
          <ArrowLeft size={16} />
          选择实验
        </a>
        <span>/</span>
        <span>实验四</span>
      </nav>
      <header className="lab4-overview-hero">
        <div>
          <span className="lab4-section-kicker">
            EXPERIMENT 04 · 小样本学习
          </span>
          <h1>
            少一点示例，
            <br />
            多一点泛化能力。
          </h1>
          <p>
            用轻量 ResNet12
            认识未见过的图像类别，研究均值方差校准与原型自适应究竟能改善什么。
          </p>
          <a className="button primary" href="#/labs/lab04/inference">
            <Play size={18} weight="fill" />
            开始分类实验
            <ArrowRight size={18} />
          </a>
          <span className="lab4-overview-live">
            {error
              ? "状态暂不可用，仍可查看方法"
              : loading
                ? "正在核对实验状态"
                : snapshot?.available
                  ? "模型就绪 · 本机真实推理"
                  : "模型准备中"}
          </span>
        </div>
        <div className="lab4-overview-model" aria-label="实验技术路线">
          <div className="lab4-overview-model-title">
            <Flask size={21} />
            <strong>从像素，到类别</strong>
            <span>{snapshot?.model.name || "轻量 ResNet12"}</span>
          </div>
          <ol>
            <li>
              <span>输入图像</span>
              <strong>84 × 84</strong>
              <small>mini-ImageNet · RGB</small>
            </li>
            <li>
              <span>特征编码</span>
              <strong>12 层卷积</strong>
              <small>
                {snapshot ? snapshot.model.channels.join(" / ") : "四个残差块"}
              </small>
            </li>
            <li>
              <span>特征校准</span>
              <strong>均值 · 方差 · 原型</strong>
              <small>训练类先验 + 无标签查询</small>
            </li>
            <li>
              <span>新类识别</span>
              <strong>5-way · 1 / 5-shot</strong>
              <small>支持标签可见，查询标签隔离</small>
            </li>
          </ol>
          <p>
            与实验一至三共用平台，独立验证视觉任务。图像分类成绩不代表企业股权分析效果。
          </p>
        </div>
      </header>
      {error && (
        <div className="error-banner lab4-overview-error" role="alert">
          <span>
            {error}
            {snapshot && " 仍保留上次有效结果。"}
          </span>
          <button
            className="button secondary"
            disabled={loading}
            onClick={() => setAttempt((n) => n + 1)}
          >
            重试实验状态
          </button>
        </div>
      )}
      <nav className="lab4-overview-paths" aria-label="实验四任务入口">
        {sections.map((s) => (
          <a key={s.id} href={`#/labs/${lab.id}/${s.id}`}>
            <s.icon size={24} />
            <h2>{s.title}</h2>
            <p>{s.description}</p>
            <span>
              {s.action}
              <ArrowRight size={17} />
            </span>
          </a>
        ))}
      </nav>
      <nav className="lab4-overview-shortcuts" aria-label="实验专题入口">
        <a href="#/labs/lab04/ablations">
          模块消融 <ArrowRight size={14} />
        </a>
        <a href="#/labs/lab04/reliability">
          可靠性诊断 <ArrowRight size={14} />
        </a>
        <a href="#/labs/lab04/deployment">
          端侧性能 <ArrowRight size={14} />
        </a>
        <a href="#/labs/lab04/history">
          历史与来源 <ArrowRight size={14} />
        </a>
      </nav>
      {snapshot && snapshot.comparisons.length > 0 && (
        <section
          className="lab4-overview-evidence"
          aria-labelledby="lab4-overview-evidence-title"
        >
          <div>
            <span className="lab4-section-kicker">MEASURED, NOT ASSUMED</span>
            <h2 id="lab4-overview-evidence-title">先看证据，再谈改善。</h2>
            <p>
              完整方案不一定更好。这里展示默认轻量方案；其他策略与独立较宽模型在评测页逐项比较，不把单次演示当作平均成绩。
            </p>
            <a href="#/labs/lab04/evaluation">
              查看性能对比与独立消融页
              <ArrowRight size={15} />
            </a>
          </div>
          <div className="table-scroll">
            <table>
              <caption className="sr-only">当前实测准确率概览</caption>
              <thead>
                <tr>
                  <th scope="col">设置</th>
                  <th scope="col">Baseline</th>
                  <th scope="col">
                    {snapshot.graph_extension?.available
                      ? "单视图图传播"
                      : "完整校准"}
                  </th>
                </tr>
              </thead>
              <tbody>
                {([1, 5] as const).map((shots) => (
                  <tr key={shots}>
                    <th scope="row">{shots}-shot</th>
                    {[
                      "baseline",
                      snapshot.graph_extension?.available ? "graph" : "full",
                    ].map((m) => {
                      const row = snapshot.comparisons.find(
                        (r) => r.shot === shots && r.method === m,
                      );
                      return (
                        <td key={m}>
                          {row ? percent(row.accuracy, 2) : "未记录"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}
