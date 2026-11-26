/** Stable routes: report pages are independent destinations, not scroll anchors. */
export const reportPages = [
  {
    id: "evaluation",
    label: "性能对比",
    title: "先比较效果，再判断改善",
    description:
      "同一组任务、同一批图片。选择方法查看配对结论，点击图表也可切换。",
  },
  {
    id: "ablations",
    label: "模块消融",
    title: "拆开模块，看清贡献",
    description:
      "固定骨干的推理消融与训练对照分开解释，不把模块越多当作效果越好。",
  },
  {
    id: "reliability",
    label: "可靠性",
    title: "平均准确率之外，还有什么？",
    description:
      "任务波动、改善与下降、分数可靠性。只诊断固定预测，不反向调整测试参数。",
  },
  {
    id: "deployment",
    label: "端侧性能",
    title: "准确率的计算代价",
    description:
      "Windows 本机 CPU 的统一实测口径；各方法都重新编码图片，不用缓存特征计时排名。",
  },
  {
    id: "history",
    label: "历史与来源",
    title: "完整保留研究过程",
    description:
      "负结果、旧权重复测和来源指纹均保留。不同协议不能直接宣称配对提升。",
  },
] as const;
export type ReportPage = (typeof reportPages)[number]["id"];
export function asReportPage(id: string): ReportPage | undefined {
  return reportPages.find((page) => page.id === id)?.id;
}
