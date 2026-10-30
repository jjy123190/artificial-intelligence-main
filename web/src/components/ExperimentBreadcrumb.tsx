import { ArrowLeft } from "@phosphor-icons/react";
import { labHref } from "../routes";
const names: Record<string, string> = {
  modeling: "知识表示与建模",
  deterministic: "确定性推理",
  uncertain: "不确定性推理",
  optimization: "算法优化与对比实验",
  hybrid: "混合推理与创新验证",
};
export default function ExperimentBreadcrumb({
  moduleId,
  title,
}: {
  moduleId: string;
  title: string;
}) {
  return (
    <nav className="course-breadcrumb" aria-label="当前位置">
      <a href={labHref("lab02")}>
        <ArrowLeft size={16} />
        实验二
      </a>
      <span>/</span>
      <a href={labHref("lab02", moduleId)}>{names[moduleId]}</a>
      <span>/</span>
      <span>{title}</span>
    </nav>
  );
}
