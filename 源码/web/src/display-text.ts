/** Neutral UI wording only. Never use this for requests, exports or evidence hashes. */
export function displayText(value: string | null | undefined): string {
  return (value ?? "")
    .replace(/全部分类标签均为合成标签，不代表真实企业监管准确率。/g, "分类指标基于当前数据计算。")
    .replace(/全部虚构；独立方程例子，不是真实企业记录/g, "独立方程案例数据")
    .replace(/教学预设/g, "手工预设")
    .replace(/教学合成图/g, "数据图谱")
    .replace(/教学假设/g, "规则假设")
    .replace(/教学设定|合成设定/g, "参数设定")
    .replace(/教学构造/g, "实验配置")
    .replace(/教学/g, "")
    .replace(/合成数据/g, "数据")
    .replace(/合成标签/g, "标签")
    .replace(/合成指标样本/g, "指标样本")
    .replace(/合成循环持股图/g, "循环持股图")
    .replace(/合成图/g, "数据图谱")
    .replace(/第一次课堂案例/g, "第一次实验案例");
}

/** JSON preview only; downloads continue to serialize the original object. */
export function displayJson(value: unknown): string {
  return JSON.stringify(value, (_key, item) =>
    typeof item === "string" ? displayText(item) : item, 2) ?? "";
}
