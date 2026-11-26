import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import pptxgen from "pptxgenjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const arg = (name, fallback) => {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
};
const legacy = path.join(root, "源码/outputs/lab04/history/v1");
const artifacts = arg("--artifacts", fs.existsSync(legacy) ? legacy : path.join(root, "源码/outputs/lab04"));
const assets = arg("--assets", "");
const temporary = arg("--tmp", "");
if (!assets || !temporary) throw new Error("Provide --assets and --tmp directories for verified captures and private rendering.");
const result = JSON.parse(fs.readFileSync(path.join(artifacts, "results.json"), "utf8"));
if (result.status !== "complete") throw new Error("Measured results are not complete.");
if (result.schema_version !== 1) throw new Error("This historical slide generator describes v1; use the v2 addendum for updated results.");
const output = path.join(root, "deliverables/lab4");
fs.mkdirSync(output, { recursive: true });
fs.mkdirSync(temporary, { recursive: true });
const row = (method, shots) => {
  const value = result.rows.find(r => r.method === method && r.shots === shots);
  if (!value || !Number.isFinite(value.accuracy)) throw new Error(`Missing measured result: ${method}/${shots}`);
  return value;
};
const percent = n => (100 * n).toFixed(2) + "%";
const interval = r => `${percent(r.accuracy)} ± ${(100 * r.ci95).toFixed(2)}`;
const delta = r => `${r.paired_delta >= 0 ? "+" : ""}${(100 * r.paired_delta).toFixed(2)}`;
const full1 = row("full", 1), full5 = row("full", 5);
const robust = r => Math.abs(r.paired_delta) > r.paired_ci95;
const protocol = result.protocol;
const epochs = result.training.warm_ce.epochs.length;
const branchEpochs = result.training.aligned.epochs.length;
const image = name => {
  const file = path.join(assets, name);
  if (!fs.existsSync(file)) throw new Error(`Capture missing: ${file}`);
  return file;
};
const sources = {
  data: "Ravi & Larochelle (2017), Optimization as a Model for Few-Shot Learning. https://openreview.net/pdf?id=rJY0-Kcll\nDataset: https://github.com/yaoyao-liu/mini-imagenet-tools",
  maml: "Finn, Abbeel & Levine (2017), Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. https://proceedings.mlr.press/v70/finn17a.html",
  proto: "Snell, Swersky & Zemel (2017), Prototypical Networks for Few-shot Learning. https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html",
  dc: "Yang, Liu & Xu (2021), Free Lunch for Few-shot Learning: Distribution Calibration. https://openreview.net/forum?id=JWOiYxMG92s\nOur implementation transfers means and diagonal variances, without Tukey transformation, Gaussian augmentation or LR/SVM; it is not a reproduction of DC.",
};
const narration = [
  "这次实验研究小样本图像分类中的特征校准。我们使用轻量残差网络，比较监督学习基线、MAML元学习和特征校准方案，并把实际模型接入原有的统一实验平台。",
  "小样本任务每类只有一张或五张带标签的支持图片，却要识别新的查询图片。样本过少会让类均值和方差估计不稳定。实验重点是利用训练类别的统计信息，改善新类别的判断。",
  `数据采用标准的六十四、十六、二十类划分，类别之间互不重叠。训练使用全部三万八千四百张图片。验证和测试每类抽取${protocol.evaluation_pool_per_class}张，每种样本设置评测${protocol.test_episodes_per_shot}个相同任务，测试查询标签只用于评分。`,
  `主干保留十二层卷积，通道缩窄到八、十六、二十四、三十二。监督暖启动之后，两条分支使用相同的图片、增强与训练步数，唯一差别是是否加入均值方差对齐损失，从而分离这项改动的影响。`,
  `MAML学习可以快速适应新类别的参数初始化。内循环用支持集更新全网络，外循环通过查询损失反向更新初始化。这里保留二阶梯度，共训练${result.training.maml.tasks}个任务；测试阶段再用支持集适应。`,
  "校准方案先借用相近基础类的均值和方差信息，再以未标注查询图片的软分配更新原型。一张支持图片无法估计方差，因此必须引入先验。原型更新属于转导推理，不会读取查询的真实标签。",
  `完整方案的一样本准确率为${(full1.accuracy * 100).toFixed(1)}%，五样本为${(full5.accuracy * 100).toFixed(1)}%。一样本差值的配对区间跨零，尚不能确认提升；五样本相对基线下降约${Math.abs(full5.paired_delta * 100).toFixed(1)}个百分点。`,
  `固定对齐主干的消融显示，方差校准表现较弱，仅原型更新的一样本准确率为${(row("prototype_only", 1).accuracy * 100).toFixed(1)}%。它比基线增加${(row("prototype_only", 1).paired_delta * 100).toFixed(1)}个百分点，但完整组合在五样本任务中下降，说明模块组合需要验证。`,
  "模型已在本机CPU部署，并接入原平台实验四入口。可以固定种子，对相同任务切换算法，查看支持图片、分类结果和原型变化。每次点击都会实际编码图片并运行推理，页面也提供完整对比和消融结果。",
  "实验验证了这套轻量方法的可运行性，也暴露了小模型和有限元训练预算的限制。结果只对应一个训练种子，不能当成论文复现分数。下一步需要增加训练种子，扩大主干，并验证真实业务数据。",
];

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "第33组";
pptx.subject = "人工智能课堂报告4：mini-ImageNet小样本学习与特征校准";
pptx.title = "面向小样本场景的深度神经网络特征校准方法研究";
pptx.company = "华南农业大学";
pptx.lang = "zh-CN";
pptx.theme = { headFontFace: "Microsoft YaHei", bodyFontFace: "Microsoft YaHei", lang: "zh-CN" };
const C = { paper: "F8F3E9", ink: "302A24", muted: "74685A", accent: "914021", rule: "DED1C1", blue: "506B74", olive: "7A8461", white: "FFFDFA" };
const font = "Microsoft YaHei";
function text(slide, value, x, y, w, h, size = 22, options = {}) {
  slide.addText(value, { x, y, w, h, fontFace: font, fontSize: size, color: C.ink,
    margin: 0, breakLine: false, valign: "mid", paraSpaceAfterPt: 0, ...options });
}
const sub = value => ({ text: value, options: { subscript: true } });
function base(title, index, source = "") {
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  if (title) text(slide, title, 0.72, 0.50, 11.9, 0.68, 30, { bold: true });
  text(slide, "人工智能课堂报告4 · 第33组", 0.74, 7.04, 8.5, 0.20, 10.5, { color: C.muted });
  text(slide, `${index} / 10`, 11.67, 7.03, 0.86, 0.22, 11, { color: C.muted, align: "right" });
  slide.addNotes(`${narration[index - 1]}\n\n来源与实验边界\n${source}\n实测数据：源码/outputs/lab04/results.json\n模型结果完成时间：${result.completed_utc}`);
  return slide;
}
function note(slide, value, y = 6.36) { text(slide, value, 0.76, y, 11.82, 0.48, 13.5, { color: C.muted }); }
function table(slide, rows, x, y, w, widths, size = 17, rowH = 0.56) {
  const formatted = rows.map((r, ri) => r.map(cell => ({ text: String(cell), options: ri === 0
    ? { bold: true, color: C.white, fill: C.accent }
    : { color: C.ink, fill: ri % 2 ? C.white : "F1EADF" } })));
  slide.addTable(formatted, { x, y, w, colW: widths, rowH, fontFace: font, fontSize: size,
    margin: [0.12, 0.16, 0.12, 0.16], border: { type: "solid", pt: 0.5, color: C.rule },
    color: C.ink, fill: C.white, valign: "middle", autoPage: false });
}
function picture(slide, file, x, y, w, h) {
  const png = fs.readFileSync(file);
  if (png.subarray(1, 4).toString("ascii") !== "PNG") throw new Error("Verified screenshots must use PNG.");
  const sourceW = png.readUInt32BE(16), sourceH = png.readUInt32BE(20);
  const scale = Math.min(w / sourceW, h / sourceH);
  slide.addImage({ path: file, x: x + (w - sourceW * scale) / 2, y: y + (h - sourceH * scale) / 2,
    w: sourceW * scale, h: sourceH * scale });
}

{
  const s = base("", 1, sources.data);
  text(s, "面向小样本场景的", 0.78, 1.20, 10.7, 0.65, 31, { color: C.accent });
  text(s, "深度网络特征校准", 0.74, 2.00, 11.85, 1.10, 44, { bold: true });
  text(s, "ResNet12 · MAML · 均值方差校准与原型更新", 0.80, 3.48, 11.65, 0.63, 21, { color: C.muted });
  text(s, "mini-ImageNet  /  5-way 1-shot & 5-shot", 0.80, 4.61, 11.5, 0.6, 23);
  text(s, "软件工程5班 · 第33组", 0.81, 5.65, 11.2, 0.45, 18, { color: C.muted });
}
{
  const s = base("小样本分类任务", 2, sources.proto);
  text(s, "每类 1 张或 5 张支持图片", 0.78, 1.47, 11.75, 0.63, 29, { bold: true, color: C.accent });
  picture(s, image("support.png"), 0.78, 2.35, 11.78, 2.34);
  text(s, "利用少量支持图片，预测未见的查询图片", 0.81, 5.03, 11.7, 0.65, 24);
  note(s, "问题：支持样本太少，类均值与方差估计容易偏移。图片来自本次实际演示任务。", 6.12);
}
{
  const s = base("数据与评测协议", 3, sources.data);
  table(s, [["数据划分", "互斥类别", "本次使用"],
    ["训练", "64 类", `${protocol.training_images.toLocaleString()} 张，600 张/类`],
    ["验证", "16 类", `${protocol.evaluation_pool_per_class} 张/类，选择超参数`],
    ["测试", "20 类", `${protocol.evaluation_pool_per_class} 张/类，仅最终评测`]], 0.78, 1.49, 11.77, [2.75, 2.65, 6.37], 21, 0.79);
  text(s, `5-way  /  1-shot、5-shot  /  每类 15 张查询图片`, 0.81, 5.00, 11.7, 0.55, 22);
  text(s, `各 ${protocol.test_episodes_per_shot} 个配对任务，报告均值与 95% 置信区间`, 0.81, 5.73, 11.7, 0.55, 22);
  note(s, "区间仅反映任务抽样变化；本次使用一个训练种子。", 6.46);
}
{
  const s = base("轻量 ResNet12 与分布对齐", 4, sources.proto);
  text(s, "12 层卷积 · 32 维特征 · CE 模型 50,352 参数", 0.8, 1.5, 11.7, 0.62, 28, { color: C.accent, bold: true });
  text(s, "4 个残差块，通道数 8 / 16 / 24 / 32，GroupNorm", 0.82, 2.41, 11.65, 0.53, 22);
  text(s, [{ text: "L = L" }, sub("CE"), { text: " + λ [ MSE(μ" }, sub("a"),
    { text: ", μ" }, sub("b"), { text: ") + MSE(v" }, sub("a"), { text: ", v" }, sub("b"), { text: ") ]" }],
    0.85, 3.45, 11.55, 0.7, 26, { fontFace: "Cambria Math" });
  text(s, "a、b 是同一批图片的两种增强；μ、v 是特征均值与方差", 0.83, 4.36, 11.63, 0.52, 19, { color: C.muted });
  text(s, `${epochs} 轮共同暖启动后，双方各 ${branchEpochs} 轮相同 CE 训练`, 0.82, 5.27, 11.55, 0.6, 23);
  note(s, `公平对照：相同图片、顺序、增强和步数，仅 λ = 0 与 λ = ${result.training.aligned.alignment_weight} 不同。`);
}
{
  const s = base("MAML 学习可快速适应的初始化", 5, sources.maml + "\nCE backbone initialization. The outer-loop equation illustrates gradient-based meta-learning; the implementation uses Adam.");
  text(s, "支持集内循环", 0.80, 1.53, 11.8, 0.6, 26, { color: C.accent, bold: true });
  text(s, [{ text: "θ′ = θ − α ∇" }, sub("θ"), { text: " L" }, sub("support"), { text: "(θ)" }],
    0.84, 2.34, 11.3, 0.72, 31, { fontFace: "Cambria Math" });
  text(s, "查询集外循环", 0.80, 3.55, 11.8, 0.6, 26, { color: C.accent, bold: true });
  text(s, [{ text: "θ ← θ − β ∇" }, sub("θ"), { text: " L" }, sub("query"), { text: "(θ′)" }],
    0.84, 4.35, 11.3, 0.72, 31, { fontFace: "Cambria Math" });
  note(s, `完整二阶梯度；全网络适应。元训练 ${result.training.maml.tasks} 个任务，训练内循环 1 步，测试 3 步。`, 5.80);
  note(s, "元训练使用训练类别；测试阶段的查询标签不参与参数更新。", 6.36);
}
{
  const s = base("均值方差校准与原型更新", 6, sources.dc + "\n" + sources.proto);
  table(s, [["步骤", "做法", "作用"],
    ["均值校准", "支持类均值向相近基础类统计收缩", "降低少量样本的均值偏移"],
    ["方差校准", "支持集方差与基础类先验加权", "1-shot 时仍有可用方差"],
    ["原型更新", "以未标注查询特征的软分配更新", "结合当前任务分布"]], 0.77, 1.48, 11.81, [2.18, 5.68, 3.95], 18.5, 0.89);
  text(s, "原型更新使用整批无标签查询，属于转导推理", 0.82, 5.40, 11.7, 0.63, 23, { color: C.accent });
  note(s, "方差改变距离度量，样本散点可以保持不动。本方法受统计迁移启发，不是原论文 DC 的完整复现。", 6.31);
}
{
  const s = base("三种方法的实测对比", 7);
  const names = ["Baseline", "MAML", "完整校准"];
  const ids = ["baseline", "maml", "full"];
  s.addChart(pptx.ChartType.bar, [
    { name: "1-shot", labels: names, values: ids.map(id => +(row(id, 1).accuracy * 100).toFixed(2)) },
    { name: "5-shot", labels: names, values: ids.map(id => +(row(id, 5).accuracy * 100).toFixed(2)) },
  ], { x: 0.78, y: 1.44, w: 6.45, h: 4.61, catAxisLabelFontFace: font, valAxisLabelFontFace: font,
    catAxisLabelFontSize: 17, valAxisLabelFontSize: 13, showTitle: false,
    chartColors: [C.accent, C.blue], showLegend: true, legendPos: "b", legendFontFace: font, legendFontSize: 15,
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20, valAxisTitle: "准确率 / %", showValue: true,
    dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", dataLabelColor: C.ink, dataLabelBkgrdColor: C.paper,
    dataLabelFontFace: font, dataLabelFontSize: 13, showCatName: false,
    barDir: "col", showBorder: false, showShadow: false,
    catAxisLineColor: C.rule, valAxisLineColor: C.rule,
    valGridLine: { color: C.rule, width: 0.5 },
    chartArea: { fill: { color: C.paper } }, plotArea: { fill: { color: C.paper } },
  });
  table(s, [["方法", "1-shot", "5-shot"], ...ids.map((id, i) => [names[i], interval(row(id, 1)).replace(" ± ", "\n± "), interval(row(id, 5)).replace(" ± ", "\n± ")])],
    7.48, 1.83, 5.07, [1.65, 1.71, 1.71], 17, 0.84);
  text(s, `完整方案相对基线：\n1-shot ${delta(full1)} pp，配对区间跨零\n5-shot ${delta(full5)} pp，配对区间未跨零`, 7.63, 5.33, 4.88, 1.02, 15.5, { color: C.accent });
  note(s, "均值以 % 表示，± 后为 95% 区间半宽，pp 为百分点。本次窄通道模型实测，不用于对标论文成绩。", 6.39);
}
{
  const s = base("消融实验分离各项改动", 8);
  const ids = ["alignment_only", "mean_only", "variance_only", "prototype_only", "mean_variance", "full", "no_alignment_full"];
  const labels = ["仅分布对齐训练", "仅均值校准", "仅方差校准", "仅原型更新", "均值与方差", "完整方案", "去除对齐训练"];
  table(s, [["设置", "1-shot", "5-shot"], ...ids.map((id, i) => [labels[i], interval(row(id, 1)), interval(row(id, 5))])],
    0.79, 1.41, 11.73, [4.11, 3.81, 3.81], 18, 0.56);
  note(s, "前六行固定对齐主干；最后一行使用 CE-only 主干。原型更新项包含转导信息，不能只归因于均值方差校准。", 6.27);
}
{
  const s = base("接入原有平台的实验四入口", 9);
  picture(s, image("platform.png"), 0.77, 1.39, 11.8, 4.62);
  text(s, "固定种子切换算法，查看真实预测、特征与原型", 0.80, 6.15, 11.69, 0.55, 21, { color: C.accent });
}
{
  const s = base("结论与后续验证", 10);
  text(s, `5-shot 完整方案 ${percent(full5.accuracy)}`, 0.8, 1.57, 11.75, 0.68, 32, { bold: true, color: C.accent });
  text(s, `相对基线 ${delta(full5)} 个百分点${robust(full5) ? "，配对区间未跨零" : "，配对区间仍跨零"}`, 0.83, 2.57, 11.63, 0.62, 23);
  text(s, "当前边界：窄主干、一个训练种子、有限元训练任务", 0.84, 3.80, 11.62, 0.62, 23);
  text(s, "后续：多种子复验、增加模型容量、独立业务数据验证", 0.84, 4.79, 11.62, 0.62, 23);
  note(s, "本机 CPU 部署验证，未作手机或嵌入式测试；图像结果不代表股权风险识别能力。", 6.04);
}

const final = path.join(output, "小样本学习与特征校准.pptx");
// PptxGenJS 4.0.1 declares one slide master per slide but writes only actual masters.
// Reuse its ZIP dependency and remove declarations whose master parts do not exist.
const require = createRequire(import.meta.url);
const JSZip = createRequire(require.resolve("pptxgenjs"))("jszip");
const packageZip = await JSZip.loadAsync(await pptx.write({ outputType: "nodebuffer" }));
const contentTypes = await packageZip.file("[Content_Types].xml").async("string");
packageZip.file("[Content_Types].xml", contentTypes.replace(
  /<Override\b[^>]*\bPartName="(\/ppt\/slideMasters\/[^\"]+)"[^>]*\/>/g,
  (declaration, partName) => packageZip.file(partName.slice(1)) ? declaration : "",
));
fs.writeFileSync(final, await packageZip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
fs.writeFileSync(path.join(temporary, "ppt_narration.json"), JSON.stringify({
  duration_seconds: 180, voice: "zh-CN-YunxiNeural", slides: narration.map((text, i) => ({ slide: i + 1, text })),
  source_result_sha256: (await import("node:crypto")).createHash("sha256").update(fs.readFileSync(path.join(artifacts, "results.json"))).digest("hex"),
}, null, 2), "utf8");
console.log(JSON.stringify({ file: final, slides: pptx._slides.length, narration_characters: narration.join("").length }));
