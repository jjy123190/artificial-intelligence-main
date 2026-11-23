"""Build the lab 4 report from completed, independently checked episode results.

Usage (Windows, Python with python-docx and PyMuPDF):
    python build_report.py --artifacts <directory> --pdf --qa-dir <temporary-dir>

The default artifact directory is ../../源码/outputs/lab04 relative to this file.
No result is supplied by a fixture or hard-coded into the report. The script
refuses unfinished results and recomputes all table means, confidence intervals,
paired differences and timings from episodes.json before writing a document.
Word is opened invisibly only when --pdf is requested. PNGs are optional QA
intermediates, not part of the deliverable.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


REPO = Path(__file__).resolve().parents[2]
TITLE = "面向小样本场景的深度神经网络特征校准方法研究"
METHODS = (
    "baseline", "alignment_only", "mean_only", "variance_only",
    "prototype_only", "mean_variance", "full", "no_alignment_full", "maml",
)
LABELS = {
    "baseline": "Baseline",
    "alignment_only": "仅分布正则",
    "mean_only": "正则＋均值",
    "variance_only": "正则＋方差",
    "prototype_only": "正则＋原型更新",
    "mean_variance": "正则＋均值方差",
    "full": "完整校准",
    "no_alignment_full": "完整校准去正则",
    "maml": "二阶 MAML",
}
TRANSDUCTIVE = {"prototype_only", "full", "no_alignment_full"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, description):
    require(math.isfinite(float(actual)), f"Nonfinite value: {description}")
    require(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10),
            f"Result mismatch for {description}: {actual} vs {expected}")


def mean_ci(values):
    center = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return center, half


def read_verified_results(directory):
    results_path, episodes_path = directory / "results.json", directory / "episodes.json"
    require(results_path.is_file() and episodes_path.is_file(),
            "Complete results.json and episodes.json are both required; no placeholder output is created.")
    result = json.loads(results_path.read_text(encoding="utf-8-sig"))
    episode_data = json.loads(episodes_path.read_text(encoding="utf-8-sig"))
    require(result.get("status") == "complete", "The experiment has not completed.")
    require(result.get("schema_version") == 1 and episode_data.get("schema_version") == 1,
            "Unsupported result schema.")
    protocol = result["protocol"]
    require(protocol["ways"] == 5 and protocol["shots"] == [1, 5], "Unexpected episode protocol.")
    require(protocol["queries_per_class"] == 15, "Unexpected query count.")
    require(protocol["no_test_tuning"], "The recorded protocol does not guarantee held-out test tuning.")
    require(result["training"]["maml"]["second_order"], "Report requires full second-order MAML.")
    require(result["runtime"]["device"] == "cpu", "This report describes a CPU experiment.")
    require(episode_data["seed"] == protocol["seed"] + 200, "Episode seed mismatch.")
    rows = {(r["method"], r["shots"]): r for r in result["rows"]}
    require(len(rows) == len(result["rows"]) == len(METHODS) * 2, "Missing or duplicate method rows.")
    expected_keys = {(method, shot) for method in METHODS for shot in (1, 5)}
    require(set(rows) == expected_keys, "Unexpected result methods.")
    episodes = episode_data["episodes"]
    count = int(protocol["test_episodes_per_shot"])
    require(count >= 2 and len(episodes) == 2 * count, "Incomplete evaluation episodes.")
    query_count = protocol["ways"] * protocol["queries_per_class"]
    for shot in (1, 5):
        selected = [e for e in episodes if e["shots"] == shot]
        require(len(selected) == count, f"Incomplete {shot}-shot episodes.")
        require({e["episode"] for e in selected} == set(range(count)), "Episode IDs are incomplete.")
        for episode in selected:
            support_ids, query_ids = episode["support_ids"], episode["query_ids"]
            require(len(episode["classes"]) == len(set(episode["classes"])) == protocol["ways"],
                    "Episode classes are not distinct.")
            require(len(support_ids) == len(set(support_ids)) == shot * protocol["ways"],
                    "Incorrect or duplicated support IDs.")
            require(len(query_ids) == len(set(query_ids)) == query_count, "Incorrect or duplicated query IDs.")
            require(set(support_ids).isdisjoint(query_ids), "Support and query sets overlap.")
            require(set(episode["methods"]) == set(METHODS), "Episode is missing a method.")
            for method in METHODS:
                entry = episode["methods"][method]
                acc, seconds = float(entry["accuracy"]), float(entry["seconds"])
                require(math.isfinite(acc) and 0 <= acc <= 1, "Invalid episode accuracy.")
                close(acc * query_count, round(acc * query_count), "integer query correctness")
                require(math.isfinite(seconds) and seconds >= 0, "Invalid timing.")
        baseline = [e["methods"]["baseline"]["accuracy"] for e in selected]
        for method in METHODS:
            row = rows[method, shot]
            values = [e["methods"][method]["accuracy"] for e in selected]
            acc, ci = mean_ci(values)
            delta, delta_ci = mean_ci([v - b for v, b in zip(values, baseline)])
            for key, value in (("accuracy", acc), ("ci95", ci),
                               ("ci95_low", max(0.0, acc - ci)),
                               ("ci95_high", min(1.0, acc + ci)),
                               ("paired_delta", delta), ("paired_ci95", delta_ci)):
                close(row[key], value, f"{method}/{shot}/{key}")
            close(row["mean_episode_seconds"],
                  statistics.mean(e["methods"][method]["seconds"] for e in selected),
                  f"{method}/{shot}/timing")
            require(row["episodes"] == count, "Result count mismatch.")
            require(bool(row["transductive"]) == (method in TRANSDUCTIVE), "Inference setting mismatch.")
            expected_timing = ("image_encoding_and_support_adaptation" if method == "maml"
                               else "inference_from_cached_features")
            require(row["timing_scope"] == expected_timing, "Timing scope mismatch.")
    return result, episodes, rows


def set_font(run, *, size=None, bold=None, east_asia="宋体", latin="Times New Roman"):
    run.font.name = latin
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn("w:eastAsia"), east_asia)
    fonts.set(qn("w:ascii"), latin)
    fonts.set(qn("w:hAnsi"), latin)


def style_font(style, size, *, bold=False, east_asia="宋体", latin="Times New Roman"):
    style.font.name = latin
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for key, value in (("eastAsia", east_asia), ("ascii", latin), ("hAnsi", latin)):
        fonts.set(qn("w:" + key), value)
    color = style.element.get_or_add_rPr().find(qn("w:color"))
    if color is not None:
        for attr in ("themeColor", "themeTint", "themeShade"):
            color.attrib.pop(qn("w:" + attr), None)


INLINE_SUBSCRIPTS = {
    "p_c": ("p", "c"), "n_c": ("n", "c"), "μ_b": ("μ", "b"), "v_b": ("v", "b"),
    "ν_c": ("ν", "c"), "s_c": ("s", "c"), "w_qc": ("w", "qc"),
    "p̃_c": ("p̃", "c"), "a_i": ("a", "i"), "b_i": ("b", "i"),
}


def add_text(p, text, *, size=None, bold=None):
    pattern = "(" + "|".join(re.escape(k) for k in INLINE_SUBSCRIPTS) + ")"
    for piece in re.split(pattern, text):
        if piece in INLINE_SUBSCRIPTS:
            base, index = INLINE_SUBSCRIPTS[piece]
            set_font(p.add_run(base), size=size, bold=bold)
            run = p.add_run(index)
            set_font(run, size=size, bold=bold)
            run.font.subscript = True
        elif piece:
            set_font(p.add_run(piece), size=size, bold=bold)


def paragraph(doc, text="", *, style=None, bold_lead=None, size=None, after=5, indent=True):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.widow_control = True
    if style is None and indent:
        p.paragraph_format.first_line_indent = Pt(22)
    if bold_lead and text.startswith(bold_lead):
        add_text(p, bold_lead, bold=True, size=size)
        add_text(p, text[len(bold_lead):], size=size)
    else:
        add_text(p, text, size=size)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph(text, f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(8 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(6 if level == 1 else 4)
    return p


def page(doc, title):
    doc.add_page_break()
    heading(doc, title)


def caption(doc, text):
    p = paragraph(doc, text, size=9.5, after=4, indent=False)
    p.paragraph_format.keep_with_next = True
    return p


def table(doc, headers, records, widths, *, size=10, left_cols=(0,)):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, width in zip(t.columns, widths):
        col.width = Cm(width)
    props = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        item = OxmlElement("w:" + edge)
        for name, value in (("val", "single"), ("sz", "6"), ("color", "D9D9D9")):
            item.set(qn("w:" + name), value)
        borders.append(item)
    props.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for side, value in (("top", 85), ("bottom", 85), ("left", 100), ("right", 100)):
        element = OxmlElement("w:" + side)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")
        margins.append(element)
    props.append(margins)
    for row_index, values in enumerate([headers] + list(records)):
        row = t.rows[0] if row_index == 0 else t.add_row()
        row_props = row._tr.get_or_add_trPr()
        no_split = OxmlElement("w:cantSplit")
        row_props.append(no_split)
        if row_index == 0:
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            row_props.append(repeat)
        for column_index, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Cm(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if column_index in left_cols
                           else WD_ALIGN_PARAGRAPH.CENTER)
            set_font(p.add_run(str(value)), size=size, bold=row_index == 0)
            if row_index == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "EDEDED")
                cell._tc.get_or_add_tcPr().append(shade)
    paragraph(doc, "", after=2, indent=False).paragraph_format.line_spacing = 0.25
    return t


def math_run(text):
    r = OxmlElement("m:r")
    prop = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), "Cambria Math")
    fonts.set(qn("w:hAnsi"), "Cambria Math")
    prop.append(fonts)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "22")
    prop.append(size)
    r.append(prop)
    t = OxmlElement("m:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def append_math(parent, value):
    if isinstance(value, (list, tuple)):
        for item in value:
            append_math(parent, item)
    else:
        parent.append(math_run(value) if isinstance(value, str) else value)


def sub(base, index):
    root = OxmlElement("m:sSub")
    for tag, value in (("e", base), ("sub", index)):
        element = OxmlElement("m:" + tag)
        append_math(element, value)
        root.append(element)
    return root


def sup(base, exponent):
    root = OxmlElement("m:sSup")
    for tag, value in (("e", base), ("sup", exponent)):
        element = OxmlElement("m:" + tag)
        append_math(element, value)
        root.append(element)
    return root


def sub_sup(base, index, exponent):
    root = OxmlElement("m:sSubSup")
    for tag, value in (("e", base), ("sub", index), ("sup", exponent)):
        element = OxmlElement("m:" + tag)
        append_math(element, value)
        root.append(element)
    return root


def summation(index, expression, upper=None):
    root = OxmlElement("m:nary")
    prop = OxmlElement("m:naryPr")
    operator = OxmlElement("m:chr")
    operator.set(qn("m:val"), "∑")
    prop.append(operator)
    location = OxmlElement("m:limLoc")
    location.set(qn("m:val"), "subSup")
    prop.append(location)
    if upper is None:
        hidden = OxmlElement("m:supHide")
        hidden.set(qn("m:val"), "1")
        prop.append(hidden)
    root.append(prop)
    for tag, value in (("sub", index), ("sup", upper or ""), ("e", expression)):
        element = OxmlElement("m:" + tag)
        append_math(element, value)
        root.append(element)
    return root


def fraction(numerator, denominator):
    root = OxmlElement("m:f")
    for tag, text in (("num", numerator), ("den", denominator)):
        element = OxmlElement("m:" + tag)
        append_math(element, text)
        root.append(element)
    return root


def equation(doc, *parts):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    root = OxmlElement("m:oMathPara")
    math = OxmlElement("m:oMath")
    for part in parts:
        append_math(math, part)
    root.append(math)
    p._p.append(root)
    return p


def accuracy(row):
    return f"{row['accuracy'] * 100:.2f} ± {row['ci95'] * 100:.2f}"


def paired(row):
    delta = row["paired_delta"] * 100
    if abs(delta) < 0.005:
        delta = 0.0
    return f"{delta:+.2f} ± {row['paired_ci95'] * 100:.2f}"


def interval(center, half):
    return f"[{(center - half) * 100:+.2f}, {(center + half) * 100:+.2f}]"


def paired_values(episodes, method, reference, shot):
    return [e["methods"][method]["accuracy"] - e["methods"][reference]["accuracy"]
            for e in episodes if e["shots"] == shot]


def interpretation(center, half):
    if center - half > 0:
        return "区间整体为正"
    if center + half < 0:
        return "区间整体为负"
    return "区间包含零"


def delta_sentence(episodes, method, reference, shot):
    mean, half = mean_ci(paired_values(episodes, method, reference, shot))
    return (f"{shot}-shot 下，{LABELS[method]}相对{LABELS[reference]}的平均配对差为 "
            f"{mean * 100:+.2f} 个百分点，95% 区间为 {interval(mean, half)}，"
            f"{interpretation(mean, half)}。")


def main_finding(rows):
    statements = []
    for shot in (1, 5):
        row = rows["full", shot]
        low = row["paired_delta"] - row["paired_ci95"]
        high = row["paired_delta"] + row["paired_ci95"]
        if low > 0:
            statements.append(f"{shot}-shot 的配对区间整体为正")
        elif high < 0:
            statements.append(f"{shot}-shot 的完整校准低于 Baseline")
        else:
            statements.append(f"{shot}-shot 的配对区间包含零，尚不能确认改善")
    return "；".join(statements) + "。"


def training_epoch_count(record):
    if "epochs" in record:
        return record["epochs"] if isinstance(record["epochs"], int) else len(record["epochs"])
    return len(record["history"])


def create_document(result, episodes, rows):
    p = result["protocol"]
    model = result["model"]
    runtime = result["runtime"]
    train = result["training"]
    warm_epochs = training_epoch_count(train["warm_ce"])
    branch_epochs = training_epoch_count(train["baseline_control"])
    maml = train["maml"]
    n = p["test_episodes_per_shot"]
    completed = datetime.fromisoformat(result["completed_utc"].replace("Z", "+00:00"))
    date = completed.astimezone().strftime("%Y 年 %m 月 %d 日")
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(1.8), Cm(1.8)
    section.left_margin, section.right_margin = Cm(2.2), Cm(2.2)
    section.header_distance, section.footer_distance = Cm(0.75), Cm(0.75)
    style_font(doc.styles["Normal"], 11)
    normal = doc.styles["Normal"].paragraph_format
    normal.line_spacing = 1.25
    normal.space_after = Pt(5)
    style_font(doc.styles["Title"], 23, bold=True, east_asia="黑体", latin="Arial")
    style_font(doc.styles["Subtitle"], 14, east_asia="黑体", latin="Arial")
    for level, size in ((1, 15), (2, 12)):
        style_font(doc.styles[f"Heading {level}"], size, bold=True, east_asia="黑体", latin="Arial")
    # Strip inherited title/heading borders if a future Word template adds any.
    for name in ("Title", "Subtitle", "Heading 1", "Heading 2"):
        for border in list(doc.styles[name].element.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(footer.add_run("第 "), size=9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    set_font(footer.add_run(" 页"), size=9)
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "人工智能课堂报告4 小样本图像分类"
    doc.core_properties.author = "第33组"
    doc.core_properties.keywords = "mini-ImageNet, ResNet12, MAML, 特征校准"

    # Page 1: identity, conclusion and the learning paradigms.
    title = doc.add_paragraph("面向小样本场景的深度神经网络\n特征校准方法研究", "Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(8)
    title.paragraph_format.space_after = Pt(12)
    subtitle = doc.add_paragraph("人工智能课堂报告4", "Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(12)
    for text in ("第33组  软件工程5班", "组长  梁镇贤", "组员  陈宗波  温文辉  陈宇冲  刘亦达", f"指导教师  王美华    {date}"):
        metadata = paragraph(doc, text, after=4, indent=False)
        metadata.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading(doc, "摘要", 2)
    paragraph(doc,
              f"针对新类别只有少量标注图像、支持集统计量不稳定的问题，本实验在 mini-ImageNet 上构建轻量 ResNet12，"
              f"比较监督 Baseline、完整二阶 MAML，以及均值方差校准与无标签查询原型更新。采用 {p['ways']}-way "
              f"1-shot 和 5-shot，每类 {p['queries_per_class']} 张查询图像，各评估 {n} 个共同 episode。训练使用全部 "
              f"{p['training_images']:,} 张基础类图像；验证与测试每类固定抽取 {p['evaluation_pool_per_class']} 张，"
              "参数选择在验证集完成后锁定。")
    paragraph(doc,
              f"完整校准在 1-shot 和 5-shot 的准确率分别为 {accuracy(rows['full', 1])}% 与 "
              f"{accuracy(rows['full', 5])}%（均值 ± 95% 置信区间半宽）。相对 Baseline 的配对差分别为 "
              f"{paired(rows['full', 1])} 与 {paired(rows['full', 5])} 个百分点。" + main_finding(rows) +
              "结论仅适用于本次窄网络、固定测试池和单次训练。")
    heading(doc, "研究问题与学习范式", 2)
    paragraph(doc,
              "医疗影像、工业缺陷等任务的标注成本较高，常面临新类样本不足。训练时的基础类与测试时的新类互不重叠。"
              "基础类的判别方向未必适合新类，少量支持样本又会使类中心和方差估计波动。1-shot 无法独立估计类内方差，"
              "图像增强也会改变 CNN 特征。因此分别控制训练表征、统计迁移和任务内适应，检验各模块对新类分类的实际影响。")
    table(doc, ["范式", "适应方式", "本实验对应"], [
        ["度量分类", "固定嵌入，支持集均值作原型", "监督 CE 预训练后的 Baseline"],
        ["优化式元学习", "学习初始化，再用支持集梯度适应", "完整二阶 MAML [1]"],
        ["统计迁移", "基础类统计收缩与查询原型更新", "轻量均值方差校准 [3]"],
    ], [2.8, 6.1, 7.7], size=9.5, left_cols=(0, 1, 2))

    # Page 2: the protocol and leakage controls.
    page(doc, "1 数据集与评估协议")
    paragraph(doc,
              "mini-ImageNet 共 100 类、每类 600 张 84×84 图像。实验采用 Ravi 与 Larochelle 的 64/16/20 类划分 [4]，"
              "从公开的 mini-ImageNet 工具项目取得数据，按文件清单核对类别互斥、图像数量和可解码性。训练先验只由训练类图像计算。")
    caption(doc, "表 1 数据使用范围")
    table(doc, ["子集", "类别数", "本次每类图像", "实际使用图像", "用途"], [
        ["训练", p["training_classes"], p["train_per_class"], f"{p['training_images']:,}", "CE、元训练、基础类先验"],
        ["验证", p["validation_classes"], p["evaluation_pool_per_class"], f"{p['validation_images']:,}", "训练诊断与校准配置选择"],
        ["测试", p["test_classes"], p["evaluation_pool_per_class"], f"{p['test_images']:,}", "配置锁定后的共同任务评估"],
    ], [1.5, 1.5, 3.1, 3.0, 7.5], left_cols=(0, 4))
    paragraph(doc,
              "测试 episode 先抽取 5 个新类，再从每类抽取互斥的支持集与查询集。方法之间共享同一 episode 的类、"
              "支持图像及查询图像，查询顺序打乱。每个 episode 内没有重复图像；不同 episode 可以复用测试池中的图像。")
    caption(doc, "表 2 每个测试任务的样本组成")
    table(doc, ["设定", "支持图像", "查询图像", "任务数", "随机猜测准确率"], [
        ["5-way 1-shot", "5 × 1 = 5", "5 × 15 = 75", n, "20%"],
        ["5-way 5-shot", "5 × 5 = 25", "5 × 15 = 75", n, "20%"],
    ], [3.6, 3.2, 3.3, 2.0, 4.5], left_cols=(0,))
    heading(doc, "1.1 验证集选择与公平控制", 2)
    paragraph(doc,
              "分别为 1-shot 与 5-shot 在验证集固定的 40 个 episode 上比较 18 组校准参数；选中后统一应用于测试和消融。"
              "候选只改变均值权重、原型更新权重与温度。测试图像在训练和校准参数选择完成后读取，测试查询标签仅用于计算最终准确率。")
    paragraph(doc,
              "CE 对照分支与分布正则分支从同一个预训练检查点出发，使用完全相同的样本顺序、两种增强、训练轮数和 CE 更新步数。"
              "唯一受控差异为分布正则权重。MAML 额外执行元训练，其训练开销和推理机制不同，报告不将其称为严格等训练预算对照。")
    heading(doc, "1.2 可复核记录", 2)
    paragraph(doc,
              f"主随机种子为 {p['seed']}。results.json 保存协议、训练摘要、权重摘要及方法汇总；episodes.json 保存每个任务的"
              "类别、支持与查询图像 ID、各方法准确率和耗时。生成报告时，逐行从 episode 原始值重算均值、置信区间和配对差，"
              "并检查支持与查询集合不相交。文件哈希用于固定训练权重、统计先验和演示数据池。")
    paragraph(doc,
              "本次测试池为每类 100 张，未覆盖每类全部 600 张。报告中的区间以当前测试池和同一训练检查点为条件，"
              "不代表换训练种子、换测试池或换硬件后的不确定性。", bold_lead="本次测试池", after=0)

    # Page 3: backbone, aligned CE and exact MAML.
    page(doc, "2 轻量网络与训练方法")
    heading(doc, "2.1 窄 ResNet12 表征", 2)
    paragraph(doc,
              f"网络包含 4 个残差块，每块 3 个 3×3 卷积，通道数依次为 {', '.join(map(str, model['widths']))}，"
              f"总计 12 个主路径卷积。块间使用 2×2 最大池化，末端全局均值池化得到 {model['embedding_dim']} 维特征；"
              "1×1 捷径投影不计入 ResNet12 的层数。GroupNorm(4) 不使用支持集与查询集混合的批归一化统计。")
    paragraph(doc,
              f"Baseline 含 64 类训练头，共 {model['parameters_baseline']:,} 个参数；MAML 含 5 类任务头，共 "
              f"{model['parameters_maml']:,} 个参数。此窄网络用于 CPU 课程实验，不能将其分数与常见 640 通道标准 ResNet12 的论文结果直接排名。")
    heading(doc, "2.2 交叉熵与分布正则", 2)
    paragraph(doc,
              f"先以 CE 训练 {warm_epochs} 轮得到公共初始化，再复制成两个分支，各训练 {branch_epochs} 轮。两分支均对同一图像生成"
              "两种随机增强并平均 CE。正则分支额外约束两批特征的一阶与二阶矩；统计沿批次计算，维度之间取平均。")
    equation(doc, "L = ", fraction([sub("CE", "1"), " + ", sub("CE", "2")], "2"),
             " + λR,    R = ", fraction([
                 sup(["||", sub("μ", "1"), " − ", sub("μ", "2"), "||"], "2"), " + ",
                 sup(["||", sub("v", "1"), " − ", sub("v", "2"), "||"], "2"),
             ], "d"))
    paragraph(doc,
              "其中 d=32；μ 与 v 分别为批次特征的均值及总体方差。对照分支 λ=0，正则分支 λ=0.5。"
              "这里的分布对齐发生在同一训练图像的两种增强视图之间，目标是减小表征随增强产生的偏移；"
              "没有利用测试类图像或标签做域对齐，也不等同于已经实证消除了全部跨类分布差异。")
    heading(doc, "2.3 完整二阶 MAML", 2)
    paragraph(doc,
              f"MAML 以 Baseline 编码器为起点，重新初始化 5 类线性头。元训练共 {maml['tasks']} 个任务，交替使用 1-shot 与 "
              f"5-shot，每类 {maml['queries_per_class']} 张训练查询图像，meta-batch 为 {maml['meta_batch_size']}。"
              f"内循环使用 {maml['inner_steps']} 步、学习率 {maml['inner_lr']}，外循环 Adam 学习率 {maml['outer_lr']}。")
    equation(doc, sub("θ′", "τ"), " = θ − α", sub("∇", "θ"), sub("L", sub("S", "τ")),
             "(θ),    ", sub("min", "θ"), " ",
             summation("τ", [sub("L", sub("Q", "τ")), "(", sub("θ′", "τ"), ")"]))
    paragraph(doc,
              "Sτ 与 Qτ 分别是元训练任务的支持集和查询集，查询标签用于外循环损失。内循环更新全部骨干与分类头参数，并保留梯度计算图；"
              "外循环通过支持集更新过程反传，因此包含二阶项。实现以 create_graph=True 和 functional_call 保持参数链路，"
              "同时用二阶与一阶梯度差异检查区分 MAML 与 FOMAML。")
    paragraph(doc,
              f"测试时仅使用支持集进行 {p['maml_test_inner_steps']} 步、学习率 0.1 的梯度适应；查询标签不参加更新。"
              "测试阶段无需构建二阶图。预测由适应后的分类头产生，不能用最近原型规则替代 MAML 的最终决策。", after=0)

    # Page 4: all calibrated inference operations and their statistical meaning.
    page(doc, "3 均值方差校准与原型自适应")
    paragraph(doc,
              "特征先做 L2 归一化。令 z 为 32 维特征，p_c 为新类 c 的支持集均值，n_c 为支持样本数。"
              "从训练类均值中取距离 p_c 最近的 K=3 类组成 N(c)，其均值与对角方差分别记作 μ_b、v_b。")
    equation(doc, sub("μ̄", "c"), " = ", fraction(summation("b∈N(c)", sub("μ", "b")), "K"),
             " ,    ", sub("ν", "c"), " = ", fraction(summation("b∈N(c)", [
                 "[", sub("v", "b"), " + ", sup(["(", sub("μ", "b"), " − ", sub("μ̄", "c"), ")"], "2"), "]",
             ]), "K"))
    paragraph(doc,
              "ν_c 同时计入近邻基础类内部方差和类中心之间的差异。近邻搜索只用支持统计与训练先验，"
              "不读取查询标签。近邻统计只提供收缩方向，不能视为新类的真实分布。")
    heading(doc, "3.1 均值与方差收缩", 2)
    equation(doc, sub("p̃", "c"), " = (1 − a)", sub("p", "c"), " + a", sub("μ̄", "c"))
    equation(doc, sub("ṽ", "c"), " = max(", fraction([
        "(", sub("n", "c"), " − 1)", sub_sup("s", "c", "2"), " + κ", sub("ν", "c"),
    ], [sub("n", "c"), " − 1 + κ"]), ", ε)")
    paragraph(doc,
              "a 为验证集选择的均值权重，s_c² 为支持集无偏方差，κ=4 为先验强度；1-shot 的观测自由度为零，"
              "方差完全来自训练先验。ε 为逐维下界，取训练类平均方差的 0.1 倍与 10⁻⁵ 的较大值。"
              "所有平方、除法和最大值运算均逐维进行。")
    equation(doc, "D(q,c) = ", fraction("1", "d"), " ",
             summation("j", fraction(sup(["(", sub("z", "qj"), " − ", sub("p̃", "cj"), ")"], "2"), sub("ṽ", "cj"))),
             " ,    P(c|q) = ", sub("softmax", "c"), "(−D(q,c)/T)")
    paragraph(doc,
              "分类采用对角 Mahalanobis 距离，没有加入 log-determinant 项，因此不声称是完整高斯密度分类器。"
              "关闭方差校准时各类、各维共用训练所得尺度，预测退化为欧氏最近原型。softmax 数值是相对模型分数，未验证为可靠概率。")
    heading(doc, "3.2 无标签查询原型更新", 2)
    equation(doc, sub_sup("p", "c", "(t+1)"), " = ", fraction([
        sub("n", "c"), " ", sub("p̃", "c"), " + β ",
        summation("q", [sub_sup("w", "qc", "(t)"), " ", sub("z", "q")]),
    ], [sub("n", "c"), " + β ", summation("q", sub_sup("w", "qc", "(t)"))]))
    paragraph(doc,
              "w_qc 为当前预测的 softmax 权重；只接受最大分数至少 0.25 的查询样本，未达到阈值者权重为零。"
              "β 由验证集选择，最多更新 2 步，每步仍以校准后的支持原型 p̃_c 为锚，方差保持不变。"
              "误分类查询可能把原型拉向错误方向，因而更新并不保证提高准确率。")
    paragraph(doc,
              "启用原型更新时，预测依赖整个无标签查询批次，属于转导学习；关闭更新时，各查询可独立分类，属于归纳学习 [5]。"
              "本方法借鉴分布校准的统计迁移思想 [3]，未复现其 Tukey 变换、完整协方差、高斯特征补样及线性分类器流程。"
              "本 Baseline 也未执行 ProtoNet 的 episodic 元训练 [2]。", after=0)

    # Page 5: headline comparisons, with actual conditional paired inference.
    page(doc, "4 Baseline 与 MAML 及完整校准比较")
    paragraph(doc,
              f"表 3 报告每种 shot 的 {n} 个共同测试任务。准确率单位为百分比；配对差为当前方法准确率减去同任务 Baseline 准确率，"
              "单位为百分点。所有 ± 数值均表示 95% 置信区间半宽，不能把准确率百分比与百分点差混为一谈。")
    caption(doc, "表 3 主要方法的准确率与配对差")
    records = []
    for method in ("baseline", "maml", "full"):
        records.append([LABELS[method], accuracy(rows[method, 1]), paired(rows[method, 1]),
                        accuracy(rows[method, 5]), paired(rows[method, 5])])
    table(doc, ["方法", "1-shot\n准确率 %", "1-shot\n配对差 pp", "5-shot\n准确率 %", "5-shot\n配对差 pp"],
          records, [2.6, 3.5, 3.5, 3.5, 3.5], size=10, left_cols=(0,))
    paragraph(doc,
              "Baseline 和 MAML 为归纳设定；完整校准含无标签查询联合更新，为转导设定。它们使用相同支持与查询图像，"
              "但测试时允许利用的信息结构不同。归纳条件下的校准结果见表 4 中的均值方差变体。", size=10)
    heading(doc, "4.1 统计量计算", 2)
    equation(doc, "Acc = ", fraction(summation("i", sub("a", "i")), "N"),
             " ,    ", sub("CI", "95"), " = Acc ± ", fraction("1.96 s(a)", "√N"))
    equation(doc, "Δ = ", fraction(summation("i", ["(", sub("a", "i"), " − ", sub("b", "i"), ")"]), "N"),
             " ,    ", sub("CI", "95"), "(Δ) = Δ ± ", fraction("1.96 s(a − b)", "√N"))
    paragraph(doc,
              "a_i 是某方法在第 i 个 episode 的查询准确率，b_i 为同任务 Baseline 准确率；s 为样本标准差，N 为任务数。"
              "使用 episode 作为统计单位可保留任务内相关性；配对差直接反映共同任务上的变化。两个单独准确率区间是否重叠，"
              "不能代替配对差的判断。这里使用正态近似，未进行多重比较校正。")
    heading(doc, "4.2 实测结果解释", 2)
    for shot in (1, 5):
        paragraph(doc, delta_sentence(episodes, "full", "baseline", shot))
    maml_comparisons = []
    for shot in (1, 5):
        mean, half = mean_ci(paired_values(episodes, "maml", "baseline", shot))
        signed_delta = f"{mean * 100:+.2f}".replace("-", "−")
        maml_comparisons.append(f"{shot}-shot 为 {signed_delta} pp，95% 区间 {interval(mean, half)}")
    paragraph(doc,
              "MAML 相对 Baseline 的配对差：" + "；".join(maml_comparisons) + "。"
              f"这反映当前初始化、{maml['tasks']} 个元训练任务与固定适应步数下的表现，"
              "不能外推为 MAML 这一方法在充分训练预算下的统一结论。")
    paragraph(doc,
              "完整校准的模块包括训练正则与推理时统计更新，表 3 不能单独区分每个模块的贡献。下一节保持验证集选定配置，"
              "通过开关模块比较变化，并同时说明各变体的归纳或转导性质。", after=0)

    # Page 6: actual ablations and the non-independent tuning caveat.
    page(doc, "5 固定配置消融实验")
    paragraph(doc,
              "除 Baseline 与“完整校准去正则”使用 CE 对照骨干外，其余校准变体使用带分布正则的骨干。"
              "所有变体共享完整方法在验证集选出的配置；本节比较模块开关的影响，没有为每个消融变体重新单独调参。")
    caption(doc, "表 4 消融准确率 均值 ± 95% 区间半宽 单位为百分比")
    table(doc, ["变体", "1-shot", "5-shot", "推理设定"], [
        [LABELS[method], accuracy(rows[method, 1]), accuracy(rows[method, 5]),
         "转导" if method in TRANSDUCTIVE else "归纳"]
        for method in METHODS if method != "maml"
    ], [5.1, 4.15, 4.15, 3.2], size=10, left_cols=(0,))
    comparisons = (
        ("alignment_only", "baseline", "分布正则对无校准骨干的影响"),
        ("full", "mean_variance", "均值方差基础上的原型更新增量"),
        ("full", "no_alignment_full", "完整校准条件下的分布正则增量"),
    )
    heading(doc, "5.1 关键模块的配对比较", 2)
    for method, reference, label in comparisons:
        pieces = []
        for shot in (1, 5):
            mean, half = mean_ci(paired_values(episodes, method, reference, shot))
            display_mean = mean * 100 if abs(mean * 100) >= 0.005 else 0.0
            pieces.append(f"{shot}-shot 为 {display_mean:+.2f} ± {half * 100:.2f} pp")
        paragraph(doc, label + "：" + "；".join(pieces) + "。")
    variance_deltas = [mean_ci(paired_values(episodes, "mean_variance", "mean_only", shot))[0]
                       for shot in (1, 5)]
    paragraph(doc,
              f"固定均值收缩后再加入方差校准，1-shot 与 5-shot 的配对变化分别为 "
              f"{variance_deltas[0] * 100:+.2f} 和 {variance_deltas[1] * 100:+.2f} pp。"
              "这直接反映当前方差先验的影响；其原因可能涉及基础类与新类统计不匹配、窄嵌入的表达能力，"
              "但本实验未通过额外对照区分这些解释。")
    heading(doc, "5.2 配置与解释范围", 2)
    conf1, conf5 = result["calibration"]["1"], result["calibration"]["5"]
    paragraph(doc,
              f"1-shot 最终配置为 a={conf1['mean_weight']}、β={conf1['prototype_weight']}、T={conf1['temperature']}；"
              f"5-shot 为 a={conf5['mean_weight']}、β={conf5['prototype_weight']}、T={conf5['temperature']}。"
              "两种设定均取 3 个基础类近邻、方差先验强度 4、最多 2 次原型更新和 0.25 的接受阈值。")
    paragraph(doc,
              "单独均值、单独方差与联合均值方差变体用于观察统计收缩的影响；“完整校准”相对“正则＋均值方差”"
              "增加了查询原型更新，同时从归纳设定转为转导设定。现有消融未遍历所有模块组合，"
              "也未测量各模块在不同训练种子下的交互效应。区间包含零时，只能说明当前证据不足以确认方向一致的变化。", after=0)

    # Page 7: deployment and the exact scope of the timing measurements.
    page(doc, "6 CPU 部署与实验平台集成")
    paragraph(doc,
              f"实验运行于本机 Windows CPU，PyTorch {runtime['torch']}，使用 {runtime['threads']} 个计算线程。"
              "模型在服务端本地加载，图像、支持集适应及查询推理均在本机完成。这里的端侧验证指通用 CPU 环境中的本地运行，"
              "未在手机、嵌入式芯片或专用 NPU 上测试，也未声称完成量化、剪枝或 ONNX 部署。")
    heading(doc, "6.1 推理流程与可复用产物", 2)
    paragraph(doc,
              "部署流程为：加载权重和训练类先验，抽取测试新类及支持与查询图像，完成图像编码，再按选定方法适应或校准，"
              "最后返回预测、准确率与可视化数据。训练权重、先验、校准配置及结果清单均保存为独立文件；"
              "服务器校验文件摘要并按版本失效缓存，避免同一路径更新后仍使用旧模型。")
    paragraph(doc,
              f"网页演示池来自测试类，每类保留 {result['demo_pool']['images_per_class']} 张，共 "
              f"{result['demo_pool']['images']} 张。单次演示每次真实编码图像，其结果属于一个 episode，不能替代上文 "
              f"{n} 个任务的基准汇总。统计校准仅修改原型或距离，图像嵌入保持不变；MAML 会通过支持集更新网络参数和图像特征。")
    heading(doc, "6.2 时间测量范围", 2)
    caption(doc, "表 5 本机每 episode 平均耗时 单位为毫秒")
    table(doc, ["方法", "1-shot", "5-shot", "计时内容"], [
        [LABELS[method], f"{rows[method, 1]['mean_episode_seconds'] * 1000:.2f}",
         f"{rows[method, 5]['mean_episode_seconds'] * 1000:.2f}",
         "图像编码及支持集梯度适应" if method == "maml" else "已缓存特征后的分类或校准"]
        for method in ("baseline", "maml", "full")
    ], [3.0, 2.6, 2.6, 8.4], size=10, left_cols=(0, 3))
    paragraph(doc,
              "基准为了复用相同表征，预先提取 Baseline 与校准骨干的图像特征，其行内计时不包含图像编码。"
              "MAML 必须对支持图像适应网络，因此计时包含图像编码和梯度适应。这些数字的范围不同，不能据此给出端到端速度倍率。"
              "网页单次耗时还包含请求处理与实际编码，不应与缓存特征计时混写。")
    heading(doc, "6.3 与实验一至三的关系", 2)
    paragraph(doc,
              "实验一至三围绕股权知识表示、推理及多跳路径搜索。实验四按教师题目研究小样本图像分类，"
              "接入同一个网站的实验入口、方法切换和结果展示流程，便于统一演示。两部分共享工程平台，"
              "数据与学习任务保持独立，图像分类准确率不能作为股权风险识别有效性的证据。")
    paragraph(doc,
              "二维散点图用于观察投影后的特征与原型位置。方差校准实际使用 32 维距离，MAML 使用适应后的分类头；"
              "因此图上二维最近原型不一定对应真实决策边界，图形只作为辅助说明。", after=0)

    # Page 8: limitations, conclusion and original literature.
    page(doc, "7 局限与结论")
    paragraph(doc,
              "本次结果受窄网络容量、单训练种子、有限元训练任务与固定测试池限制。95% 区间仅反映 episode 抽样波动，"
              "没有包含重新训练、数据池选择和参数搜索的不确定性。完整方法使用无标签查询批次，"
              "其转导结果需与归纳结果分开解释。校准先验可能存在负迁移，原型更新也可能放大错误高置信预测。")
    paragraph(doc,
              "实验完成了 CE 公平对照、完整二阶 MAML、均值与对角方差收缩、查询原型自适应、共同任务比较与固定配置消融，"
              "并在 CPU 平台形成可运行推理流程。" + main_finding(rows) +
              f"仅更新原型的 1-shot 准确率为 {accuracy(rows['prototype_only', 1])}%，"
              f"相对 Baseline 为 {paired(rows['prototype_only', 1])} pp；完整模块叠加没有延续这一收益。"
              "后续应增加训练种子和网络宽度，补充全部模块组合，分别测试端到端延迟、内存峰值与移动硬件表现。")
    heading(doc, "参考文献", 2)
    references = [
        ("[1] Finn C, Abbeel P, Levine S. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. "
         "ICML, PMLR 70:1126–1135, 2017.\nhttps://proceedings.mlr.press/v70/finn17a.html"),
        ("[2] Snell J, Swersky K, Zemel R. Prototypical Networks for Few-shot Learning. "
         "NeurIPS 30, 2017.\nhttps://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html"),
        ("[3] Yang S, Liu L, Xu M. Free Lunch for Few-shot Learning: Distribution Calibration. "
         "ICLR, 2021.\nhttps://openreview.net/forum?id=JWOiYxMG92s"),
        ("[4] Ravi S, Larochelle H. Optimization as a Model for Few-Shot Learning. ICLR, 2017.\n"
         "https://openreview.net/pdf?id=rJY0-Kcll"),
        ("[5] Liu Y, Lee J, Park M, et al. Learning to Propagate Labels: Transductive Propagation Network "
         "for Few-shot Learning. ICLR, 2019.\nhttps://arxiv.org/abs/1805.10002"),
    ]
    for reference in references:
        ref = paragraph(doc, reference, size=9.5, after=7, indent=False)
        ref.paragraph_format.keep_together = True
        ref.paragraph_format.line_spacing = 1.1
        ref.paragraph_format.left_indent = Cm(0.55)
        ref.paragraph_format.first_line_indent = Cm(-0.55)
    heading(doc, "数据与复现入口", 2)
    paragraph(doc,
              "mini-ImageNet 数据组织与划分：https://github.com/yaoyao-liu/mini-imagenet-tools\n"
              "课程项目：https://github.com/jjy123190/artificial-intelligence-main\n"
              "源码：源码/experiments/lab04/\n"
              "原始统计：源码/outputs/lab04/（results.json、episodes.json）。",
              size=9.5, after=0, indent=False)
    return doc


def export_pdf(docx_path, pdf_path):
    require(sys.platform == "win32", "--pdf requires Microsoft Word on Windows.")
    # Literal PowerShell strings and an encoded script avoid shell expansion.
    literal = lambda value: "'" + str(value).replace("'", "''") + "'"
    script = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "$reportWord = $null",
        "$reportDocument = $null",
        "try {",
        "  $reportWord = New-Object -ComObject Word.Application",
        "  $reportWord.Visible = $false",
        "  $reportWord.DisplayAlerts = 0",
        "  $reportWord.AutomationSecurity = 3",
        f"  $reportDocument = $reportWord.Documents.Open({literal(docx_path)}, $false, $false)",
        "  $reportDocument.Fields.Update() | Out-Null",
        "  $reportDocument.Repaginate()",
        f"  $reportDocument.ExportAsFixedFormat({literal(pdf_path)}, 17)",
        "  $reportDocument.Save()",
        "} finally {",
        "  if ($null -ne $reportDocument) { $reportDocument.Close(0) }",
        "  if ($null -ne $reportWord) { $reportWord.Quit() }",
        "  if ($null -ne $reportDocument) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportDocument) }",
        "  if ($null -ne $reportWord) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportWord) }",
        "}",
    ])
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace") or "Word PDF export failed.")
    require(pdf_path.is_file() and pdf_path.stat().st_size > 0, "Word did not create the PDF.")


def render_pdf(pdf_path, qa_directory):
    import fitz
    qa_directory.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as pdf:
        count = len(pdf)
        for index, sheet in enumerate(pdf):
            sheet.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(
                qa_directory / f"page-{index + 1:02d}.png")
    return count


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO / "源码" / "outputs" / "lab04")
    parser.add_argument("--output", type=Path, default=REPO / "deliverables" / "lab4" / "第33组_实验四_实验报告.docx")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--qa-dir", type=Path)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".docx", "Output must be a DOCX path.")
    require(not args.qa_dir or args.pdf, "--qa-dir requires --pdf.")
    result, episodes, rows = read_verified_results(args.artifacts.resolve())
    document = create_document(result, episodes, rows)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    summary = {"docx": str(output), "bytes": output.stat().st_size,
               "verified_rows": len(rows), "verified_episodes": len(episodes)}
    if args.pdf:
        pdf_path = output.with_suffix(".pdf")
        export_pdf(output, pdf_path)
        summary.update(pdf=str(pdf_path), pdf_bytes=pdf_path.stat().st_size)
        if args.qa_dir:
            summary["pages"] = render_pdf(pdf_path, args.qa_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
