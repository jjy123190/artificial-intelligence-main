"""Build an editable classroom deck from the CURRENT 30-page Lab 3 report.

All charts use real, frozen observations. The PPTX is staged in this work folder;
the exporter and publisher never rebuild the experimental results.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_DATA_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE as SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
REPORT = BASE / "source/第31组_实验三_实验报告.docx"
NAME = "第31组_实验三_汇报.pptx"
DEST = BASE / "draft" / NAME
EV = ROOT / "deliverables/lab3_备份/evidence"

COL = {
    "paper": "F7F4EE", "white": "FFFDFC", "ink": "22313A", "muted": "6D7679",
    "border": "DADBD5", "rust": "A44B29", "rust_soft": "F3E2D6",
    "teal": "237B70", "teal_soft": "E1EEEA", "blue": "536D8D", "blue_soft": "E5EAF1",
    "red": "B44C47", "red_soft": "F6E5E1", "gray": "859094", "gray_soft": "E9EBE7",
    "olive": "7C8B6D", "dark": "172D37", "light": "F9F6EF", "dark_muted": "B9C8C9",
    "dark_line": "39505A", "orange_light": "E58D5C",
}
FONT = "Microsoft YaHei"
NUMBER_FONT = "Segoe UI"
W, H = 16.0, 9.0
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
prs.core_properties.title = "第31组 · 实验三 · 大模型辅助的企业股权知识图谱多跳路径搜索与智能优化"
prs.core_properties.subject = "当前30页实验报告的新版课堂汇报；16页主讲、2页答辩备用"
prs.core_properties.author = "梁镇贤、陈宗波、温文辉、陈宇冲、刘亦达"
prs.core_properties.keywords = "BFS,A*,MCTS,GA,路径价值学习,专家规则,LLM,迁移设计"
prs.core_properties.comments = "Charts and diagrams are editable. Evidence is tied to the current report; no experiment was rerun to build this presentation."

doc = Document(REPORT)
tables = [[[c.text for c in row.cells] for row in table.rows] for table in doc.tables]
ga = json.loads((EV / "raw/ga_history.json").read_text(encoding="utf-8-sig"))
examples = json.loads((ROOT / "deliverables/lab3_备份/可选拓展_20261113/example_verification.json").read_text(encoding="utf-8-sig"))
case = lambda name: json.loads((EV / "cases" / (name + ".json")).read_text(encoding="utf-8-sig"))
S02, F01, F02, F03 = (case(n) for n in ["S02_three_hop", "F01_bfs_budget", "F02_minimum_budget", "F03_no_expert"])
assert len(tables) == 30
assert len(doc.inline_shapes) == 14
assert examples["positive_examples_accepted"] == 2 and examples["negative_cases_rejected"] == 8

slides_spec = []
source_metrics = {}


def rgb(color):
    return RGBColor.from_string(COL.get(color, color))


def named(shape, name):
    shape.name = name
    return shape


def no_shadow(shape):
    prop = shape._element.spPr
    for tag in ["a:effectLst", "a:effectDag"]:
        old = prop.find(tag, prop.nsmap)
        if old is not None:
            prop.remove(old)
    prop.append(OxmlElement("a:effectLst"))
    for effect in shape._element.xpath("./p:style/a:effectRef"):
        effect.set("idx", "0")
    return shape


def set_run_font(run, size, color="ink", bold=False, family=FONT):
    run.font.name = family
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    prop = run._r.get_or_add_rPr()
    prop.set("lang", "zh-CN")
    for tag in ["a:ea", "a:cs"]:
        node = prop.find(tag, prop.nsmap)
        if node is None:
            node = OxmlElement(tag)
            prop.append(node)
        node.set("typeface", FONT)


def txt(slide, text, x, y, w, h, size=21, color="ink", bold=False,
        align=PP_ALIGN.LEFT, family=FONT, name=None, line=1.2, valign=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        named(shape, name)
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = Pt(0)
    frame.margin_top = frame.margin_bottom = Pt(0)
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.vertical_anchor = valign
    for i, value in enumerate(text.split("\n")):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        p.line_spacing = line
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        run = p.add_run()
        run.text = value
        set_run_font(run, size, color, bold, family)
    return shape


def box(slide, x, y, w, h, fill="white", stroke=None, radius=False, name=None):
    shape = slide.shapes.add_shape(SHAPE.ROUNDED_RECTANGLE if radius else SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        named(shape, name)
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
    else:
        shape.fill.background()
    if stroke:
        shape.line.color.rgb = rgb(stroke)
        shape.line.width = Pt(0.7)
    else:
        shape.line.fill.background()
    if radius:
        shape.adjustments[0] = 0.09
    return no_shadow(shape)


def line(slide, x1, y1, x2, y2, color="border", width=1, dash=None, arrow=False):
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    if dash is not None:
        shape.line.dash_style = dash
    if arrow:
        tail = OxmlElement("a:tailEnd"); tail.set("type", "triangle"); tail.set("w", "sm"); tail.set("len", "sm")
        shape._element.spPr.get_or_add_ln().append(tail)
    return no_shadow(shape)


def arrow(slide, x, y, w=0.4, color="muted", h=0.2):
    return box_arrow(slide, x, y, w, h, color)


def box_arrow(slide, x, y, w, h, color):
    s = slide.shapes.add_shape(SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(color); s.line.fill.background()
    return no_shadow(s)


def circle(slide, x, y, d, fill="white", stroke=None):
    s = slide.shapes.add_shape(SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if stroke:
        s.line.color.rgb = rgb(stroke); s.line.width = Pt(1.4)
    else:
        s.line.fill.background()
    return no_shadow(s)


def chip(slide, text, x, y, w, fill="rust_soft", color="rust", size=16, h=0.4):
    box(slide, x, y, w, h, fill)
    return txt(slide, text, x + 0.06, y + 0.015, w - 0.12, h - 0.03,
               size, color, True, PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, line=1.0)


def picture_contain(slide, path, x, y, w, h, name):
    im = Image.open(path)
    iw, ih = im.size
    ratio = min(w / iw, h / ih)
    pw, ph = iw * ratio, ih * ratio
    s = slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                 width=Inches(pw), height=Inches(ph))
    named(s, name)
    s._element.nvPicPr.cNvPr.set("descr", "当前项目实验三入口，2026-11-13抓取。已核对服务workspace_id和当前网页构建文件。")
    return s


def base(title, subtitle, section, source, dark=False, appendix=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb("dark" if dark else "paper")
    idx = len(prs.slides)
    fg, muted = ("light", "dark_muted") if dark else ("ink", "muted")
    box(slide, 0.68, 0.49, 0.48, 0.055, "orange_light" if dark else "rust")
    txt(slide, section, 1.30, 0.33, 13.7, 0.34, 14, muted, True)
    txt(slide, title, 0.68, 0.87, 14.7, 0.69, 32, fg, True, name=f"S{idx:02}-headline", line=1.03)
    if subtitle:
        txt(slide, subtitle, 0.70, 1.60, 14.65, 0.47, 17, muted, name=f"S{idx:02}-subtitle", line=1.14)
    line(slide, 0.68, 8.28, 15.32, 8.28, "dark_line" if dark else "border", 0.7)
    txt(slide, source, 0.72, 8.44, 13.95, 0.29, 11, muted, name=f"S{idx:02}-source", line=1.0)
    txt(slide, f"{idx:02}", 14.85, 8.40, 0.44, 0.34, 14, muted, align=PP_ALIGN.RIGHT, family=NUMBER_FONT)
    slides_spec.append({"slide": idx, "title": title, "subtitle": subtitle, "section": section,
                        "source": source, "appendix": appendix, "dark": dark})
    return slide


def notes(slide, speech, page, details=""):
    index = len(prs.slides)
    text = speech + "\n\n【报告对应】当前30页《第31组_实验三_实验报告》，" + page + "。"
    if details:
        text += "\n【证据与口径】" + details
    slide.notes_slide.notes_text_frame.text = text
    slides_spec[index - 1]["speaker_notes"] = text
    slides_spec[index - 1]["report_pages"] = page


def style_chart(chart, minimum, maximum, major, number_format, horizontal=False, dark=False, legend=True):
    fg, grid = ("light", "dark_line") if dark else ("ink", "border")
    chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.name = FONT
        chart.legend.font.size = Pt(14)
        chart.legend.font.color.rgb = rgb(fg)
    va = chart.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = minimum, maximum, major
    va.tick_labels.font.name = FONT
    va.tick_labels.font.size = Pt(13)
    va.tick_labels.font.color.rgb = rgb("dark_muted" if dark else "muted")
    va.tick_labels.number_format = number_format
    va.tick_labels.number_format_is_linked = False
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(grid)
    va.major_gridlines.format.line.width = Pt(0.65)
    ca = chart.category_axis
    ca.tick_labels.font.name = FONT
    ca.tick_labels.font.size = Pt(16 if horizontal else 15)
    ca.tick_labels.font.color.rgb = rgb(fg)
    for ax in [va, ca]:
        ax.format.line.color.rgb = rgb(grid)
        ax.format.line.width = Pt(0.65)
    if horizontal:
        ca.reverse_order = True
    for series in chart.series:
        series.format.line.fill.background()
    # Transparent chart areas keep the slide's warm paper background.
    for tag in ["c:chartSpace", "c:plotArea"]:
        parent = chart._chartSpace if tag.endswith("chartSpace") else chart._chartSpace.chart.plotArea
        old = parent.find("c:spPr", parent.nsmap)
        if old is not None:
            parent.remove(old)
        prop = OxmlElement("c:spPr")
        prop.append(OxmlElement("a:noFill"))
        ln = OxmlElement("a:ln"); ln.append(OxmlElement("a:noFill")); prop.append(ln)
        parent.append(prop)


def percent_chart(slide, x, y, w, h, categories, series):
    data = CategoryChartData()
    data.categories = categories
    for label, values, _ in series:
        data.add_series(label, values)
    shape = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    named(shape, "editable-test-ood-success-chart")
    chart = shape.chart
    style_chart(chart, 0, 1.10, 0.2, "0%")
    plot = chart.plots[0]; plot.gap_width = 60
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.position = XL_DATA_LABEL_POSITION.INSIDE_END
    labels.number_format = "0.00%"; labels.number_format_is_linked = False
    labels.font.name = NUMBER_FONT; labels.font.size = Pt(13); labels.font.bold = True
    labels.font.color.rgb = rgb("light")
    # PowerPoint sometimes ignores defRPr on automatic chart labels. Explicit
    # end-paragraph properties make the intended light label color portable.
    label_xml = plot._element.find("c:dLbls", plot._element.nsmap)
    tx = label_xml.find("c:txPr", label_xml.nsmap)
    para = tx.find("a:p", tx.nsmap)
    end = OxmlElement("a:endParaRPr"); end.set("sz", "1200"); end.set("b", "1")
    fill = OxmlElement("a:solidFill"); color_el = OxmlElement("a:srgbClr"); color_el.set("val", COL["light"])
    fill.append(color_el); end.append(fill)
    latin = OxmlElement("a:latin"); latin.set("typeface", NUMBER_FONT); end.append(latin)
    para.append(end)
    for ser, (_, _, color) in zip(chart.series, series):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(color)
    return chart


def horizontal_chart(slide, x, y, w, h, values, maxval, series_name, colors):
    data = CategoryChartData(); data.categories = ["BFS", "固定A*", "GA A*", "MCTS"]
    data.add_series(series_name, values)
    shape = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    named(shape, "editable-" + series_name)
    chart = shape.chart
    style_chart(chart, 0, maxval, maxval / 5, "0", horizontal=True, legend=False)
    plot = chart.plots[0]; plot.gap_width = 80; plot.has_data_labels = True
    labels = plot.data_labels; labels.position = XL_DATA_LABEL_POSITION.OUTSIDE_END
    labels.number_format = "0.000"; labels.number_format_is_linked = False
    labels.font.name = NUMBER_FONT; labels.font.size = Pt(17); labels.font.bold = True
    labels.font.color.rgb = rgb("ink")
    for point, color in zip(chart.series[0].points, colors):
        point.format.fill.solid(); point.format.fill.fore_color.rgb = rgb(color)
        point.format.line.fill.background()
    return chart


def native_table(slide, headers, rows, widths, x, y, row_h, size=17, header_size=16,
                 colors=None, name="editable-table"):
    w = sum(widths); h = row_h * (len(rows) + 1)
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    named(shape, name)
    table = shape.table
    for c, value in zip(table.columns, widths):
        c.width = Inches(value)
    for ri, (row, values) in enumerate(zip(table.rows, [headers] + rows)):
        row.height = Inches(row_h)
        for ci, (cell, value) in enumerate(zip(row.cells, values)):
            cell.text = str(value)
            cell.margin_left = Inches(0.10); cell.margin_right = Inches(0.10)
            cell.margin_top = Inches(0.035); cell.margin_bottom = Inches(0.035)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill = "ink" if ri == 0 else ("white" if ri % 2 else "gray_soft")
            cell.fill.solid(); cell.fill.fore_color.rgb = rgb(fill)
            for p in cell.text_frame.paragraphs:
                p.space_before = p.space_after = Pt(0); p.line_spacing = 1.06
                for run in p.runs:
                    color = "light" if ri == 0 else (colors[ri-1] if colors and ci == 0 else "ink")
                    set_run_font(run, header_size if ri == 0 else size, color, ri == 0 or ci == 0)
            # A table with light horizontal rules and no dark default cell frames.
            prop = cell._tc.get_or_add_tcPr()
            for tag in ["a:lnL", "a:lnR", "a:lnT", "a:lnB"]:
                previous = prop.find(tag, prop.nsmap)
                if previous is not None: prop.remove(previous)
                ln = OxmlElement(tag); ln.set("w", "6350")
                solid = OxmlElement("a:solidFill"); srgb = OxmlElement("a:srgbClr"); srgb.set("val", COL["border"])
                solid.append(srgb); ln.append(solid); prop.append(ln)
    return table


def node(slide, label, x, y, w=1.25, h=0.8, fill="white", color="ink", stroke=None, size=25, sub=None):
    box(slide, x, y, w, h, fill, stroke, radius=True)
    txt(slide, label, x + 0.05, y + 0.10, w - 0.10, 0.42, size, color, True, PP_ALIGN.CENTER, line=1.0)
    if sub:
        txt(slide, sub, x - 0.30, y + h + 0.13, w + 0.60, 0.64, 15, color, align=PP_ALIGN.CENTER, line=1.15)


def build_cover():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = rgb("dark")
    slides_spec.append({"slide": 1, "title": "大模型辅助的企业股权知识图谱多跳路径搜索与智能优化",
                        "section": "封面", "source": "当前报告第1页与第16页三跳样例", "appendix": False, "dark": True})
    box(slide, 0.8, 0.68, 0.65, 0.07, "orange_light")
    txt(slide, "人工智能导论  /  实验三", 1.65, 0.49, 8.4, 0.44, 17, "dark_muted", True)
    txt(slide, "华南农业大学", 11.0, 0.49, 4.20, 0.44, 17, "dark_muted", align=PP_ALIGN.RIGHT)
    txt(slide, "大模型辅助", 0.80, 1.75, 7.5, 0.55, 24, "orange_light", True)
    txt(slide, "企业股权知识图谱\n多跳路径搜索与智能优化", 0.78, 2.53, 8.65, 1.92, 40, "light", True, line=1.2, name="S01-headline")
    txt(slide, "用真实关系生成路径，用可核验证据解释结果。", 0.82, 4.87, 7.65, 0.88, 22, "dark_muted", line=1.4)
    chip(slide, "BFS · A* · MCTS", 0.82, 6.04, 3.65, "dark_line", "light", 20, h=0.55)
    chip(slide, "GA · ML · LLM · 规则", 4.65, 6.04, 4.04, "dark_line", "light", 19, h=0.55)
    # Real S02 path, with two real trace branches; this is a structure visual.
    pts = [(9.40, 2.05), (11.35, 3.84), (13.62, 2.45), (14.13, 5.06)]
    centers = [(x + 0.47, y + 0.47) for x, y in pts]
    for a, b in zip(centers, centers[1:]): line(slide, *a, *b, "orange_light", 3)
    branch = [(12.06, 1.00), (10.22, 5.81)]
    line(slide, *centers[0], branch[0][0] + .38, branch[0][1] + .38, "dark_line", 1.4)
    line(slide, *centers[1], branch[1][0] + .38, branch[1][1] + .38, "dark_line", 1.4)
    for (x, y), label in zip(pts, ["N3", "N20", "N40", "N15"]):
        circle(slide, x, y, .94, "dark_line", "orange_light")
        txt(slide, label, x, y + .28, .94, .33, 21, "light", True, PP_ALIGN.CENTER, line=1.0)
    for (x, y), label in zip(branch, ["N23", "N41"]):
        circle(slide, x, y, .76, "dark_line")
        txt(slide, label, x, y + .23, .76, .29, 16, "dark_muted", align=PP_ALIGN.CENTER, line=1.0)
    txt(slide, "持股", 9.82, 3.34, 1.05, .37, 16, "dark_muted", align=PP_ALIGN.CENTER)
    txt(slide, "持股", 12.30, 3.49, 1.05, .37, 16, "dark_muted", align=PP_ALIGN.CENTER)
    txt(slide, "交易", 14.43, 4.03, .95, .37, 16, "dark_muted", align=PP_ALIGN.CENTER)
    txt(slide, "三跳样例 · N3 → N20 → N40 → N15", 9.1, 6.96, 6.1, .58, 17, "dark_muted", align=PP_ALIGN.CENTER)
    line(slide, .8, 7.84, 15.2, 7.84, "dark_line", .8)
    txt(slide, "软件工程5班 · 第31组", .82, 8.00, 6.6, .33, 17, "light", True)
    txt(slide, "梁镇贤（组长）  陈宗波  温文辉  陈宇冲  刘亦达", .82, 8.47, 12.5, .32, 15, "dark_muted", line=1.0)
    txt(slide, "2026.11.13", 13.25, 8.14, 1.95, .42, 17, "dark_muted", align=PP_ALIGN.RIGHT, family=NUMBER_FONT)
    notes(slide, "本实验将企业股权知识图谱的2—4跳问答转成状态空间搜索。汇报依次说明状态建模、三种搜索内核、GA与学习辅助、真实对比和失败案例，最后展示迁移到证明与规划的设计及两个核验样例。", "第1页、第16页", "封面的N3→N20→N40→N15是报告S02的真实三跳路径，实体ID省略test-7-0前缀。主讲第1—16页，第17—18页为答辩备用。")


def build_problem():
    slide = base("从企业关系图中找到一条可核验证据路径", "延续前两次实验的股权场景，问题给定关系顺序和终点属性。", "01  任务与建模", "来源：当前报告第3—4、16页；右图为本次抓取的当前网站实验三入口。")
    txt(slide, "三跳问题", .78, 2.27, 5.7, .42, 23, "rust", True)
    txt(slide, "从教学投资人4出发，\n依次经过持股 → 持股 → 交易，\n找到金融上市企业的有效路径。", .78, 2.93, 6.5, 1.63, 23, line=1.45)
    xs = [.92, 2.54, 4.16, 5.78]
    for a, b in zip(xs, xs[1:]): arrow(slide, a + .86, 5.05, .62, "rust", .16)
    for x, label in zip(xs, ["N3", "N20", "N40", "N15"]):
        circle(slide, x, 4.65, .86, "rust_soft" if label != "N15" else "teal_soft")
        txt(slide, label, x, 4.91, .86, .29, 19, "rust" if label != "N15" else "teal", True, PP_ALIGN.CENTER, line=1.0)
    txt(slide, "返回完整的实体序列、边ID与证据属性", .83, 6.06, 6.75, .7, 21, "ink", True)
    txt(slide, "持股 HOLDS  ·  担保 GUARANTEES  ·  交易 TRADES", .83, 7.01, 6.80, .72, 17, "muted", line=1.3)
    box(slide, 7.62, 2.28, 7.66, 5.62, "white", "border")
    picture_contain(slide, BASE / "assets/current_lab3_entry.png", 7.70, 2.36, 7.50, 5.28, "current-workspace-lab3-screenshot")
    txt(slide, "当前网站 · 搜索 / 优化 / LLM / 对比分析", 7.72, 7.69, 7.45, .31, 14, "muted", align=PP_ALIGN.CENTER, line=1.0)
    notes(slide, "场景沿用实验一、二的股权关系。主基准使用教学合成图，便于控制分支、干扰和证据错误。这里的真实三跳问题要求经过两条持股边和一条交易边，终点还必须满足金融行业和上市属性。右侧是当前项目网站的新截图，四个任务入口分别展示搜索、优化、LLM和实验对比。", "第3—4、16页", "问题test-7-0-Q7；报告样例S02。网页仅抓取入口，没有执行搜索或模型推断；workspace_id与当前根目录一致。")


def build_state():
    slide = base("一条完整路径就是一个搜索状态", "保留完整前缀，才能区分相同终点的不同证据路径。", "01  任务与建模", "来源：当前报告第4页表3；示例来自第16页S02。")
    box(slide, .74, 2.34, 14.52, .89, "white")
    txt(slide, "状态 s = (P, E)      累积代价 g(s) = |E|      目标深度 L ∈ {2, 3, 4}", .95, 2.58, 14.02, .48, 25, "ink", True, line=1.0)
    xx = [1.24, 4.74, 8.24, 11.74]
    for a, b in zip(xx, xx[1:]): arrow(slide, a + 1.33, 4.06, 1.74, "rust", .23)
    for x, label, sub in zip(xx, ["N3", "N20", "N40", "N15"], ["初态 · d=0", "追加R0036 · d=1", "追加R0010 · d=2", "追加R0057 · d=3"]):
        node(slide, label, x, 3.75, 1.35, .85, "teal_soft" if label == "N15" else "rust_soft", "teal" if label == "N15" else "rust", sub=sub)
    line(slide, .82, 5.50, 15.16, 5.50)
    txt(slide, "动作与转移", .84, 5.73, 6.85, .43, 23, "rust", True)
    txt(slide, "关系匹配 · 不回到已走实体 · 证据通过\n向P追加实体，向E追加边ID；单边代价为1。", .84, 6.30, 6.85, 1.01, 19, line=1.45)
    txt(slide, "目标与结果", 8.13, 5.73, 6.82, .43, 23, "teal", True)
    txt(slide, "完成关系序列，并匹配行业与上市属性。\n候选found之后，还要独立核查valid。", 8.13, 6.30, 6.88, 1.01, 19, line=1.45)
    txt(slide, "found：候选找到     exhausted：候选穷尽     budget_exhausted：预算用尽、仍有候选", .87, 7.65, 14.35, .42, 16, "muted", line=1.1)
    notes(slide, "状态由完整实体路径P和完整边序列E组成，当前节点是P的末项，g就是边数。不能只按当前实体去重，否则不同证据前缀会被错误合并。每个动作必须使用真实出边，并满足下一条关系、环约束和专家证据条件。found只表示程序找到候选，valid才表示问题和证据均通过；预算耗尽保留未知。", "第4页", "表3；当前代码采用固定2—4跳问题。箭头上的边ID属于S02，省略test-7-0前缀。")


def build_algorithms():
    slide = base("三套内核共享验证标准，搜索顺序各不相同", "BFS作为盲搜基线，A*按f排序，MCTS按选择、扩展、模拟、回传运行。", "02  搜索与混合架构", "来源：当前报告第4页、第17页；BFS/下界A*/纯MCTS不使用有估值A*的前瞻剪枝。")
    starts = [.76, 5.75, 10.74]
    for x in starts: box(slide, x, 2.35, 4.48, 5.04, "white")
    for x, head, sub, col in zip(starts, ["BFS", "A*", "MCTS"], ["逐层出队", "优先扩展较小f", "搜索与模拟交替"], ["gray", "rust", "teal"]):
        txt(slide, head, x + .28, 2.59, 3.92, .58, 30, col, True, family=NUMBER_FONT)
        txt(slide, sub, x + .28, 3.27, 3.94, .46, 22, "ink", True)
    for yy, dots in [(4.08, [2.86]), (4.70, [2.30, 3.42]), (5.32, [1.73, 2.49, 3.25, 4.01])]:
        for xx in dots: circle(slide, xx, yy, .27, "gray")
    for a, b in [(2.99, 2.43), (2.99, 3.55)]: line(slide, a, 4.35, b, 4.70, "border", 1)
    for a, b in [(2.43,1.86),(2.43,2.62),(3.55,3.38),(3.55,4.14)]:
        line(slide, a, 4.97, b, 5.32, "border", 1)
    line(slide, 1.18, 5.97, 4.78, 5.97)
    txt(slide, "不调用h；同层不区分潜力。\n预算计出队状态数。", 1.04, 6.20, 3.93, .9, 18, line=1.45)
    txt(slide, "f(s) = g(s) + h(s)", 6.01, 4.02, 3.94, .58, 25, "rust", True, family=NUMBER_FONT)
    for yy, label, fill in [(4.81, "较小f → 优先出队", "rust_soft"), (5.36, "同分再使用固定规则", "gray_soft")]:
        chip(slide, label, 6.02, yy, 3.92, fill, "ink", 17, h=.44)
    txt(slide, "固定、GA、ML及混合估值。\n学习估值的最优性需另论证。", 6.04, 6.20, 3.90, .9, 18, line=1.45)
    for x, y, label in [(11.07, 4.05, "选择"), (13.11, 4.05, "扩展"), (11.07, 5.12, "回传"), (13.11, 5.12, "模拟")]:
        chip(slide, label, x, y, 1.56, "teal_soft", "teal", 19, h=.57)
    arrow(slide, 12.70, 4.23, .28, "teal", .15)
    for kind, x, y, w, h in [(SHAPE.DOWN_ARROW,13.80,4.72,.15,.29),
                               (SHAPE.LEFT_ARROW,12.70,5.34,.28,.15),
                               (SHAPE.UP_ARROW,11.76,4.72,.15,.29)]:
        loop = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        loop.fill.solid(); loop.fill.fore_color.rgb=rgb("teal"); loop.line.fill.background(); no_shadow(loop)
    txt(slide, "树扩展与rollout分别计数。\n少树节点不等于少总工作。", 11.02, 6.20, 3.94, .9, 18, line=1.45)
    txt(slide, "所有方法都要核查最终关系、终点条件与每条边证据。", .86, 7.76, 14.18, .43, 21, "ink", True)
    notes(slide, "BFS按层处理所有前缀，是盲搜对照。A星使用f=g+h，下界版本和学习估值版本分别保留，学习估值不保证可采纳。MCTS采用UCT选择、树扩展、模拟和回传；模拟的每步工作必须额外记录。三套算法共享问题和最终验证标准，但纯BFS和纯MCTS没有使用估值A星的有界前瞻或证明剪枝。", "第4、17页", "MCTS机制示意；实际四跳案例为2次树扩展、7次模拟步，共9单位扩展+模拟工作。")


def build_architecture():
    slide = base("GA与模型优化排序，规则和程序核验完整证据", "学习建议搜索顺序，证据条件决定候选能否作为有效答案。", "02  搜索与混合架构", "来源：当前报告第5—8页；图中箭头为本项目模块职责关系。")
    for x, label, fill, color in [(5.07, "GA权重", "rust_soft", "rust"), (7.35, "路径价值分类器", "blue_soft", "blue"), (10.03, "LLM节点评分", "teal_soft", "teal")]:
        chip(slide, label, x, 2.42, 2.10 if x == 5.07 else 2.40, fill, color, 17, h=.54)
    for x in [6.10, 8.55, 11.20]: line(slide, x, 3.00, 8.40, 3.68, "border", 1.2)
    nodes = [(0.82, 2.20, "真实出边", "关系/环约束", "white", "ink"),
             (3.42, 2.10, "专家规则", "证据合法性", "teal_soft", "teal"),
             (5.94, 3.13, "启发排序", "固定h / GA / ML / LLM", "rust_soft", "rust"),
             (9.55, 2.20, "出队扩展", "记录预算与轨迹", "white", "ink"),
             (12.19, 2.93, "目标核验", "返回路径与证据", "teal_soft", "teal")]
    for x, w, title, desc, fill, color in nodes:
        box(slide, x, 3.78, w, 1.16, fill)
        txt(slide, title, x + .10, 4.00, w - .20, .42, 23, color, True, PP_ALIGN.CENTER)
        txt(slide, desc, x - .15, 5.10, w + .30, .62, 16, "muted", align=PP_ALIGN.CENTER, line=1.2)
    for x in [3.09, 5.63, 9.16, 11.82]: arrow(slide, x, 4.24, .24, "muted", .15)
    line(slide, .84, 6.02, 15.17, 6.02)
    txt(slide, "专家约束", .86, 6.27, 3.22, .4, 22, "teal", True)
    txt(slide, "verified ∧ active ∧ confidence ≥ 0.55", 4.17, 6.30, 10.96, .47, 23, "ink", True, family=NUMBER_FONT)
    txt(slide, "路径学习", .86, 6.97, 3.22, .4, 22, "blue", True)
    txt(slide, "6维逻辑回归；训练图穷举“是否有有效延伸”的正负标签。", 4.17, 7.00, 10.93, .47, 20, "ink")
    txt(slide, "低模型分数只改变优先级；已完整核验的无解后缀可剪枝，截断未知仍保留。", .87, 7.72, 14.38, .48, 17, "muted")
    notes(slide, "真实关系候选由图谱和程序生成。专家规则要求边已核验、有效，且置信度至少0.55。路径分类器使用6维特征，由训练图的成功或失败延伸标签学习；GA优化启发权重。LLM只判断真实前缀合法性、评估reward，并在搜索后解释已核验结果。低分类概率或LLM低分不能单独硬剪枝，完整证明无解的后缀与证据拒绝分别记录。", "第5—8页", "固定权重[1.7,0.5,0.6,0.2,1.5,0]；GA优化前五项，混合分类权重0.25。LLM未评分前缀取中性值，调整最多±1且不能覆盖已知合法目标后缀。")


def build_ga():
    slide = base("遗传算法在验证集优化五个启发参数", "三种子的真实曲线只出现少量改善，固定手工基线已在验证集全部答对。", "03  GA优化与实验协议", "来源：当前报告第5—6页表5、图1；ga_history中0—10代实测值，纵轴局部放大。")
    txt(slide, "14个体 · 10代 · 3种子", .86, 2.37, 6.03, .56, 25, "rust", True)
    txt(slide, "hθ = (L − d) + Σ θᵢzᵢ", .86, 3.34, 6.18, .53, 26, "ink", True, family=NUMBER_FONT)
    txt(slide, "J = 有效答案率\n     − 0.12 × 平均(扩展/8)", .86, 4.14, 6.18, 1.21, 25, "ink", True, line=1.35)
    txt(slide, "每个种子27道验证题；\n精英保留、锦标赛、交叉与变异。\n热启动候选在初代明确保留。", .86, 5.55, 6.10, 1.44, 19, "muted", line=1.45)
    data = CategoryChartData(); data.categories = list(range(11))
    for seed in ["7", "19", "43"]:
        data.add_series("种子" + seed, [x["fitness"] for x in ga[seed]["history"]])
    shape = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(7.08), Inches(2.91), Inches(8.16), Inches(3.79), data)
    named(shape, "editable-real-ga-fitness")
    chart = shape.chart; style_chart(chart, .9360, .9400, .001, "0.0000", legend=False)
    for x, seed, color in [(8.15,"7","gray"),(10.47,"19","rust"),(12.91,"43","teal")]:
        line(slide,x,2.58,x+.38,2.58,color,2)
        txt(slide,"种子"+seed,x+.50,2.41,1.47,.38,16,"muted")
    for ser, color in zip(chart.series, ["gray", "rust", "teal"]):
        ser.format.line.color.rgb = rgb(color); ser.format.line.width = Pt(2.0)
        ser.marker.style = XL_MARKER_STYLE.CIRCLE; ser.marker.size = 5
        ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = rgb(color)
        ser.marker.format.line.fill.background(); ser.smooth = False
    txt(slide, "代数（0为初代）", 9.40, 6.80, 3.55, .33, 14, "muted", align=PP_ALIGN.CENTER)
    box(slide, .83, 7.12, 14.38, .91, "rust_soft")
    txt(slide, "81道验证题", 1.03, 7.37, 3.11, .42, 21, "rust", True)
    txt(slide, "平均扩展 4.1605 → 4.1111", 4.21, 7.33, 6.26, .60, 27, "rust", True, family=NUMBER_FONT)
    txt(slide, "总计少4次 · 约1.19%", 10.75, 7.41, 4.13, .42, 19, "rust", True)
    source_metrics["ga_history"] = {s: [x["fitness"] for x in ga[s]["history"]] for s in ["7", "19", "43"]}
    notes(slide, "启发函数由剩余跳数和六项罚项构成，GA只优化前五个参数，路径分类项另设。适应度结合验证集有效答案率与扩展成本，每个种子27题，14个体进化10代。图中种子7完全不变，另两个种子在第2代只有小幅变化。合并81道验证题仅减少4次总扩展，约1.19%，因此不能把验证适应度提升写成稳定提高测试答案率。", "第5—6页", "图表直接使用ga_history的33个实测点，无平滑、无新训练；种子7/19/43最终J分别0.936667/0.939444/0.938889。纵轴范围0.936—0.940为局部放大。")


def build_protocol():
    slide = base("固定任务、预算与随机种子，比较同一批问题", "图间实体隔离；原test/OOD已被迭代查看，当前主表按同题回归解释。", "03  GA优化与实验协议", "来源：当前报告第3页表2、第9页实验协议；LLM为独立9题/新增18题配对。")
    txt(slide, "45张图 · 405道问题", .88, 2.35, 13.9, .72, 34, "ink", True)
    txt(slide, "教学合成图，每图9题；2、3、4跳各3题。常规48实体，OOD为64实体。", .89, 3.18, 14.26, .51, 20, "muted")
    x = .87
    for count, label, qs, fill, fg in [(18, "训练", 162, "blue_soft", "blue"), (9, "验证", 81, "rust_soft", "rust"),
                                      (12, "测试", 108, "teal_soft", "teal"), (6, "OOD", 54, "gray_soft", "muted")]:
        width = 14.26 * count / 45
        box(slide, x, 4.08, width - .05, 1.08, fill)
        txt(slide, f"{label} {count}图", x + .15, 4.26, width - .32, .38, 21, fg, True)
        txt(slide, f"{qs}题", x + .15, 4.73, width - .32, .39, 17, fg)
        x += width
    txt(slide, "162道test/OOD题 × 3档预算 × 8种方法", .91, 5.70, 10.2, .54, 27, "ink", True)
    txt(slide, "3,888次", 11.54, 5.60, 3.63, .76, 40, "rust", True, family=NUMBER_FONT)
    txt(slide, "预算 4 / 8 / 16  ·  种子 7 / 19 / 43  ·  各组保留成功与失败", .92, 6.56, 14.3, .48, 20, "muted")
    for x, title, sub in [(0.90, "有效答案率", "分母为全部固定任务"), (4.58, "平均扩展", "包含失败题"),
                           (8.25, "推理步数", "只统计有效成功路径"), (11.93, "冗余路径", "完整前缀重复到达比例")]:
        txt(slide, title, x, 7.32, 3.34, .39, 21, "ink", True)
        txt(slide, sub, x, 7.89, 3.32, .28, 14, "muted")
    notes(slide, "总共45图405题，训练18图、验证9图、测试12图、OOD6图，图之间不共享实体。主比較仅对108道测试题和54道OOD题，使用4、8、16三档预算和8种方法，得到3,888条运行。LLM推断单独使用9题和新增18题，不混入主表。四项指标分别保留有效答案率、所有题均扩展、有效成功路径边数和冗余路径比例；失败记录不能删除。", "第3、9页", "每种子构造固定图；各图9题，每种跳数3题。主test/OOD为已见回归，冻结后新图确认另列。")


def build_results():
    slide = base("排序优势明显，GA收益仍依赖测试分布", "预算8；test的108题与OOD的54题分别统计有效答案率。", "04  对比结果与失败机制", "来源：当前报告第10页表9、表10；两组有效率分母不同，图中均从0起画。")
    chosen = ["BFS", "A* 固定", "A* GA", "MCTS", "混合"]
    indices = {row[0]: row for row in tables[8][1:]}
    ood = {row[0]: row for row in tables[9][1:]}
    fraction = lambda value: float(value.split("(")[1].rstrip(")%").strip()) / 100
    test_values = [fraction(indices[key][1]) for key in chosen]
    ood_values = [fraction(ood[key][1]) for key in chosen]
    percent_chart(slide, .76, 2.44, 10.52, 5.45, ["BFS", "固定A*", "GA A*", "MCTS", "混合"],
                  [("测试集", test_values, "rust"), ("OOD", ood_values, "teal")])
    line(slide, 11.62, 2.52, 11.62, 7.98)
    txt(slide, "测试固定A* = GA", 12.02, 2.80, 3.18, .50, 18, "muted", True)
    txt(slide, "99.07%", 12.02, 3.40, 3.20, .72, 36, "rust", True, family=NUMBER_FONT)
    txt(slide, "均107/108题有效", 12.02, 4.12, 3.15, .45, 18, "muted")
    txt(slide, "OOD：固定 → GA", 12.02, 5.08, 3.20, .49, 18, "muted", True)
    txt(slide, "100.00%\n   → 98.15%", 12.01, 5.67, 3.30, 1.29, 29, "teal", True, family=NUMBER_FONT, line=1.28)
    txt(slide, "GA在OOD少答对1题", 12.02, 7.17, 3.17, .63, 18, "ink", True)
    source_metrics["success_chart"] = {"methods": chosen, "test": test_values, "ood": ood_values, "budget": 8, "source_tables": [9, 10]}
    notes(slide, "固定启发式测试有效答案率为99.07%，BFS为14.81%；主改善来自局部证据和排序。GA与固定版在测试上都为107/108成功，没有答案率增益。OOD上固定版54/54成功，GA变成53/54，表明验证最优不代表跨分布最优。MCTS测试为78.70%、OOD为75.93%。图中只画五个主要方法，全部八种方法的四项指标在答辩备用页。", "第10页", "读取当前报告表9/10；PPTX内嵌可编辑Excel图表。测试有效率14.81/99.07/99.07/78.70/99.07%；OOD9.26/100/98.15/75.93/98.15%。")


def build_cost():
    slide = base("少扩展节点，还要看扫描和模拟开销", "平均值包含成功与失败；扩展+模拟仍不是完整计算成本。", "04  对比结果与失败机制", "来源：当前报告第10页表9、第12页表11；预算8、test 108题。")
    method_rows = {r[0]: r for r in tables[8][1:]}
    cost_rows = {r[0]: r for r in tables[10][1:]}
    chosen = ["BFS", "A* 固定", "A* GA", "MCTS"]
    expanded = [float(method_rows[m][2]) for m in chosen]
    scans = [float(cost_rows[m][3]) for m in chosen]
    txt(slide, "每题平均扩展", .88, 2.37, 6.84, .47, 23, "ink", True)
    txt(slide, "每题全部出边扫描", 8.02, 2.37, 7.00, .47, 23, "ink", True)
    horizontal_chart(slide, .88, 3.02, 6.73, 3.76, expanded, 10, "每题扩展", ["gray", "blue", "rust", "olive"])
    horizontal_chart(slide, 8.04, 3.02, 6.94, 3.76, scans, 60, "全部出边扫描", ["gray", "blue", "rust", "olive"])
    line(slide, 7.76, 2.48, 7.76, 6.86)
    box(slide, .91, 7.11, 14.22, .93, "teal_soft")
    txt(slide, "MCTS：", 1.08, 7.37, 1.65, .42, 22, "teal", True)
    txt(slide, "4.139树扩展 + 6.935模拟步 = 11.074", 2.76, 7.33, 10.9, .55, 27, "teal", True, family=NUMBER_FONT)
    source_metrics["cost_charts"] = {"expanded": expanded, "edge_scans": scans, "methods": chosen, "mcts_rollout_mean": 6.935}
    notes(slide, "固定A星和GA减少了扩展，但它们为评估后缀扫描了更多出边。BFS每题扩展7.861，扫描33.713；GA扩展4.102，扫描47.657。MCTS扩展看似只有4.139，却还有平均6.935步模拟，总扩展加模拟为11.074。因此不能把少树节点或少扩展写成更少总算力，更不能直接推导墙钟速度。LLM加载、训练与推断开销需要另报。", "第10、12页", "所有数值来自当前表9和表11。MCTS模拟均值按11.074−4.139得到三位小数；原CSV在QA中另核对均值。")


def build_trace():
    slide = base("同一三跳题：混合4次扩展，BFS在8次后仍未出队目标", "持股 → 持股 → 交易，目标为金融上市公司；同图、同题、同预算8。", "04  对比结果与失败机制", "来源：当前报告第16、18页，S02/F01轨迹；图为真实分支节选，ID省略test-7-0前缀。")
    xpos = [1.0, 3.71, 6.42, 9.13]
    for x, depth in zip(xpos, range(4)): txt(slide, f"d={depth}", x, 2.43, 1.25, .35, 17, "muted", align=PP_ALIGN.CENTER)
    for a, b in zip(xpos, xpos[1:]): arrow(slide, a + 1.22, 4.18, 1.26, "teal", .20)
    for x, name in zip(xpos, ["N3", "N20", "N40", "N15"]):
        node(slide, name, x, 3.92, 1.26, .81, "teal_soft", "teal")
    node(slide, "N23", 3.71, 2.77, 1.26, .73, "gray_soft", "muted", size=23, sub=None)
    txt(slide, "保留 · 未扩展", 5.15, 2.88, 2.30, .40, 16, "muted")
    line(slide, 2.26, 4.00, 3.71, 3.18, "border", 1.3)
    node(slide, "N7", 1.03, 6.14, 1.21, .73, "red_soft", "red", size=23)
    line(slide, 1.64, 4.76, 1.64, 6.14, "red", 1.3)
    txt(slide, "专家拒绝\n未核验 / 低置信度", 2.51, 6.06, 3.24, .94, 17, "red", line=1.36)
    node(slide, "N41", 6.44, 6.14, 1.21, .73, "gray_soft", "muted", size=23)
    line(slide, 4.96, 4.60, 6.44, 6.34, "border", 1.3)
    txt(slide, "已证无解\n一步后继均不匹配目标", 7.80, 6.08, 2.81, .96, 16, "muted", line=1.4)
    for x, label in [(2.44, "R0036"), (5.15, "R0010"), (7.86, "R0057")]:
        txt(slide, label, x, 4.66, 1.26, .3, 14, "muted", align=PP_ALIGN.CENTER)
    line(slide, 11.01, 2.44, 11.01, 7.98)
    txt(slide, "混合：4次", 11.44, 2.47, 3.68, .49, 25, "teal", True)
    txt(slide, "N3 → N20 → N40 → N15\nvalid=True", 11.45, 3.14, 3.75, .94, 18, "ink", line=1.45)
    txt(slide, "BFS：8次", 11.44, 4.46, 3.68, .49, 25, "rust", True)
    txt(slide, "N3 → N20 → N23 → N22\n→ N11 → N35 → N41 → N40", 11.45, 5.13, 3.71, 1.08, 17, "ink", line=1.55)
    txt(slide, "N15已生成，但尚未出队。\n独立oracle仍有2条有效解。", 11.45, 6.66, 3.70, 1.07, 18, "muted", line=1.45)
    txt(slide, "找到有效目标后仍按正常出队计费。其他分支见报告完整搜索树。", .89, 7.67, 9.84, .46, 17, "ink", True)
    source_metrics["trace"] = {"hybrid_order": [e["node"] for e in S02["result"]["trace"] if e["status"]=="expanded"],
                               "bfs_order": [e["node"] for e in F01["result"]["trace"] if e["status"]=="expanded"],
                               "query_id": S02["query"]["id"], "hybrid_expanded": 4, "bfs_expanded": 8}
    notes(slide, "用同一道三跳题解释为什么预算内排序有用。混合方法沿N3、N20、N40、N15出队，4次完成有效路径。专家拒绝N7的不合格证据，完整核验的一步死路N41被剪枝，另有N23保留未扩展。BFS依次处理同层干扰，8次出队止于N40，目标N15虽已生成仍未出队。独立oracle有2条有效解，所以这次是预算失败，而不是图无解。", "第16、18页", "S02与F01共同问题test-7-0-Q7，预算8，图指纹和问题指纹相同。图是轨迹节选，不是完整搜索树；节点合并按完整边前缀识别。")


def build_failures():
    slide = base("失败定位到预算、排序和证据三个环节", "完整保留失败记录，才能解释有效率下降发生在哪里。", "04  对比结果与失败机制", "来源：当前报告第19—21页；F02、F03、F04—F06均为真实归档案例。")
    cards = [(.87, "预算不足", "rust", "rust_soft"), (5.84, "GA排序退化", "blue", "blue_soft"), (10.82, "专家规则关闭", "red", "red_soft")]
    for x, head, color, fill in cards:
        box(slide, x, 2.44, 4.31, 4.93, "white")
        chip(slide, head, x + .22, 2.69, 3.87, fill, color, 22, h=.6)
    txt(slide, "四跳题Q2", 1.13, 3.64, 3.8, .47, 23, "ink", True)
    txt(slide, "预算4：目标已生成\n还需要第5次出队。\n预算8：5次扩展成功。", 1.13, 4.45, 3.8, 1.66, 20, line=1.65)
    txt(slide, "增加预算后可恢复", 1.13, 6.58, 3.8, .47, 19, "rust", True)
    txt(slide, "OOD四跳Q8", 6.09, 3.64, 3.8, .47, 23, "ink", True)
    txt(slide, "固定A*：预算8成功。\nGA：预算8仍耗尽。\nGA：预算16，第9次成功。", 6.09, 4.45, 3.83, 1.66, 19, line=1.70)
    txt(slide, "验证最优受分布影响", 6.09, 6.58, 3.84, .47, 19, "blue", True)
    txt(slide, "N0 → N43 → N47", 11.06, 3.68, 3.83, .52, 22, "ink", True, family=NUMBER_FONT)
    txt(slide, "候选found=True，\n但valid=False。\nR0039未核验且已失效。", 11.06, 4.45, 3.80, 1.66, 20, line=1.65)
    txt(slide, "到达终点还要核查证据", 11.06, 6.58, 3.84, .47, 18, "red", True)
    txt(slide, "预算耗尽保留未知；未经核验的边不能作为有效答案。", .93, 7.74, 14.15, .47, 22, "ink", True)
    notes(slide, "三个案例把失败原因分开。四跳Q2在预算4时生成目标仍未完成第5次出队，提高预算8后成功。OOD Q8中，固定A星8次扩展成功，GA因不同优先顺序预算8耗尽，提高到16后第9次成功。关闭专家规则的Q0返回N0到N43再到N47，路径终点符合问题，但R0039未核验且失效，最终valid为False。", "第19—21页", "F02_minimum_budget与S03，F04_ood_fixed/F05_ood_ga/F06_ood_ga_recovery，F03_no_expert。它们不是本次重跑或人工伪造模型评分。")


def build_llm():
    slide = base("LLM完成三项辅助，实测收益需要逐项核对", "Qwen2.5-0.5B-Instruct；原始模型与有监督隐藏表示分类头分列。", "05  LLM实测与证据边界", "来源：当前报告第8、13、21页表13；分类头由训练图提示拟合，不能计为原始模型能力。")
    for x, head, desc in [(0.90,"合法性辅助","真实行动前缀 → True/False"), (5.89,"路径reward","给节点评价值，参与排序"), (10.88,"核验后解释","输入已获得并核验的路径")]:
        box(slide, x, 2.39, 4.24, 1.12, "teal_soft")
        txt(slide, head, x + .17, 2.55, 3.9, .43, 22, "teal", True)
        txt(slide, desc, x + .17, 3.05, 3.92, .32, 15, "teal")
    txt(slide, "新增18题配对", .93, 4.01, 5.5, .53, 25, "ink", True)
    txt(slide, "18 / 18", .93, 4.83, 5.35, .97, 47, "rust", True, family=NUMBER_FONT)
    txt(slide, "三组均成功，扩展均78次", .95, 5.97, 5.67, .68, 22, "ink", True)
    txt(slide, "无LLM / 原始LLM / 分类头", .95, 6.81, 5.65, .52, 18, "muted")
    txt(slide, "100条数值挑战", 7.02, 4.02, 8.06, .49, 25, "ink", True)
    rows = [["原始LLM", "12", "0", "88.00%", "93.10%"],
            ["分类头", "1", "2", "97.00%", "91.73%"]]
    native_table(slide, ["版本", "误放行", "误拒", "准确率", "均衡率"], rows,
                 [2.14, 1.42, 1.15, 1.80, 1.60], 7.02, 4.78, .75, 17, 16, name="editable-llm-confusion-table")
    txt(slide, "误放行减少，误拒增加；\n自由解释仍有真实语义失真案例。", 7.05, 7.20, 8.0, .96, 18, "muted", line=1.45)
    source_metrics["llm_table"] = rows
    notes(slide, "LLM承担行动合法性、reward评分和推理解释三项辅助。新增18题的无LLM、原始LLM和分类头三组都18题成功、总扩展78次，没有测得端到端搜索收益。100条数值挑战中分类头把误放行12降为1，却产生2个误拒，均衡准确率从93.10%降到91.73%，所以总准确率上升并不能覆盖类别不平衡。自由解释中保留了一例虚构机构的真实语义失真；未认证文本数量不等于幻觉数量。", "第8、13、21页", "模型固定revision 7ae557604adf67be50417f59c2c2f167def9a775。表13原始/分类头数值均照当前报告；没有新模型调用。")


def build_migration():
    slide = base("迁移先重定义状态、动作、终止与成本", "复用搜索策略、GA流程和日志评价；领域适配层负责新任务语义。", "06  可选拓展：证明与规划", "来源：当前报告第24页表23；本轮完成迁移方案和给定样例核验。")
    box(slide, .91, 2.40, 14.17, .70, "white")
    txt(slide, "初态  ·  合法动作  ·  状态转移  ·  目标测试  ·  步代价  ·  状态键  ·  证书核验", 1.12, 2.61, 13.75, .41, 20, "ink", True, PP_ALIGN.CENTER)
    txt(slide, "数学定理证明", .98, 3.61, 6.43, .56, 28, "rust", True)
    txt(slide, "规划任务", 8.32, 3.61, 6.43, .56, 28, "teal", True)
    line(slide, 7.72, 3.61, 7.72, 7.85)
    left = [("状态", "上下文Γ、全部未闭合目标、证明记录。"),
            ("动作", "选规则是OR；子目标是AND，\n必须全部完成。"),
            ("终止", "目标全部关闭，完整证书还须独立检查。"),
            ("成本/去重", "约定推理/工具成本；\n状态键保留全部目标与约束。")]
    right = [("状态", "世界事实、必要资源与计划前缀。"),
             ("动作", "核查前置条件，执行Add/Delete效应。"),
             ("终止", "完整动作重放合法，最终状态满足目标。"),
             ("成本/去重", "计划累加真实代价；\nbest_g去重，低成本到达可重开。")]
    for x, rows, color in [(.99, left, "rust"), (8.32, right, "teal")]:
        for j, (label, body) in enumerate(rows):
            yy = 4.49 + j * .86
            txt(slide, label, x, yy, 1.60, .4, 19, color, True)
            txt(slide, body, x + 1.70, yy, 4.70, .90, 18, "ink", line=1.3)
    notes(slide, "目前代码显式限制2—4跳，状态防环与关系特征都为企业图设计。迁移不能只换输入。适配接口需要初态、合法动作、转移、目标判定、真实代价、语义状态键和证书验证。数学证明要维护完整目标集合，规则产生两个子目标时两个都要完成；规划要检查动作前置条件与Add/Delete，路径长度不固定，以best_g保留低成本到达。", "第24页", "本页为设计方案。GA和价值分类器必须重建新领域特征与标签；预算截断unknown不自动成为负例；LLM仍只评价合法性/节点价值、解释已核验结果。")


def block_world(slide, x, y, final=False):
    line(slide, x - .05, y + 1.82, x + 2.18, y + 1.82, "muted", 1.1)
    if final:
        items = [(x + .73, y + 1.24, "C"), (x + .73, y + .67, "B"), (x + .73, y + .10, "A")]
    else:
        items = [(x + .07, y + 1.24, "A"), (x + .84, y + 1.24, "B"), (x + 1.61, y + 1.24, "C")]
    for xx, yy, label in items:
        color = {"A":"rust", "B":"blue", "C":"teal"}[label]
        fill = {"A":"rust_soft", "B":"blue_soft", "C":"teal_soft"}[label]
        box(slide, xx, yy, .58, .54, fill, color)
        txt(slide, label, xx, yy + .11, .58, .33, 21, color, True, PP_ALIGN.CENTER, family=NUMBER_FONT, line=1.0)


def build_examples():
    slide = base("两个推演样例可以逐步复核", "限定规则下检查证明证书；从初态重放计划并核查每步与最终目标。", "06  可选拓展：证明与规划", "来源：当前报告第25—26页表24—26；verify_examples重放记录。")
    txt(slide, "等式定理", .92, 2.38, 6.32, .5, 25, "rust", True)
    txt(slide, "a=b,  b=c  ⊢  c=a", .92, 3.00, 6.36, .64, 29, "ink", True, family=NUMBER_FONT)
    for yy, val, desc in [(3.91,"c=a","对称，转为a=c"), (4.71,"a=c","传递，中间项b"), (5.51,"a=b ∧ b=c","exact h1与h2，全部闭合")]:
        chip(slide, val, .99, yy, 2.43, "rust_soft", "rust", 22, h=.5)
        txt(slide, desc, 3.65, yy + .02, 3.76, .62, 18, "ink", line=1.2)
    box(slide, .97, 6.29, 6.39, .62, "white")
    txt(slide, "Eq.symm (Eq.trans h1 h2)", 1.16, 6.47, 6.02, .32, 19, "ink", family="Consolas", line=1.0)
    txt(slide, "4次逆向动作；2个等式推理节点。", 1.0, 7.05, 6.26, .42, 17, "muted")
    line(slide, 7.70, 2.49, 7.70, 7.48)
    txt(slide, "三积木规划", 8.17, 2.38, 6.75, .5, 25, "teal", True)
    block_world(slide, 8.26, 3.04, False)
    block_world(slide, 12.71, 3.04, True)
    arrow(slide, 10.86, 4.12, 1.19, "teal", .19)
    txt(slide, "初态：都在桌面", 8.18, 5.15, 2.70, .41, 16, "muted", align=PP_ALIGN.CENTER)
    txt(slide, "目标：A在B上，B在C上", 12.25, 5.15, 2.91, .67, 16, "muted", align=PP_ALIGN.CENTER)
    txt(slide, "PickUp(B) → Stack(B,C)\nPickUp(A) → Stack(A,B)", 8.19, 5.88, 6.79, 1.06, 22, "teal", True, family=NUMBER_FONT, line=1.42)
    txt(slide, "4步、代价4，末态手空；单位代价下界4。", 8.19, 7.05, 6.86, .42, 17, "muted")
    box(slide, .93, 7.58, 14.15, .58, "teal_soft")
    txt(slide, "2个正例通过 · 8个错误分支拒绝  |  本轮核验有限样例，未运行Lean内核", 1.13, 7.72, 13.77, .37, 18, "teal", True, PP_ALIGN.CENTER, line=1.0)
    source_metrics["migration"] = {"proof_actions": examples["theorem"]["backward_actions"], "plan_length": 4,
                                  "plan_cost": 4, "positive": 2, "negative": 8, "plan_actions": [x["action"] for x in examples["planning"]["trace"][1:]]}
    notes(slide, "等式例从c=a逆向先用对称转为a=c，再以b为中间项分成a=b和b=c两个AND目标，分别引用h1和h2。证书自底向上核验传递再对称。规划例初态ABC都在桌面，先抓B叠到C，再抓A叠到B。每步检查前置条件和增加/删除效应，最终手空且满足两项On目标。2次Stack和2次获取动作构成下界4，给定计划正好达到。两个正例通过，四个证明错误和四个规划错误被拒绝。", "第25—26页", "这里只核验给定证书与给定计划，没有新领域搜索基准、没有Lean内核调用、没有GA/LLM迁移性能数据。规划初态手空，目标还要求最终手空。")


def build_migration_eval():
    slide = base("用完成率、总成本和泛化指标检验迁移", "后续先接入可信验证器，再在独立任务族中比较搜索与学习增益。", "06  可选拓展：证明与规划", "来源：当前报告第27页表27；表中完整迁移评估属于待实施协议。")
    rows = [("完成正确性", "全部固定任务的证书/计划通过率；\n并列记录unknown、非法结果和工具错误。"),
            ("搜索与工具成本", "扩展、候选、模拟、模型调用/token；\n另记墙钟耗时和峰值内存。"),
            ("解质量", "证明节点、计划总代价；\n小任务用已知最优解检查代价差距。"),
            ("泛化与安全", "隔离定理结构/规划任务族；\n负例误接受、误拒绝与重复状态率。")]
    for i, (head, body) in enumerate(rows):
        yy = 2.60 + i * 1.28
        line(slide, .99, yy - .11, 9.46, yy - .11)
        txt(slide, head, 1.00, yy + .11, 2.29, .54, 21, "ink", True)
        txt(slide, body, 3.42, yy + .05, 5.94, .94, 19, "muted", line=1.4)
    box(slide, 10.01, 2.48, 5.08, 5.40, "teal_soft")
    txt(slide, "本轮已核验", 10.29, 2.82, 4.52, .45, 23, "teal", True)
    txt(slide, "2正例 + 8反例", 10.29, 3.61, 4.52, .63, 28, "teal", True)
    txt(slide, "有限等式规则、三积木；\n完整中间状态与拒绝原因留档。", 10.30, 4.52, 4.51, 1.06, 20, "teal", line=1.4)
    line(slide, 10.31, 5.97, 14.78, 5.97, "teal", .7)
    txt(slide, "下一步", 10.29, 6.23, 4.52, .4, 22, "teal", True)
    txt(slide, "可信内核 + 隔离任务集 + 消融\n扩大对象规模和目标组合。", 10.30, 6.82, 4.50, .87, 18, "teal", line=1.4)
    notes(slide, "跨领域完整评估应先固定任务和预算，独立验证所有输出，记录完成率和unknown，不能剔除预算超限。扩展、模拟与模型token要共同统计；变长规划检查真实代价，证明检查证书规模。数据划分按定理结构或规划任务族，另测对象规模变化。当前核验范围仅是有限等式和三积木，不能估计通用泛化或把样例通过写成完整系统准确率。", "第27页", "两例Python核验不等价于通用形式证明内核认证。学习估值可用于排序，但不可采纳时不保证最优性。")


def build_close():
    slide = base("让每次搜索返回可核验的证据", "有限预算下，搜索排序、规则约束和完整核验共同决定结果。", "07  汇报结论", "依据：当前报告第10、12—13、23—27页；主实验、LLM配对和迁移样例分别统计。", dark=True)
    for x, label, number, body, color in [(.98,"启发式搜索","99.07%","测试固定A*有效率；\n同预算BFS为14.81%。","orange_light"),
                                         (5.96,"GA优化","分布依赖","测试与固定版同率；\nOOD少答对1题。","light"),
                                         (10.94,"LLM与规则","证据优先","新增18题未测得搜索增益；\n规则核验保留失效边界。","dark_muted")]:
        txt(slide, label, x, 2.72, 4.01, .49, 23, "dark_muted", True)
        txt(slide, number, x, 3.72, 4.05, .80, 37, color, True, family=NUMBER_FONT)
        txt(slide, body, x, 4.92, 4.05, 1.20, 21, "light", line=1.45)
    line(slide, 5.52, 2.74, 5.52, 6.34, "dark_line", 1)
    line(slide, 10.50, 2.74, 10.50, 6.34, "dark_line", 1)
    txt(slide, "演示：实验三 → 多跳搜索工作台 → 扩展树 → 证据核验", 1.02, 7.18, 13.97, .65, 23, "light", True)
    notes(slide, "本实验交付的是可核验的混合搜索课程系统。固定启发式在有限预算下明显优于盲搜，但额外扫描必须计费。GA在验证集只有小幅扩展改善，并在OOD失去一题，结论具有分布依赖性。LLM完成三项辅助，新增配对没有测得搜索收益；专家规则和程序证据核验保护输出正确性。两个迁移样例说明接口和验证设计可具体化，后续仍要在独立任务集检验泛化。", "第10、12—13、23—27页", "主讲到此结束。若需要解释四项指标、消融或成员职责，使用后两页答辩备用。")


def build_appendix_metrics():
    slide = base("答辩备用：四项指标与消融明细", "预算8，test 108题；OOD有效率单列。推理步数仅对有效成功路径求均值。", "附录  A  指标与解释", "来源：当前报告第10页表9/10；无效答案率和有效答案率不是互补，因为预算失败另占部分。", appendix=True)
    ood = {r[0]: r for r in tables[9][1:]}
    names = {"A* 下界":"A*下界", "A* 固定":"固定A*", "A* GA":"GA A*", "A* ML":"ML A*"}
    rows = [[names.get(r[0],r[0]), r[1].split("(")[1].rstrip(")"), r[2], r[3], r[4], r[5], ood[r[0]][1].split("(")[1].rstrip(")")] for r in tables[8][1:]]
    native_table(slide, ["方法", "有效率", "均扩展", "有效步数", "冗余占比", "无效答案率", "OOD有效率"], rows,
                 [2.13, 2.06, 1.83, 1.94, 1.95, 2.25, 2.02], .91, 2.55, .535, 17, 16,
                 colors=["gray","gray","blue","rust","blue","olive","teal","red"], name="editable-four-metrics-ablation-table")
    txt(slide, "分类器低概率不直接删分支；无专家组的部分候选found=True但证据valid=False。", 1.00, 7.68, 14.03, .48, 18, "muted")
    source_metrics["full_metrics_table"] = rows
    notes(slide, "备查八种方法的全部关键指标。下界A星在固定跳数中没有排序区分度，与BFS同为14.81%。关闭专家后测试有效率85.19%，无效答案率14.81%；OOD有效率81.48%，无效答案率16.67%。MCTS树扩展不包含模拟。冗余路径按完整前缀重复到达已有端点记录，再按题平均，不能当作全图潜在冗余率。", "第10页", "当前表9/10原数值完整保留。方法ID与输出：bfs, astar_admissible, astar_fixed, astar_ga, astar_ml, mcts, hybrid, hybrid_no_expert。")


def build_appendix_team():
    slide = base("答辩备用：成员分工与证据定位", "按主要模块拟定职责，各模块使用统一数据、接口与评价口径。", "附录  B  协作与来源", "来源：当前报告第28—30页；分工沿用附录B的拟定安排，具体交接由小组确认。", appendix=True)
    member_rows = [["梁镇贤（组长）", "总体设计、A*、系统集成", "接口、预算、联调与验收"],
                   ["陈宗波", "启发函数、GA、路径学习", "特征、适应度、数据隔离"],
                   ["温文辉", "图谱数据、BFS/MCTS", "问题整理、对照与轨迹复核"],
                   ["陈宇冲", "LLM辅助、专家规则", "模型权限、证据与失败分析"],
                   ["刘亦达", "实验统计、图表、报告", "指标汇总、案例与一致性核对"]]
    native_table(slide, ["成员", "主要模块", "协作重点"], member_rows, [2.27, 3.55, 3.53], .93, 2.60, .81,
                 17, 17, name="editable-member-responsibilities")
    txt(slide, "报告查阅位置", 10.74, 2.64, 4.29, .49, 24, "rust", True)
    for y, title, page in [(3.45,"核心指标与LLM实测","第10—13页"), (4.41,"搜索树与失败轨迹","第15—21页"),
                            (5.37,"迁移方案及两个样例","第24—27页"), (6.33,"参考文献与复核说明","第28—29页")]:
        txt(slide, title, 10.77, y, 4.22, .42, 18, "ink", True)
        txt(slide, page, 10.77, y + .47, 4.20, .35, 16, "muted")
    notes(slide, "成员职责沿用当前报告附录B，不注明工作量或贡献比例。梁镇贤负责总体、A星和集成；陈宗波负责GA和路径学习；温文辉负责数据与BFS/MCTS；陈宇冲负责LLM与专家规则；刘亦达负责实验统计、可视化和报告。各负责人共同交叉复核数据、模型权限和结论。\n\n参考来源：Hart等，A*经典论文，1968，DOI 10.1109/TSSC.1968.300136；Kocsis与Szepesvari，UCT，ECML 2006；Qwen官方固定模型卡；Lean社区Quantifiers and Equality官方文档；Fikes与Nilsson，STRIPS，1971，DOI 10.1016/0004-3702(71)90010-5。外部文献只解释方法，不提供本项目实验数值。", "第28—30页", "所有PPT数字、轨迹和核验范围均取当前30页报告；主讲16页，2页答辩备用。")


def main():
    for fn in [build_cover, build_problem, build_state, build_algorithms, build_architecture, build_ga,
               build_protocol, build_results, build_cost, build_trace, build_failures, build_llm,
               build_migration, build_examples, build_migration_eval, build_close,
               build_appendix_metrics, build_appendix_team]:
        fn()
    # User saved a newer cover while the PPT was being prepared. Always use
    # the latest saved class line, preserving its deliberate group-label edit.
    for shape in prs.slides[0].shapes:
        if shape.has_text_frame and shape.text == "软件工程5班 · 第31组":
            shape.text_frame.paragraphs[0].runs[0].text = doc.paragraphs[7].text.strip()
    # Apply explicit Office labels after all chart construction; the chart's
    # embedded workbook and source values remain native and editable.
    success = next(sh.chart for sh in prs.slides[7].shapes if sh.has_chart)
    for series in success.series:
        series.has_data_labels = True
        for point in series.points:
            label = point.data_label
            label.position = XL_DATA_LABEL_POSITION.INSIDE_END
            label.font.name = NUMBER_FONT
            label.font.size = Pt(11)
            label.font.bold = True
            label.font.color.rgb = rgb("light")
        container = series._element.find("c:dLbls", series._element.nsmap)
        for label_node in [container] + list(container.findall("c:dLbl", container.nsmap)):
            existing = label_node.find("c:numFmt", label_node.nsmap)
            if existing is not None:
                label_node.remove(existing)
            fmt = OxmlElement("c:numFmt"); fmt.set("formatCode", "0.00%"); fmt.set("sourceLinked", "0")
            insert_before = label_node.find("c:txPr", label_node.nsmap)
            if insert_before is not None:
                label_node.insert(list(label_node).index(insert_before), fmt)
            else:
                label_node.append(fmt)
    assert len(prs.slides) == 18
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if DEST.exists():
        DEST.rename(DEST.with_name(DEST.stem+"-backup-"+datetime.now().strftime("%Y%m%d-%H%M%S")+DEST.suffix))
    prs.save(DEST)
    manifest = {"created_at":datetime.now().astimezone().isoformat(), "source_report_sha256":sha256(REPORT.read_bytes()).hexdigest(),
                "slides":slides_spec, "native_charts":4, "metrics":source_metrics,
                "style":{"ratio":"16:9","palette":COL,"fonts":[FONT,NUMBER_FONT,"Consolas"],
                         "format":"editable native PowerPoint text/charts/tables/diagrams, current browser screenshot"},
                "scope":"Current report only; no prior deck or old UI recording was used. No training, LLM inference, or experiment benchmark was run."}
    (BASE/"source/deck_spec.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    outline=["# 实验三新版PPT内容与风格", "", "以当前30页报告为唯一内容基准；16页主讲、2页答辩备用。", "",
             "16:9；暖白背景、深石墨标题、赭橙/青绿数据强调。标题和正文可编辑，统计用嵌入Excel图表。", ""]
    for row in slides_spec:
        outline += [f"## {row['slide']:02d} {row['title']}", row.get("subtitle", ""),
                    "报告来源："+row["report_pages"], "讲解提示："+row["speaker_notes"].split("【报告对应】")[0].strip(), ""]
    (BASE/"outline.md").write_text("\n".join(outline),encoding="utf-8")
    print("PPTX",DEST,"bytes",DEST.stat().st_size)
    print("SLIDES",len(prs.slides),"CHARTS",sum(sh.has_chart for s in prs.slides for sh in s.shapes))
    print("NOTES",sum(s.has_notes_slide for s in prs.slides))
    for i,s in enumerate(prs.slides,1): print(i,len(s.shapes),slides_spec[i-1]["title"])


if __name__ == "__main__":
    main()
