"""Build an evidence-checked v2 addendum; keep v1 deliverables as historical records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .verify import verify_artifacts


LABELS = {"baseline":"Baseline", "maml":"二阶 MAML", "calibrated":"完整校准", "full":"完整校准", "alignment_only":"仅训练对齐",
    "mean_only":"中心化＋均值", "variance_only":"中心化＋方差", "prototype_only":"中心化＋原型更新",
    "mean_variance":"中心化＋均值方差", "no_alignment_full":"去除训练对齐",
    "transform_only":"仅训练类中心化", "no_transform_full":"去除中心化", "unbalanced_full":"去除均衡先验",
    "original_baseline":"原版 Baseline 复测", "original_full":"原版完整校准复测"}


def write_report(artifacts, output):
    artifacts, output = Path(artifacts), Path(output)
    audit = verify_artifacts(artifacts)
    result=json.loads((artifacts/"results.json").read_text(encoding="utf-8"))
    if result.get("schema_version")!=2:raise ValueError("A verified v2 study is required")
    rows={(r["method"],r["shots"]):r for r in result["rows"]}
    counts=result["protocol"]["test_episodes_per_shot"]
    def score(method,shot):
        r=rows[(method,shot)];return f"{r['accuracy']*100:.2f}% ± {r['ci95']*100:.2f} pp"
    def delta(method,shot):
        r=rows[(method,shot)];return f"{r['paired_delta']*100:+.2f} ± {r['paired_ci95']*100:.2f} pp"
    lines=["# 实验四优化补充报告", "", "## 版本与结论", "",
        f"本报告读取并独立核验 {result['completed_utc']} 生成的 v2 实测结果，重算{audit['episodes_recomputed']}个任务、{audit['summary_rows_recomputed']}行汇总。原版报告、PPT和视频是 v1 历史记录，不以新分数覆盖旧负结果。",
        "", "| 方法 | 1-shot | 5-shot |", "|---|---:|---:|"]
    for method in ("baseline","maml","full","prototype_only","original_baseline","original_full"):
        lines.append(f"| {LABELS[method]} | {score(method,1)} | {score(method,5)} |")
    lines += ["", "± 表示95%置信区间半宽；pp为百分点，不能解释为相对百分比涨幅。所有方法在本轮相同任务上比较。", ""]
    for shot in (1,5):
        row=rows[("full",shot)]
        outcome=("区间高于零，本次固定模型下可确认改善" if row["paired_delta"]-row["paired_ci95"]>0
                 else "区间低于零，完整方案低于基线" if row["paired_delta"]+row["paired_ci95"]<0
                 else "区间跨零，暂不能确认改善")
        lines.append(f"{shot}-shot完整校准相对同版Baseline的配对差为{delta('full',shot)}，{outcome}。")
    lines += ["", "## 数据与公平比较", "",
        f"沿用mini-ImageNet的64/16/20互斥类别划分；训练64×600=38,400张、验证16×600=9,600张、测试20×600=12,000张。最终测试每种shot固定{counts}个任务，每任务5类、每类15张查询图片。支持与查询互不重叠，方法间共享样本ID。",
        "原版测试成绩已经公开，因此本轮属于已有基准上的后续优化，而不是从未接触过测试分布的独立外部验证。所有新参数仍只依据验证类选择，测试打开前写入权重指纹与配置锁。",
        "校准搜索使用每验证类200张图片，复核使用另外400张；两池图片互斥，但来自同一组验证类别，复核结果参与选择，不把复核分数作为无偏最终泛化估计。",
        "", "## 训练与算法改进", "",
        f"骨干从原版[8,16,24,32]扩展为{result['model']['widths']}，参数量{result['model']['parameters_baseline']:,}，仍为轻量ResNet12而非论文标准宽度模型。40轮监督预训练，依据验证集1/5-shot平均表现选择第{result['training']['warm_ce']['selected_epoch']}轮。标签平滑0.05、AdamW权重衰减0.0005。",
        "Baseline与对齐分支从相同检查点出发，使用相同图片、顺序、两次增强、交叉熵步数和3轮训练；唯一差别为对齐权重0与0.5。分布对齐约束两次增强在批内的均值和方差。",
        f"MAML使用全网络真实二阶梯度，训练预算{result['training']['maml']['tasks']}个任务；验证选中{result['training']['maml']['selected_tasks']}任务的检查点。测试适应参数按shot在验证类选定：1-shot为{result['maml_adaptation']['1']['steps']}步、学习率{result['maml_adaptation']['1']['learning_rate']}；5-shot为{result['maml_adaptation']['5']['steps']}步、学习率{result['maml_adaptation']['5']['learning_rate']}。测试只用支持标签更新，不使用查询标签。",
        "新推理流程为：L2特征 → 减去训练类特征中心 → 固定幂变换并归一化 → 支持原型估计 → 近邻训练类均值方差收缩 → 方差尺度混合 → 查询软分配更新原型。训练中心和先验从基础训练类计算。",
        "均衡软分配使用Sinkhorn迭代约束查询类别总质量，仅利用无标签特征与已知均衡任务协议。这比归纳式单张预测使用更多信息，表中明确区分。真实类别数量不均衡时，不保证该先验有效。",
        "", "### 验证集锁定参数", "", "| 设置 | 中心系数 | 均值系数 | 方差混合 | 原型权重 | 迭代数 | 温度 | 均衡先验 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for shot in (1,5):
        cfg=result["calibration"][str(shot)];space=result["study"]["feature_spaces"][str(shot)]
        lines.append(f"| {shot}-shot | {space['center_strength']} | {cfg['mean_weight']} | {cfg['variance_blend']} | {cfg['prototype_weight']} | {cfg['prototype_steps']} | {cfg['temperature']} | {'开启' if cfg['balanced_queries'] else '关闭'} |")
    lines += ["", "## 消融实验", "", "| 策略 | 1-shot | 相对Baseline | 5-shot | 相对Baseline |", "|---|---:|---:|---:|---:|"]
    for method in ("alignment_only","transform_only","mean_only","variance_only","prototype_only","mean_variance","full","no_alignment_full","no_transform_full","unbalanced_full"):
        lines.append(f"| {LABELS[method]} | {score(method,1)} | {delta(method,1)} | {score(method,5)} | {delta(method,5)} |")
    lines += ["", "‘仅均值/方差/原型’消融保留训练对齐骨干和固定特征中心化；另设中心化与均衡先验的独立消融，避免把预处理贡献误记为某一单模块贡献。配对差置信区间使用同任务差值计算，而非两个独立区间相减。",
        "本次完整方案与‘去除训练对齐’的两种shot均值相同，不能把训练对齐写成有效创新。仅原型更新已接近完整方案，说明其贡献大于叠加更多校准模块；MAML优化后仍低于同版Baseline，不宣称元学习普遍更差。", "", "## 网站与本地部署", "",
        "网站新增实验四独立入口与实验工作台。可选择1/5-shot、三种方法和校准策略、指定或随机生成任务种子；三方法对照核对样本ID和完整结果版本，防止混用任务或新旧权重。",
        "结果区展示支持样本、按类召回、误判与正确筛选、真实类别筛选、低分排序、单图五类分数、混淆矩阵与二维特征/原型。特征校准不伪造点位移动，MAML点位来自真实网络更新。记录导出含种子、模型指纹、配置与逐图预测，不导出重复图片数据。",
        "页面在390/768/1366宽度上通过真实浏览器测试和axe无障碍检查，覆盖错误状态、无效输入、导出、页面间状态保留及旧实验入口。模型与数据均可随源码在本机CPU运行。该部署是端侧可行性的CPU示范，不声称已在手机或ARM设备实测。"]
    edge=artifacts/"edge_benchmark.json"
    if edge.exists():
        data=json.loads(edge.read_text(encoding="utf-8"))
        lines += ["", "### 同口径CPU实测耗时", "", "所有方法均重新编码图像，以下为热启动编码+分类中位数/P95，不混用缓存特征计时。", "", "| 方法 | shot | 中位数 ms | P95 ms | 重复次数 |", "|---|---:|---:|---:|---:|"]
        for row in data["rows"]:
            lines.append(f"| {LABELS.get(row['method'],row['method'])} | {row['shots']} | {row['encoding_and_classification_median_ms']:.1f} | {row['encoding_and_classification_p95_ms']:.1f} | {row['repetitions']} |")
    lines += ["", "## 局限与后续方向", "",
        "1. 单训练种子的任务级区间不覆盖随机初始化和重训练波动；后续需多种子重复。",
        "2. 更宽轻量骨干与更长训练预算也会提高成绩，不能将原版到新版的全部差异归因于校准。",
        "3. 均衡转导使用额外的整批查询分布信息，不能直接等价于归纳式真实业务推断。",
        "4. softmax类别分数并不意味着概率已经可靠校准；真实部署还需概率校准、分布外检测和拒识。",
        "5. 模型仍远小于常见论文骨干，不能用该分数代表标准ResNet12上限。医学、工业等真实业务还需合法独立数据与领域验证。",
        "", "## 复现与证据文件", "",
        "完整命令见源码/experiments/lab04/OPTIMIZATION.md；核验命令：python main.py lab04 verify。",
        "results.json保存汇总与权重指纹；episodes.json与evaluation_episodes.jsonl保存完整逐任务预测；v2_calibration_locked.json保存测试前配置锁；v2_validation_tuning.json保留全部候选和复核；training.json、maml_training.json保存训练与适应选型；history/v1保留原版结果、权重和演示池。",
        "", "## AI使用说明", "",
        "本次代码重构、实验流程设计、界面实现与文档生成使用AI辅助。准确率和置信区间来自本地实际模型推理与可重算的逐任务记录，不由AI生成数值替代。"]
    output.mkdir(parents=True,exist_ok=True)
    markdown=output/"实验四_优化补充报告.md"
    markdown.write_text("\n".join(lines) + "\n",encoding="utf-8")
    return markdown


def export_office(markdown, *, report_title="实验四优化补充报告", footer_text="实验四 · v2优化补充报告"):
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.oxml.ns import qn
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from xml.sax.saxutils import escape
    doc=Document();section=doc.sections[0];section.top_margin=section.bottom_margin=Cm(2);section.left_margin=section.right_margin=Cm(2)
    normal=doc.styles["Normal"];normal.font.name="Microsoft YaHei";normal.font.size=Pt(10)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"),"Microsoft YaHei")
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    body=ParagraphStyle("Body",fontName="STSong-Light",fontSize=10,leading=16,spaceAfter=7)
    heading=ParagraphStyle("Heading",parent=body,fontSize=14,leading=21,spaceBefore=13,spaceAfter=8,keepWithNext=True)
    title=ParagraphStyle("Title",parent=body,fontSize=22,leading=30,spaceAfter=18,alignment=TA_CENTER,keepWithNext=True)
    tiny=ParagraphStyle("Tiny",parent=body,fontSize=8,leading=13,spaceAfter=0)
    flow=[];lines=markdown.read_text(encoding="utf-8").splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith("|"):
            table=[]
            while i<len(lines) and (not lines[i].strip() or lines[i].strip().startswith("|")):
                text=lines[i].strip()
                if text.startswith("|") and not set(text.replace("|","").replace(":","").replace(" ","")) <= {"-"}:
                    table.append([c.strip() for c in text.strip("|").split("|")])
                i+=1
            t=doc.add_table(rows=1,cols=len(table[0]));t.style="Light Shading Accent 1"
            for c,v in zip(t.rows[0].cells,table[0]):c.text=v
            for row in table[1:]:
                for c,v in zip(t.add_row().cells,row):c.text=v
            table_flow=Table([[Paragraph(escape(c),tiny) for c in row] for row in table],repeatRows=1,hAlign="LEFT",colWidths=[17*cm/len(table[0])]*len(table[0]))
            table_flow.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eee6dc")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#d2c5b8")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
            flow += [table_flow,Spacer(1,10)];continue
        if line.startswith("# "):
            text=line[2:];doc.add_heading(text,0);flow.append(Paragraph(escape(text),title))
        elif line.startswith("##"):
            text=line.lstrip("# ");doc.add_heading(text,1 if line.startswith("## ") else 2);flow.append(Paragraph(escape(text),heading))
        else:
            doc.add_paragraph(line)
            paragraph = Paragraph(escape(line),body)
            # Keep body paragraphs together when they fit on a fresh page.
            paragraph.allowOrphans = 0
            paragraph.allowWidows = 0
            flow.append(paragraph)
        i+=1
    doc.save(markdown.with_suffix(".docx"))
    def footer(canvas,document):
        canvas.setFont("STSong-Light",9);canvas.setFillColor(colors.HexColor("#68564a"));canvas.drawString(2*cm,1.1*cm,footer_text);canvas.drawRightString(19*cm,1.1*cm,str(document.page))
    SimpleDocTemplate(str(markdown.with_suffix(".pdf")),pagesize=(21*cm,29.7*cm),rightMargin=2*cm,leftMargin=2*cm,topMargin=1.6*cm,bottomMargin=1.6*cm,title=report_title).build(flow,onFirstPage=footer,onLaterPages=footer)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--artifacts",default="outputs/lab04")
    p.add_argument("--output",required=True);p.add_argument("--office",action="store_true")
    args=p.parse_args();markdown=write_report(args.artifacts,args.output)
    if args.office:export_office(markdown)
    print(markdown)


if __name__=="__main__":main()
