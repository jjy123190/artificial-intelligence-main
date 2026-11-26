"""Build editable current-version presentation from audited evidence and actual captures."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from io import BytesIO

ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'源码';sys.path.insert(0,str(SOURCE))
from experiments.lab04.artifact_audit import audit_publication
from backend.lab_four import LABELS
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.xmlchemy import OxmlElement

INK='292622';MUTED='665F56';BG='F8F4EC';SURFACE='EFE6D9';ACCENT='984424';BORDER='D7C9B7';GREEN='436248';WHITE='FFFFFF'


def text(slide,value,x,y,w,h,size=18,color=INK,bold=False):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    frame=shape.text_frame;frame.word_wrap=True;frame.margin_left=frame.margin_right=0;frame.margin_top=frame.margin_bottom=0
    for i,line in enumerate(str(value).split('\n')):
        para=frame.paragraphs[0] if i==0 else frame.add_paragraph();para.text=line
        para.font.name='Microsoft YaHei';para.font.size=Pt(size);para.font.bold=bold;para.font.color.rgb=RGBColor.from_string(color)
        para.space_after=Pt(7)
        for run in para.runs:
            run.font.name='Microsoft YaHei'
            properties=run._r.get_or_add_rPr()
            east=properties.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ea')
            if east is None:east=OxmlElement('a:ea');properties.append(east)
            east.set('typeface','Microsoft YaHei')
    return shape


def rect(slide,x,y,w,h,color,line=None):
    shape=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(color)
    shape._element.spPr.append(OxmlElement('a:effectLst'))
    if line:shape.line.color.rgb=RGBColor.from_string(line)
    else:shape.line.fill.background()
    return shape


def build(artifacts,assets,output):
    audit=audit_publication(Path(artifacts));core=audit['core'];assert audit['wideview'] and audit['edge_benchmark']
    rows={(r['method'],r['shots']):r for r in core['rows']}
    for mode in ('graph','multiview','wide','wideview','continued'):rows.update({(r['method'],r['shots']):r for r in audit[mode]['rows']})
    output=Path(output);output.mkdir(parents=True,exist_ok=True);assets=Path(assets)
    for name in ('overview','performance','error-analysis','features','deployment'):
        if not(assets/(name+'.png')).is_file():raise FileNotFoundError('Need actual current UI capture: '+name)
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    headings=[]
    def new(title,kicker,subtitle=''):
        slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor.from_string(BG)
        index=len(prs.slides);text(slide,kicker,.55,.3,10,.3,11,MUTED);text(slide,title,.55,.86,12.1,.68,30,INK,True)
        if subtitle:text(slide,subtitle,.55,1.65,12.05,.66,15,MUTED)
        rect(slide,.55,7.06,12.15,.01,BORDER);text(slide,'实验四 · 当前已发布版本 / 2026-11-25',.55,7.16,10,.23,10,MUTED);text(slide,f'{index:02d}',12.22,7.15,.5,.25,10,MUTED)
        headings.append(title);return slide
    def note(slide,value):slide.notes_slide.notes_text_frame.text=value
    def pairbar(slide,method,y):
        row1,row5=rows[(method,1)],rows[(method,5)]
        text(slide,LABELS.get(method,method),.65,y,3.05,.35,14,INK,method=='continued')
        for n,row,x in ((1,row1,3.85),(5,row5,8.35)):
            rect(slide,x,y+.02,3.1,.24,SURFACE);rect(slide,x,y+.02,3.1*row['accuracy'],.24,ACCENT if method=='continued' else 'A1835E')
            lo=x+3.1*max(0,row['accuracy']-row['ci95']);hi=x+3.1*min(1,row['accuracy']+row['ci95'])
            rect(slide,lo,y+.13,max(hi-lo,.005),.008,INK);rect(slide,lo,y+.06,.008,.14,INK);rect(slide,hi,y+.06,.008,.14,INK)
            text(slide,f"{row['accuracy']*100:.2f}%",x+3.2,y-.025,.9,.35,14,INK,True)
    s=new('少量标注，如何可靠识别新类别？','FEW-SHOT LEARNING / 实验四','轻量 ResNet12 · 二阶 MAML · 特征统计校准 · 可复核的真实实验')
    text(s,'不是追求一次幸运抽样，\n而是检验固定任务上的改善。',.7,2.58,6.1,1.55,28,INK,True)
    text(s,'第33组\nmini-ImageNet · 64 / 16 / 20 类隔离\n本机 CPU 真实推理与独立页面',.7,4.57,6,1.27,19,MUTED)
    s.shapes.add_picture(str(assets/'overview.png'),Inches(7.05),Inches(2.5),width=Inches(5.45))
    note(s,'本次展示当前已发布版本，不把仍在训练的候选放入正式成绩。原v1报告、PPT与视频保留。')
    s=new('把表征问题和统计问题分开','01 / 研究问题','医疗、工业缺陷和小众物种是现实动机；本实验只验证 mini-ImageNet。')
    for i,(title,body) in enumerate([('表征偏移','基础类的判别方向\n未必适合未见类别'),('估计波动','少量支持样本影响类中心\n1-shot不能直接估计方差'),('公平比较','训练预算、转导信息、计算代价\n不能与准确率混为一个结论')]):
        x=.7+i*4.12;text(s,f'0{i+1}',x,2.7,1,.65,35,ACCENT,True);text(s,title,x,3.65,3.8,.55,24,INK,True);text(s,body,x,4.42,3.65,1.3,18,MUTED)
    note(s,'范式包括冻结表征最近原型、优化式MAML，以及基础类统计迁移和无标签query转导。没有医学或工业领域验证。')
    s=new('同一份考题，才有配对比较','02 / 数据与协议','64训练类 / 16验证类 / 20测试类，类别与任务内支持/query均隔离。')
    for i,(n,t) in enumerate([('38,400','训练图像'),('9,600','验证图像'),('12,000','测试图像')]):
        x=.8+i*4.1;text(s,n,x,2.7,3.8,.8,38,ACCENT,True);text(s,t,x,3.57,3.5,.45,19,MUTED)
    text(s,'5-way · 1 / 5-shot · 每类15张query',.8,4.55,11.6,.55,25,INK,True)
    text(s,'每种shot 600个共同任务；query顺序打乱，标签只在预测后评分。\n验证搜索与复核图片互斥，但来自同一组验证类别，不是独立外部验证。',.8,5.4,11.7,1.15,18,MUTED)
    note(s,'对照不仅共享seed，还核对样本ID、图片池、权重和发布指纹。20测试类已多轮使用，仍有公开基准选择偏差。')
    s=new('三种学习方式，不偷换决策规则','03 / 方法与训练','轻量骨干 [8,16,32,64]、130,160参数；GroupNorm避免query批统计泄漏。')
    y=2.6
    for name,body in [('Baseline','40轮监督暖启动（验证选39）＋3轮CE对照；冻结编码器后最近原型。'),('二阶 MAML','4000个元训练任务；全网络二阶梯度。测试支持集5步适应，query不更新。'),('统计校准','与Baseline同初始化、样本、增强和CE步数。\n对齐λ=0.5；基础类先验＋无标签query。')]:
        rect(s,.7,y,11.9,.01,BORDER);text(s,name,.8,y+.2,2.1,.55,23,INK,True);text(s,body,3.2,y+.18,9.05,.7,18,MUTED);y+=1.24
    note(s,'MAML1-shot支持适应学习率.025，5-shot为.05。MAML额外元训练预算不同，不宣称严格等预算。')
    s=new('收缩统计量，但不假装知道真实分布','04 / 校准思路','训练视图分布对齐 → 基础类中心化 → 均值方差收缩 → 原型自适应')
    text(s,'p̃c = (1 − a) pc + a mc',.85,2.65,11.3,.62,27,ACCENT,True)
    text(s,'p_c^(t+1) = [nc p̃c + β Σq wqc zq] / [nc + β Σq wqc]',.85,3.63,11.3,.74,23,INK,True)
    text(s,'1-shot方差依靠训练先验；对角方差不是完整高斯分布。\n均衡分配只利用协议中每类query数量相等的先验，不读query标签。\n特征校准不等于概率校准；错误软分配也可能拉偏原型。',.85,4.72,11.4,1.62,20,MUTED)
    note(s,'分布对齐只对同图两种增强的均值/方差约束，不等价于已证明跨类漂移消除。实现不是Distribution Calibration原论文完整复现。')
    s=new('把均值、区间与信息条件一起报告','05 / 实测结果','各600个配对任务；图表从0开始，细线为95%区间；精确数值见整合报告。')
    text(s,'1-shot',3.85,2.5,3.5,.35,15,MUTED);text(s,'5-shot',8.35,2.5,3.5,.35,15,MUTED)
    for i,m in enumerate(('baseline','maml','full','graph','multiview','wide','wideview','continued')):pairbar(s,m,3.0+i*.42)
    text(s,'原型/图传播为转导；更宽模型含额外训练预算，不能把所有提升归为校准。',.8,6.65,11.8,.3,14,MUTED)
    note(s,'数值从已审计JSON直接生成；并非选择网页容易任务后的演示准确率。单训练种子的任务区间不覆盖重训练波动。')
    s=new('同一模型容量，续训带来可复核改善','06 / 最新已发布方案','仍为509,856参数；额外16轮训练，验证选第12轮；分类配置不再搜索。')
    for i,n in enumerate((1,5)):
        r=rows[('continued',n)];x=.85+i*6.2;text(s,f'{n}-shot',x,2.58,5.5,.4,19,MUTED);text(s,f"{r['accuracy']*100:.2f}%",x,3.1,5.5,.94,43,ACCENT,True)
        text(s,f"vs旧较宽双视图 {r['paired_vs_wideview']*100:+.4f} ± {r['paired_vs_wideview_ci95']*100:.4f} pp",x,4.23,5.5,.8,19,INK,True)
    text(s,'两个配对区间均高于零；额外训练预算不冒充新校准模块贡献。\n仍沿用mean / 白化0.5 / 类内0.5；5-shot图传播关闭。',.85,5.59,11.7,1,20,MUTED)
    note(s,'额外16轮训练选择第12轮，但预算披露全部实际消耗。原wide.pt与旧成绩不覆盖。验证600任务/shot门槛通过后才锁定一次原1200测试任务。')
    s=new('模块不是越多越好，负结果必须可见','07 / 消融与解释','相同骨干的推理开关与独立训练对照分开，直接配对差另行重算。')
    text(s,'分布对齐',.85,2.7,3.7,.45,23,INK,True);text(s,'完整方案去对齐后均值未下降，\n不能把λ=0.5宣称为已证实有效。',.85,3.43,5.5,1.35,19,MUTED)
    text(s,'原型更新',7.02,2.7,4,.45,23,INK,True);text(s,'仅原型更新已接近完整方案；\n均值、方差和均衡先验需各自核对。',7.02,3.43,5.5,1.35,19,MUTED)
    text(s,'网页可切换：模块直接对照 / 相对Baseline。\n原型＋均衡流程是组合改变，区间未作多重比较校正。',.85,5.45,11.7,1,20,ACCENT,True)
    note(s,'统计差来自相同任务差值，而非独立误差条相减。后处理诊断不用于重新选择测试参数。')
    s=new('可靠性与端侧代价，不只一个平均分','08 / 诊断与部署','模型分数不是正确率保证；CPU计时不冒充手机/ARM性能。')
    diag=audit['diagnostics']['rows'];text(s,f"轻量双视图任务P05—P95\n1-shot {diag[0]['task_p05']*100:.1f}%—{diag[0]['task_p95']*100:.1f}%\n5-shot {diag[1]['task_p05']*100:.1f}%—{diag[1]['task_p95']*100:.1f}%",.85,2.73,5.7,2.05,23,INK,True)
    timings={(r['method'],r['shots']):r for r in audit['edge_benchmark']['rows']}
    text(s,'旧双视图 → 续训：20次计时',7.02,2.73,5.4,.45,19,MUTED)
    text(s,'\n'.join(f"{n}-shot：{timings[('wideview',n)]['encoding_and_classification_median_ms']:.1f} → {timings[('continued',n)]['encoding_and_classification_median_ms']:.1f} ms" for n in (1,5)),7.02,3.5,5.4,1.45,23,ACCENT,True)
    text(s,'分位区间不是均值区间；相同结构的耗时差受系统负载影响，不宣称续训更快。',.85,5.7,11.6,.65,19,MUTED)
    note(s,'可靠性明确指轻量双视图，不将其ECE冒充较宽方案。新耗时和旧耗时记录分开。')
    s=new('从误判到证据，一条可检查的操作路径','09 / 当前UI与真实演示','独立路由、可视化、精确表格、来源指纹；任务切换不丢记录。')
    s.shapes.add_picture(str(assets/'error-analysis.png'),Inches(.75),Inches(2.58),width=Inches(7.75))
    text(s,'现场推理 → 误判筛选\n性能 → 消融 → 可靠性\n端侧性能 → 历史来源\n要求与交付',9.05,2.72,3.55,2.7,22,INK,True)
    text(s,'异常JSON不进入图表；取消等待\n不谎称终止服务器计算。',9.05,5.64,3.65,.95,16,MUTED)
    note(s,'配套Demo是实际浏览器录制，4次真实推理；不是静态截图拼接。另有字幕文件与响应证据。')
    s=new('实验可运行，不等于所有外部动作已完成','10 / 要求、边界与下一步','材料按当前发布版本生成；训练中的候选不混入正式结论。')
    for i,(title,body) in enumerate([('技术证据','模型、损失、MAML、统计校准、消融、mini-ImageNet和本地CPU部署均有实现与重算。'),('未外推的范围','单训练种子、公开基准重复使用、均衡转导；尚无跨域、手机/ARM或业务安全保证。'),('平台需确认','1–5人组队、关联组员、项目和管理员提交需在课程平台核对；本地代码无法代办证明。')]):
        y=2.6+i*1.28;text(s,title,.85,y,2.1,.45,21,ACCENT,True);text(s,body,3.3,y,9,.85,18,INK)
    note(s,'进一步研究优先多训练种子、独立域、不均衡query和指定硬件；不能为追目标分数反复调测试。当前追加训练候选需完成验证再决定是否锁定测试。')
    s=new('可复现，是结论可信的底线','11 / 复现与参考','源代码、原始任务、配置锁、负结果、当前报告与Demo证据同时保留。')
    text(s,'python main.py lab04 verify\npython main.py lab04 demo --calibration continued --shots 1\npython main.py web --port 8765',.85,2.55,11.7,1.3,21,ACCENT,True)
    text(s,'核心参考：Finn et al. (MAML, ICML 2017)；Snell et al. (ProtoNet, NeurIPS 2017)；\nYang et al. (Distribution Calibration, ICLR 2021)；Ravi & Larochelle (ICLR 2017)；Liu et al. (TPN, ICLR 2019)。\n本实现并非对应论文标准宽度或完整算法流程的分数复现。',.85,4.25,11.6,1.45,17,MUTED)
    text(s,'AI辅助实现与材料生成；准确率由真实推理与独立重算产生。',.85,6.14,11.6,.5,18,INK,True)
    note(s,'当前版本指纹 '+audit['artifact_revision'])
    target=output/'实验四_小样本分类与特征校准_当前汇报.pptx';prs.save(target)
    evidence={'artifact_revision':audit['artifact_revision'],'slide_titles':headings,'slides':len(prs.slides),'slide_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'source_images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.glob('*.png')},'unpublished_candidates_included':False}
    (output/'slides_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'slides':len(prs.slides),'output':str(target)},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',default=str(SOURCE/'outputs/lab04'))
    p.add_argument('--assets',default=str(SOURCE/'.local/current-demo-20261125/captures'));p.add_argument('--output',default=str(ROOT/'deliverables/lab4/当前版本'))
    args=p.parse_args();build(args.artifacts,args.assets,args.output)


if __name__=='__main__':main()
