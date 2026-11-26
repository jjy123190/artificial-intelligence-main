"""Render original evidence tables, never substitute val/teacher for test/local."""
from __future__ import annotations
import argparse
from pathlib import Path
import json

from .data import digest_file
from .study_literature import read,write


def percent(value):return f"{100*value:.2f}%"
def score(value):
    if value.get('accuracy') is None:return '数值失败（不报告准确率）'
    return f"{100*value['accuracy']:.2f} ± {100*value['ci95']:.2f}%"
def diff(value):return f"{100*value['mean']:+.3f} ± {100*value['ci95']:.3f}pp"


def build(root,output):
    inference=root/'literature_inference_20261125';geometry=root/'teacher_geometry_20261125';diagnostic=root/'literature_diagnostic_20261125'
    paths=[inference/'selection.json',geometry/'selection.json',diagnostic/'diagnostic.json',
           inference/'audit.json',geometry/'audit.json',diagnostic/'audit.json']
    if not all(p.is_file() for p in paths):raise ValueError('All studies and audits must be complete before rendering results')
    for directory in (inference,geometry,diagnostic):
        audit=read(directory/'audit.json')
        if audit.get('status')!='passed' or audit.get('test_accessed') is not False:raise ValueError('Study audit failed')
        for name,expected in audit['files'].items():
            if digest_file(directory/name)!=expected:raise ValueError('Evidence changed after audit')
    a,b,c=read(paths[0]),read(paths[1]),read(paths[2])
    if not all(x['test_accessed'] is False for x in (a,b,c)):raise ValueError('Not a validation-only study')
    lines=['# 联网方法实测：验证结果与负结果（2026-11-25）','',
           '> **这里全部是validation结果，不是test成绩；外部教师不是本地自训模型。** ±为逐任务95%区间半宽，基于复用类别且未校正多重选型。历史正式test仍为56.4733%/68.8133%。','',
           '## 1. 冻结轻量特征：搜索结果（160配对任务/shot）','',
           '| 方法 | 1-shot | 5-shot |','|---|---:|---:|']
    names={'map_whitened':'MAP＋原白化','map_taskcenter_whitened':'MAP＋白化＋任务中心化',
           'ptmap_signed_taskcenter':'signed PT-MAP','tim_whitened':'TIM-GD＋白化','tim_taskcenter_whitened':'TIM-GD＋白化＋任务中心化'}
    rows={s:{r['id']:r for r in a['shots'][s]['candidates']} for s in ('1','5')}
    lines.append('| 现有continued | '+score(a['shots']['1']['reference']['search'])+' | '+score(a['shots']['5']['reference']['search'])+' |')
    for name,label in names.items():lines.append('| '+label+' | '+score(rows['1'][name]['search'])+' | '+score(rows['5'][name]['search'])+' |')
    lines+=['','## 2. 轻量候选确认（600任务/shot）','',
            '| 方法 | 1-shot | 5-shot | 1-shot配对差 | 5-shot配对差 |','|---|---:|---:|---:|---:|']
    lines.append('| 现有continued | '+score(a['shots']['1']['reference']['confirmation'])+' | '+score(a['shots']['5']['reference']['confirmation'])+' | — | — |')
    for name in ('map_whitened','map_taskcenter_whitened'):
        x,y=rows['1'][name],rows['5'][name]
        lines.append('| '+names[name]+' | '+score(x['confirmation'])+' | '+score(y['confirmation'])+' | '+diff(x['confirmation_paired'])+' | '+diff(y['confirmation_paired'])+' |')
    lines+=['','两个shot的预注册门槛均未通过（整体eligible=false）。未进行新test或发布。查询标签不进入推断函数；失败/低分任务不剔除。','',
            '## 3. 线性教师几何迁移（独立外部来源分支）','',
            '| 阶段/方法 | 1-shot | 5-shot |','|---|---:|---:|']
    grows={s:{r['id']:r for r in b['shots'][s]['candidates']} for s in ('1','5')}
    for name in sorted(grows['1']):
        lines.append('| 搜索 / '+name+' | '+score(grows['1'][name]['search'])+' | '+score(grows['5'][name]['search'])+' |')
    for shot in ('1','5'):
        winner=grows[shot][b['shots'][shot]['selected']]
        lines+=['',f"{shot}-shot确认选择`{winner['id']}`：**{score(winner['confirmation'])}**，相对现有模型{diff(winner['confirmation_paired'])}；eligible={str(b['shots'][shot]['eligible']).lower()}。"]
    lines+=['','此结果仅否定这组线性冻结特征映射，不否定训练级蒸馏。拟合只用基础train的6400张同ID特征，未将教师val标签用于训练。','',
            '## 4. 同任务外部编码器诊断（600任务/shot）','',
            '两个模型共同覆盖的3200张val图像；centered头使用同6400张train图片的均值。以下各行之间任务ID相同，但**不与上面不同任务池的分数相减**。','',
            '| 方法 | 1-shot | 5-shot |','|---|---:|---:|']
    labels={'continued_published':'当前自训0.51M＋正式推断配置','student_centered_proto':'自训0.51M＋中心化原型',
            'teacher_centered_proto':'外部12.42M encoder＋同中心化原型','student_centered_graph':'自训0.51M＋中心化图/原型头',
            'teacher_centered_graph':'外部12.42M encoder＋同中心化图/原型头','student_ptmap':'自训0.51M＋signed PT-MAP',
            'teacher_ptmap':'外部12.42M encoder＋signed PT-MAP'}
    for name,label in labels.items():
        lines.append('| '+label+' | '+score(c['shots']['1']['methods'][name])+' | '+score(c['shots']['5']['methods'][name])+' |')
    lines+=['','| 教师−学生（同head） | 1-shot配对差 | 5-shot配对差 |','|---|---:|---:|']
    for h,label in [('centered_proto','中心化原型'),('centered_graph','中心化图/原型'),('ptmap','signed PT-MAP')]:
        first=c['shots']['1']['matched_head_pairs'].get(h);second=c['shots']['5']['matched_head_pairs'].get(h)
        lines.append('| '+label+' | '+(diff(first) if first else '数值失败，未比较')+' | '+(diff(second) if second else '数值失败，未比较')+' |')
    lines+=['','外部中心化图/原型相对当前正式推断配置：'+diff(c['shots']['1']['teacher_vs_published'])+' / '+diff(c['shots']['5']['teacher_vs_published'])+'。',
            '', '**解释边界**：容量、原训练、归一化、预处理均不同；这里支持“更强表示值得投入”，不证明差值仅由参数量产生。教师只用encoder而非FEAT attention；作者train64声明及一致划分不是训练历史独立审计。',
            '', '## 5. 数值异常、来源与发布边界','',
            '- 原300次Sinkhorn上限在外部teacher PT-MAP触发质量守卫；`diagnostic_failure.json`保留。',
            '- 单独登记的补充方案使用作者1000次上限，师生同改；其他超参/任务不变。若仍失败，整组无accuracy，不筛掉困难任务。',
            '- 校准/传导方法使用未标注query；MAP/图头利用声明的均衡协议，TIM有均匀边际熵偏好。',
            '- search和confirmation图片互斥但类别相同，且整体验证类别已经复用；confirmation参与选择，不是独立测试。',
            '- 教师读取仅本地已存在且SHA256匹配的作者权重。额外固定16张train/val cache重算通过；不是全量教师缓存重算。',
            '- 所有新研究均未访问新的test图像/特征；本次没有更改正式checkpoint、配置、结果或网页推理。',
            '- 长训练尚未执行；新增训练级蒸馏代码仅完成合成输入与回归测试。','',
            '## 6. 审计和复现','']
    for directory in (inference,geometry,diagnostic):
        report=read(directory/'audit.json')
        lines.append(f"- `{directory.name}`：{report['score_files']}份成功score文件，{report['episode_predictions']:,}条方法×任务预测记录；audit={report['status']}。")
    lines+=['','上面计数包含对照重复记录及诊断复用，不能当作相互独立的新任务数。复现命令和原理见[LITERATURE_IMPROVEMENT_PLAN.md](LITERATURE_IMPROVEMENT_PLAN.md)。','',
            '所有表格由`python -m experiments.lab04.literature_report`从审计后的JSON生成，不手填更高数字。','']
    output.write_bytes(('\n'.join(lines)).encode('utf-8'))
    evidence={'kind':'validation_only_report','test_accessed':False,'published_model_changed':False,
              'input_hashes':{str(p):digest_file(p) for p in paths},'markdown_sha256':digest_file(output),
              'generator_sha256':digest_file(__file__),'external_teacher_disclosed':True,'actual_distillation_training_completed':False}
    write(root/'literature_20261125'/'report_evidence.json',evidence)
    return {'report':str(output),'sha256':evidence['markdown_sha256'],'test_accessed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='outputs/lab04/next_study');p.add_argument('--output',default='experiments/lab04/LITERATURE_VALIDATION_RESULTS.md')
    args=p.parse_args();print(json.dumps(build(Path(args.root),Path(args.output)),ensure_ascii=False))


if __name__=='__main__':main()
