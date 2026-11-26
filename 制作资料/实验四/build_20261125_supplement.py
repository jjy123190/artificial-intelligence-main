"""Build current research supplement from frozen, audited evidence (no re-training)."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'源码'
sys.path.insert(0,str(SOURCE))
from experiments.lab04.artifact_audit import audit_publication
from experiments.lab04.study_report import export_office


def build(artifacts=None):
    directory=Path(artifacts) if artifacts else SOURCE/'outputs/lab04'
    audited=audit_publication(directory);new=audited['wideview'];wide=audited['wide']
    if new is None:raise ValueError('No audited two-view results')
    lines=['# 实验四：较宽双视图校准与分页面展示','',
        '2026-11-25增量记录。补充此前v1、v2和较宽模型报告，不覆盖旧实验、原始报告和负结果。测试结果经独立逐任务重算，本记录不将单次网页演示分数作为benchmark。','',
        '## 一、真实分类效果','',
        '| 方案 | 1-shot | 5-shot |','|---|---:|---:|']
    for name,data in [('较宽骨干＋度量训练',wide),('较宽骨干＋双视图校准',new)]:
        cells=[]
        for shot in (1,5):
            r=next(r for r in data['rows'] if r['shots']==shot);cells.append(f"{r['accuracy']*100:.2f}% ± {r['ci95']*100:.2f}pp")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','两种shot均为600个相同配对任务，5-way、每类15张query。支持和查询互斥；完整20测试类、每类600张图像池。±为任务均值95%区间半宽，不包含重新训练模型的波动。',
        '', '| 新方案相对原较宽方案 | 平均配对差 | 95%区间 |','|---|---:|---:|']
    for row in new['rows']:
        d,c=row['paired_vs_wide'],row['paired_vs_wide_ci95'];lines.append(f"| {row['shots']}-shot | +{d*100:.4f}pp | [{(d-c)*100:.4f}, {(d+c)*100:.4f}]pp |")
    lines+=['','两个区间均高于零，支持本次固定模型与协议下的改善；这不是所有数据域与训练种子的保证。相对原较宽模型使用相同权重、没有新增训练；相对轻量模型仍包含此前较宽容量与额外训练的差异，不能称为同骨干消融。','',
        '## 二、候选研究与数据隔离','',
        '先进行了18个训练类内协方差／轻量与较宽特征融合候选的验证比较。1-shot选回原方案，5-shot验证复核均值仅增加0.17pp，配对95%半宽0.194pp，区间跨零。这组没有读取测试图像、没有发布，全部负结果保留。',
        '后续较宽双视图只增加同图的水平翻转，不引入外部权重、额外类别或额外标注。候选网格在运行前保存：mean/concat两种聚合、0.25/0.5/0.75三个白化指数、0/0.5/1三个类内协方差混合系数。图传播配置保持此前验证锁定值。',
        '按原16验证类每类200张用于搜索、400张用于复核，160个搜索任务、400个复核任务；按搜索均值选前四，再按相对固定较宽参考的复核配对区间下界选型。复核参与选择，且仍重复使用同一验证类别，不是独立无偏泛化估计。',
        '两种shot选中同一特征设置：L2归一化原图与翻转特征后平均，再归一化；训练类中心化1.0，白化指数0.5，类内／总体协方差混合0.5。1-shot图传播权重0.5，5-shot保持0，即关闭图传播。',
        '验证锁和所有文件指纹写入后，才打开固定测试池；每条预测、五类分数、支持与查询ID、用于独立评分的标签和五个既有方法的参考准确率全部保留。未按测试成绩换种子、删困难任务或再次调整候选。','',
        '## 三、计算代价与本机部署','',
        '每种方法、每种shot均重新编码图片，在同一轮本机CPU环境各重复20次。下面排除加载、文件校验、HTTP及缩略图渲染，不与缓存特征计时混排。旧耗时文件保留历史记录，不将新记录的快慢变化归为算法改进。',
        '', '| 方法 | shot | 编码＋分类中位数(ms) | P95(ms) |','|---|---:|---:|---:|']
    for r in audited['edge_benchmark']['rows']:
        if r['method'] in ('wide','wideview'):
            lines.append(f"| {'较宽单视图' if r['method']=='wide' else '较宽双视图'} | {r['shots']} | {r['encoding_and_classification_median_ms']:.1f} | {r['encoding_and_classification_p95_ms']:.1f} |")
    lines+=['','双视图需要每图两次编码，计算时间约为较宽单视图两倍，网站仍默认较轻的图传播方案。已在Windows本机CPU完成独立推理；未在手机、ARM、开发板或专用NPU实测，也没有医学或工业领域验证。','',
        '## 四、信息架构与可视化','',
        '此前长页拆为独立URL：性能对比、模块消融、可靠性、端侧性能、历史与来源。另有现场推理、方法与协议、要求与交付。各页可直接访问、刷新与浏览器前进后退；站内切换保留shot、选择的方法和推理记录，不触发额外推理。',
        '性能页使用可选择的准确率与95%区间条形图；消融页绘制相对同版Baseline的配对差点区间，并明确不是相对完整模型的显著贡献证据；可靠性页展示任务分位、改善/持平/下降比例和分箱平均分数与实际正确率；端侧页可视化中位数与P95。精确表格仍可展开，图不是不可核对的装饰。',
        '可靠性图明确标为原双视图固定预测诊断，不假装所有方法都拥有相同可靠性。空分箱不补造数据，图片重复出现次数不说成独立样本数。',
        '取消等待只终止前端等待并阻止后续对比调用，不谎称服务器已停止计算；旧响应不能覆盖立即重试的新任务。推理JSON经过完整类型和统计检查，坏响应不进入图表。','',
        '## 五、要求核对与复现边界','',
        '原要求的ResNet12、小样本范式/漂移分析、交叉熵＋分布对齐、均值方差校准、原型更新、Baseline/MAML/校准比较、mini-ImageNet实验、消融与限制，以及本地CPU部署均有实现和文件证据；详细对应表为experiments/lab04/REQUIREMENTS_AUDIT.md。',
        '1–5人分组、平台组员关联、创建项目与管理员提交属于课程平台动作，当前代码不能证明这些外部动作已完成。如果教师另行要求具体ARM/手机硬件，也需额外完成该设备实测。不得据本机部署宣称指定硬件要求已经满足。',
        '原始报告/PPT/视频为v1；本补充与此前优化记录一起提供当前结论。新增脚本：study_cached_refinement.py、study_wide_views.py、evaluate_wide_views.py、wideview_audit.py。推理策略calibration=wideview；核验命令python main.py lab04 verify。',
        '网站方法并非理论最优或零误判。后续优先考虑多训练种子、独立跨域数据、非均衡查询协议与指定端侧硬件，而非按已公开测试结果反复调参。AI辅助实现、界面与文档；准确率来源为真实模型预测和可重算证据。']
    dest=ROOT/'deliverables/lab4/优化补充/实验四_双视图宽骨干与分页面说明.md';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    export_office(dest,report_title='实验四较宽双视图与分页面展示',footer_text='实验四 · 2026-11-25增量记录')
    print(dest)


if __name__=='__main__':build(sys.argv[1] if len(sys.argv)>1 else None)
