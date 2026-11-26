"""Create a standalone current report from audited artifacts; never overwrites v1.

Every numeric table is populated from current publication audit results or a
paired recomputation of its frozen episode predictions. No running candidates
are substituted for published benchmark measurements.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'源码'
sys.path.insert(0,str(SOURCE))
from experiments.lab04.artifact_audit import audit_publication
from experiments.lab04.study_report import export_office
from backend.lab_four import LABELS


def pct(v):return f"{v*100:.2f}%"
def estimate(row):return f"{pct(row['accuracy'])} ± {row['ci95']*100:.2f}pp"
def delta(values):
    mean=statistics.mean(values);ci=1.96*statistics.stdev(values)/len(values)**.5
    return mean,ci

def table(lines,headers,rows):
    lines.extend(['','| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows);lines.append('')


def build(artifacts,output,office=True):
    artifacts=Path(artifacts).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    audit=audit_publication(artifacts);core=audit['core'];protocol=core['protocol'];model=core['model']
    if not audit['verified'] or core.get('schema_version')!=2:raise ValueError('Complete current v2 evidence is required')
    rows={(r['method'],r['shots']):r for r in core['rows']}
    for name in ('graph','multiview','wide','wideview','continued'):
        if audit[name]:rows.update({(r['method'],r['shots']):r for r in audit[name]['rows']})
    methods=[m for m in ('baseline','maml','full','graph','multiview','wide','wideview','continued') if (m,1) in rows]
    name='实验四_小样本分类与特征校准_当前整合报告'
    report=output/(name+'.md')
    if output==ROOT/'deliverables/lab4':raise ValueError('Write to a versioned current directory; retain historical files')
    lines=['# 小样本图像分类与特征校准：当前整合实验报告','',
        '实验四 · 第33组 · mini-ImageNet / 轻量ResNet12 / 二阶MAML / 统计校准','',
        f"本报告由当前已核验发布产物自动生成，覆盖核心{audit['episodes_recomputed']}个任务及扩展{audit['extension_episodes_recomputed']}个配对任务记录。结果版本指纹：{audit['artifact_revision']}。后续正在训练或尚未锁定测试的候选不进入本报告。旧v1报告、PPT和视频保留为历史材料，本文件用于独立阅读当前方案，不要求读者把多份旧报告拼接后推算结论。",'',
        '## 摘要','',
        '新类仅有少量标注时，CNN从基础类学到的判别方向不一定适合新类；支持集类中心和方差估计又容易受单张图像或背景干扰。本实验采用基础类监督预训练、完整二阶MAML、均值方差收缩与无标签查询原型更新进行对比，并进一步检验训练类中心化、协方差白化、图传播和同图多视图策略。',
        f"采用互不重叠的64/16/20训练/验证/测试类别，训练{protocol['training_images']:,}张、验证{protocol['validation_images']:,}张、测试{protocol['test_images']:,}张。最终5-way 1/5-shot各{protocol['test_episodes_per_shot']}个共同任务，每类15张query；支持和查询图像互斥，查询标签只在预测后评分。轻量Baseline为{pct(rows[('baseline',1)]['accuracy'])}/{pct(rows[('baseline',5)]['accuracy'])}，完整校准为{pct(rows[('full',1)]['accuracy'])}/{pct(rows[('full',5)]['accuracy'])}，二阶MAML为{pct(rows[('maml',1)]['accuracy'])}/{pct(rows[('maml',5)]['accuracy'])}。",]
    if ('wideview',1) in rows:
        lines.append(f"较宽骨干双视图方案为{pct(rows[('wideview',1)]['accuracy'])}/{pct(rows[('wideview',5)]['accuracy'])}。它相对同一较宽权重没有新增训练，但每图编码两次；相对轻量骨干则包含此前更大容量与更多训练预算，不应把全部差异归因于校准。所有置信区间、逐任务预测、负结果与计算代价保留，不承诺95%准确率、零误判或标准论文模型上限。")
    if audit.get('continued'):
        lines.append(f"同容量续训双视图进一步达到{pct(rows[('continued',1)]['accuracy'])}/{pct(rows[('continued',5)]['accuracy'])}。容量仍为509,856参数，但增加16轮训练，验证选第12轮，沿用已有分类设置。该收益归于额外训练研究，不冒充同预算校准创新。")
    lines+=['','关键词：小样本学习；ResNet12；MAML；分布校准；原型更新；转导推理；可复核评测','',
        '## 1 研究问题与学习范式','',
        '医疗影像、工业缺陷和小众物种识别中，新类标注可能昂贵；这构成本实验的动机，而不是这些场景已完成实际验证的声明。mini-ImageNet用于方法研究，不能由其准确率推出临床安全性、工业良率或企业股权风险识别能力。',
        '数据不足有两层影响：表征层面，基础类训练和新类任务的可区分方向可能不同；统计层面，极少支持样本会导致类均值偏差，1-shot无法直接估计类内方差。过拟合不是只能通过更强模型解决，额外容量也可能增加训练和部署成本。因此分别控制模型训练、统计迁移和任务内适应。']
    table(lines,['范式','核心机制','本实验实现与边界'],[
        ['监督表征＋度量分类','冻结编码器，支持集均值作类别原型','Baseline不是episodic训练的ProtoNet复现；参考思想见[2]'],
        ['优化式元学习','训练可快速适应的初始化','全网络二阶MAML；测试只用支持标签[1]'],
        ['统计迁移校准','利用训练类统计收缩新类估计','本实现为轻量变体，不完整复现分布校准论文[3]'],
        ['转导任务推理','联合使用当前批次无标签query','原型更新、图传播和均衡先验，信息条件强于单张归纳预测[5]']])
    lines+=['## 2 数据划分、任务构造与公平比较','',
        '按mini-ImageNet既定类别划分加载64训练类、16验证类、20测试类，每类600张RGB图像，统一84×84。训练类用于优化权重、估计先验和协方差；验证类用于检查点和超参数选择；测试类只能在配置锁定后生成最终比较。类名集合做隔离检查。',
        '每个任务随机选择5类，每类从同一池抽取1或5张支持图像，再抽取不同的15张query。query顺序打乱，分类函数不接收query_labels参数。所有方法使用相同任务种子、支持ID、query ID和评分标签；方法间配对差在逐任务层面计算，而非把两个独立均值区间相减。',
        '示例网页使用固定的自包含未见类图像池进行真实模型编码，便于离线演示；该单任务不能替代完整测试池上600任务的统计。相同shot和种子只说明抽样可复现，还需检查图片池和发布版本指纹。']
    table(lines,['项目','当前协议'],[
        ['图像池','train 64×600 / validation 16×600 / test 20×600'],['输入尺寸','84×84 RGB；ImageNet均值/标准差归一化'],
        ['任务','5-way；1-shot和5-shot；每类15张query'],['共同任务数',f"每种shot {protocol['test_episodes_per_shot']}"],
        ['抽样种子',protocol['seed']],['隔离约束','类集合隔离；任务支持/query互斥；query标签预测后评分'],
        ['模型选择','训练/验证完成后写入锁与SHA256；不依据测试成绩重选']])
    lines+=['已公开测试基准在多轮研究中重复使用，因此本报告属于后续优化，不是从未接触过测试分布的外部验证。验证搜索与复核的图像池互斥，但仍来自同16验证类，且复核参与最终选择；不能当作无偏独立泛化估计。','',
        '## 3 网络、训练目标与元学习','',
        f"轻量ResNet12有四个残差块，每块三层3×3卷积，采用GroupNorm(4)、LeakyReLU、2×2最大池化，末端全局平均池化。1×1捷径投影不计入十二层卷积命名。通道为{model['widths']}，特征维度{model['embedding_dim']}，64类训练头的Baseline为{model['parameters_baseline']:,}参数，5类MAML头为{model['parameters_maml']:,}参数。GroupNorm不混用query批统计。",
        f"v2监督暖启动训练{len(core['training']['warm_ce']['epochs'])}轮，标签平滑0.05、AdamW权重衰减0.0005，按验证1/5-shot均值选择第{core['training']['warm_ce']['selected_epoch']}轮。从同一暖启动构建Baseline与对齐分支，各训练3轮，使用相同样本、顺序、增强和两次视图的交叉熵；唯一差别为对齐系数λ=0或0.5。",
        '目标为 L = 0.5×[CE(fθ(a1(x)),y)+CE(fθ(a2(x)),y)] + λ×Lalign。Lalign为两次增强特征的批均值平方差与批总体方差平方差，分别按维度平均。对齐对象是同一训练图像的增强视图，不读取新类标签，也不等价于已经消除真实跨域分布差异。',
        f"MAML从Baseline编码器初始化并重新设置5类分类头，元训练{core['training']['maml']['tasks']}任务、meta-batch {core['training']['maml']['meta_batch_size']}；元训练任务为2-shot，每类5张query，内循环1步、学习率0.1，外循环学习率0.0005。支持更新保留梯度图create_graph=True，外循环通过更新过程回传，区分真实MAML与FOMAML。验证选中{core['training']['maml']['selected_tasks']}任务检查点。",
        'MAML形式：θ′τ=θ−α∇θ Lsupport(θ)，外循环最小化 Στ Lquery(θ′τ)。元训练query标签用于外循环是训练阶段的正常监督；测试时严格不利用query标签。测试适应不需要保存二阶图，但仍更新全部骨干与分类头，最终由适应后的分类头输出，不偷换成最近原型。']
    table(lines,['测试任务','MAML支持集适应步数','学习率'],[[f'{n}-shot',core['maml_adaptation'][str(n)]['steps'],core['maml_adaptation'][str(n)]['learning_rate']] for n in (1,5)])
    lines+=['MAML额外训练预算和适应机制不同，不能称所有方法训练预算严格相同；当前负结果也不能外推为MAML方法总体更差。','',
        '## 4 均值方差校准与原型自适应','',
        '先将特征L2归一化。v2按验证配置减去只由基础训练类估计的中心，并按固定幂变换再次归一化；本轮幂为1。新类c的支持原型pc是支持特征均值。选择与pc最近的K个基础类均值，其平均为mc，邻域方差νc同时包含训练类内部方差和邻域类中心间差异。邻域搜索不读取query标签。',
        '均值收缩：p̃c=(1−a)pc+a mc。对角方差：先以支持样本自由度nc−1和训练先验强度κ加权，v̂c=[(nc−1)sc²+κνc]/[(nc−1)+κ]，随后使用训练方差下界，并用验证选择的混合ρ与共享尺度混合。1-shot自由度为0，不伪造新类经验方差。',
        '分类logit为−mean_d[(zq,d−pc,d)²/vc,d]/T。这是对角Mahalanobis类距离而非完整生成密度声明；没有声称高斯生成补样或完整协方差分类器。关闭方差时共享正尺度不改变欧氏最近原型的argmax。',
        '原型更新：p_c^(t+1)=[nc p̃c+βΣq wqc zq]/[nc+βΣq wqc]。每次更新都以初始支持原型为锚，类别分数来自softmax或均衡Sinkhorn分配。均衡约束利用每类15张query的已知协议，并不读隐藏标签；真实不均衡业务不能直接套用。分数并非经过独立概率校准。']
    table(lines,['shot','均值a','方差混合ρ','原型β','更新次数','温度T','均衡'],[[n,core['calibration'][str(n)]['mean_weight'],core['calibration'][str(n)]['variance_blend'],core['calibration'][str(n)]['prototype_weight'],core['calibration'][str(n)]['prototype_steps'],core['calibration'][str(n)]['temperature'],'是'] for n in (1,5)])
    lines+=['## 5 后续白化、图传播与多视图方案','',
        '图传播：从训练类估计固定均值和协方差，用带特征值下界的白化变换处理支持/query；任务内构造非负余弦相似度近邻图，只用支持标签作为种子求解传播，与均衡原型分数融合。二维PCA仅解释视图，不是完整分类决策。',
        '训练类内白化将总体协方差与按训练标签去除各类中心后的类内协方差混合，以抑制基础类内部变化；全部统计来自训练类。图像原图/水平翻转来自同一图片，不增加带标签图片。轻量双视图选择特征拼接；较宽双视图选择归一化特征平均，不能混为同一实现。']
    if audit['wide']:
        wide=audit['wide'];lines.append(f"较宽骨干通道{wide['model']['widths']}，{wide['model']['parameters']:,}参数。由自训Baseline进行保持初始函数的通道复制并加小扰动，额外训练{wide['training']['epochs_completed']}轮CE＋0.25度量损失，验证选中第{wide['training']['selected_epoch']}轮。没有外部预训练权重。模型容量和训练预算均不同，归类为独立研究而非纯推理消融。")
    if audit['wideview']:
        cfg=json.loads((artifacts/'wideview_configuration.json').read_text(encoding='utf-8'))
        lines+=['较宽双视图复用同一wide.pt，不再训练；mean/concat、白化指数和类内混合共18候选先登记，验证160搜索任务、400复核任务选前四再按配对区间下界选择。最终两种shot均为mean、指数0.5、类内混合0.5。1-shot保留图权重0.5，5-shot权重0，即关闭图传播。不能因为方案名字包含图传播就把已关闭模块归为收益来源。',
            '此前另18个轻量／较宽融合候选在验证未有充分提升证据，1-shot选回原方案，5-shot配对区间跨零；它们不进入测试和正式推理，但记录保留。正研究与负研究均由原始验证分数重算搜索/复核规则，不仅相信成功标志。']
    if audit.get('continued'):
        lines+=['同容量续训从原wide.pt出发，固定16轮AdamW新优化器（学习率0.0002余弦至0.00001）、标签平滑0.05与0.25度量损失，输入增强及网络宽度不变。基础训练类逐batch流式读取，第0轮也可被选择。训练阶段验证最大值选第12轮，额外预算按实际消耗16轮披露，不只报12轮。',
            '完成训练后按预登记规则使用全验证类600个配对任务/shot、mean/0.5白化/0.5类内混合及旧图参数，不再搜索超参数。相对已发布较宽双视图的验证配对差为+0.4378±0.2723pp和+0.3667±0.1558pp，两个下界为正后写锁，才进入一次原1200测试任务。验证预测、选择锁、训练计划与epoch选择全部独立重算。']
    lines+=['','## 6 主要结果与配对改善','']
    table(lines,['方案','1-shot均值±95%半宽','5-shot均值±95%半宽'],[[LABELS.get(m,m),estimate(rows[(m,1)]),estimate(rows[(m,5)])] for m in methods])
    table(lines,['方法vs同版Baseline','1-shot配对差±95%半宽','5-shot配对差±95%半宽'],[[LABELS.get(m,m),*[f"{rows[(m,n)]['paired_delta']*100:+.2f} ± {rows[(m,n)]['paired_ci95']*100:.2f}pp" for n in (1,5)]] for m in methods if m!='baseline'])
    lines+=['95%区间按任务准确率的均值±1.96×样本标准差/√N计算，配对差区间从同任务差值计算。每任务query出现次数相同，不把重复图片当作同等数量独立样本；这些区间条件于单次训练权重，不覆盖训练种子变化。',
        '完整方案相对Baseline在本协议下有配对提升；MAML仍低于Baseline，应解释为当前训练、初始化和适应选型下的结果，不是方法普遍优劣。宽模型与轻量模型的比较也不是同预算结论。']
    if audit['wideview']:
        table(lines,['较宽双视图vs参考','1-shot差值±半宽','5-shot差值±半宽'],[[reference,*[f"{rows[('wideview',n)][mean]*100:+.4f} ± {rows[('wideview',n)][ci]*100:.4f}pp" for n in (1,5)]] for reference,mean,ci in [('轻量双视图','paired_vs_multiview','paired_vs_multiview_ci95'),('同权重较宽单视图','paired_vs_wide','paired_vs_wide_ci95')]])
        lines.append('较宽双视图相对较宽单视图两种shot区间均高于零，支持本次固定协议的改善。没有新增训练的比较仍增加每图一次编码；不能同时无条件宣称更准、更快、更省算力。')
    if audit.get('continued'):
        table(lines,['同容量续训vs原较宽双视图','1-shot','5-shot'],[['均值±95%半宽',estimate(rows[('continued',1)]),estimate(rows[('continued',5)])]])
        table(lines,['设置','配对差±95%半宽','配对区间'],[[f'{n}-shot',f"{rows[('continued',n)]['paired_vs_wideview']*100:+.4f} ± {rows[('continued',n)]['paired_vs_wideview_ci95']*100:.4f}pp",
            f"[{(rows[('continued',n)]['paired_vs_wideview']-rows[('continued',n)]['paired_vs_wideview_ci95'])*100:.4f}, {(rows[('continued',n)]['paired_vs_wideview']+rows[('continued',n)]['paired_vs_wideview_ci95'])*100:.4f}]pp"] for n in (1,5)])
        lines.append('两种shot的测试配对区间均高于零，但只针对这次模型与公开基准；未扩展为新算法、普遍收敛或多种子显著性保证。原wide.pt和旧较宽双视图结果保留，新权重单列continued.pt。')
    lines+=['','## 7 基础模块消融与负结果','']
    ablations=['alignment_only','transform_only','mean_only','variance_only','prototype_only','mean_variance','full','no_alignment_full','no_transform_full','unbalanced_full']
    table(lines,['模块/条件','1-shot','5-shot'],[[LABELS.get(m,m),estimate(rows[(m,1)]),estimate(rows[(m,5)])] for m in ablations])
    # Exact module-to-module pairs are recomputed from original frozen task scores.
    episodes=json.loads((artifacts/'episodes.json').read_text(encoding='utf-8'))['episodes']
    contributions=[('分布对齐（完整条件）','full','no_alignment_full'),('均值校准（相对仅中心化）','mean_only','transform_only'),('添加方差（已有均值）','mean_variance','mean_only'),('原型更新＋均衡转导（已有均值/方差）','full','mean_variance'),('原型以外完整组合','full','prototype_only'),('均衡查询先验','full','unbalanced_full')]
    paired_rows=[];pairs={}
    for title,left,right in contributions:
        cells=[title]
        for n in (1,5):
            vals=[r['methods'][left]['accuracy']-r['methods'][right]['accuracy'] for r in episodes if r['shots']==n]
            mean,ci=delta(vals);pairs[f'{left}/{right}/{n}']={'mean':mean,'ci95':ci,'episodes':len(vals)}
            cells.append(f"{mean*100:+.2f} ± {ci*100:.2f}pp")
        paired_rows.append(cells)
    table(lines,['逐任务直接模块差','1-shot配对差','5-shot配对差'],paired_rows)
    lines+=['以上模块间差值由冻结逐任务记录直接重算，仅作报告诊断，不用于重新选择测试参数；网站提供“模块直接对照”和“相对Baseline”两种口径，必须按表头解释，不能混用显著性。原型更新的这组对照同时改变均衡分配及归纳/转导信息条件，不能全部归于原型单一模块。区间均为逐项95%区间，未作多重比较校正。',
        '完整方案去除训练对齐后两种shot均值相同，不能把该正则项宣称为已验证有效。完整方案与仅原型更新接近，不是叠加模块越多越好。均值和方差先验可能因基础类与新类不匹配导致负迁移；均衡约束的效果也依赖任务类别平衡。','',
        '## 8 分数可靠性与任务波动','']
    diagnosis=audit['diagnostics']
    if diagnosis:
        table(lines,['轻量双视图诊断','1-shot','5-shot'],[
            ['任务P05/P50/P95',*[f"{pct(r['task_p05'])} / {pct(r['task_median'])} / {pct(r['task_p95'])}" for r in diagnosis['rows']]],
            ['vs单视图图传播 改善/持平/下降',*[f"{r['paired_wins']}/{r['paired_ties']}/{r['paired_losses']}" for r in diagnosis['rows']]],
            ['ECE10（百分点）',*[f"{r['ece10']*100:.2f}" for r in diagnosis['rows']]],
            ['Brier（五类平方误差和）',*[f"{r['brier_score']:.4f}" for r in diagnosis['rows']]],
            ['负对数似然',*[f"{r['negative_log_likelihood']:.4f}" for r in diagnosis['rows']]]])
    lines+=['本节是明确标注的轻量双视图固定预测诊断，不将其数值冒充为较宽双视图概率表现。P05/P95描述任务分位而非均值区间。可靠性图按十等宽分箱比较平均模型分数和实际正确率，空分箱不补点；ECE是描述性诊断，不用于反向调测试温度，也不保证未来单个样本的正确概率。','',
        '## 9 本地CPU部署与交互实现','',
        '系统以本机FastAPI加载权重、训练先验与固定演示池。Baseline、MAML、校准均在现场编码，不播放预存结果。推理串行锁避免共享网络参数在MAML functional_call时交错。请求只允许方法、shot、种子和校准枚举，不接受路径或query标签。',
        '评测页面拆为性能、消融、可靠性、端侧耗时、历史来源；另有方法协议、现场推理及要求交付。独立路由支持直接访问/刷新/前进后退，站内切换不丢已完成任务。交互图与精确表格并存，不隐藏负结果。',
        '后端逐任务核验模型、样本ID、预测、区间及选型锁；前端将JSON视为unknown并验证结构及统计自洽，阻止坏响应造成白屏或伪造图表。异常刷新保留上次有效结果并明确提示。取消等待只终止浏览器等待及尚未发起的方法，不宣称服务器已强制停止；迟到响应不覆盖新任务。',
        '特征图使用同一PCA基、两轴相同尺度并披露解释方差。统计校准不伪造样本特征移动；MAML前后特征来自真实网络适应。高维决策不能仅由二维投影理解。']
    if audit['edge_benchmark']:
        table(lines,['方法','shot','编码＋分类中位ms','P95 ms','重复次数'],[[LABELS.get(r['method'],r['method']),r['shots'],f"{r['encoding_and_classification_median_ms']:.1f}",f"{r['encoding_and_classification_p95_ms']:.1f}",r['repetitions']] for r in audit['edge_benchmark']['rows']])
        lines.append('全方法同口径计时，每轮重新编码；排除权重加载、指纹校验、HTTP与缩略图渲染，不混排缓存特征计时。该轮系统负载/环境只代表当时测量，旧耗时另存历史。端侧验证是Windows本机CPU，不声称手机、ARM、NPU实测或ONNX/量化部署已完成。')
    lines+=['','## 10 局限、后续研究与课程要求','',
        '单训练种子不足以评估重新训练波动；重复使用公开基准和同组验证类会带来选择偏差。所有结果局限于当前模型规模与均衡5-way任务，无法保证跨域、不均衡、分布外数据或真实业务。图传播和原型更新利用整批无标签query，比归纳任务拥有更多信息。',
        '后续优先补多训练种子、独立数据集或领域验证、非均衡query适用性，以及教师明确指定的端侧硬件。追加训练必须作为独立预算研究，训练阶段不能根据测试反馈选择参数；仍在运行的候选不写入本报告。',
        '原要求的学习范式、特征漂移分析、ResNet12、CE＋分布对齐、均值方差校准、原型自适应、Baseline/MAML比较、mini-ImageNet、消融、限制与本机CPU部署都有实现与证据索引。1–5人组队、关联成员、课程项目和管理员提交需在课程平台实际确认，不能由本地代码证明。若另行限定特定硬件，本地CPU不替代该硬件要求。','',
        '## 11 复现、材料版本与AI使用','',
        '源码位置：experiments/lab04；完整原始结果：outputs/lab04；要求矩阵：experiments/lab04/REQUIREMENTS_AUDIT.md。核心results/episodes与graph/multiview/wide/wideview扩展分别锁定。旧v1目录、旧报告/PPT/视频保留，不用新数值覆盖旧实验。',
        '只读核验命令：python -B -X utf8 main.py lab04 verify。现场示例：python -B -X utf8 main.py lab04 demo --method calibrated --calibration wideview --shots 1。网站：python main.py web --port 8765。完整训练、选型与复现细节见实验四README，各依赖和缓存需按原计划准备。',
        '分数来自真实模型与冻结逐任务记录；报告表格由审计代码生成，本文模块间配对差另有report_evidence.json记录。AI用于代码、界面、实验流程和文档辅助，未替代实际推理或捏造准确率。','',
        '## 参考文献与数据来源','',
        '[1] Finn C, Abbeel P, Levine S. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML, PMLR 70:1126–1135, 2017. https://proceedings.mlr.press/v70/finn17a.html',
        '[2] Snell J, Swersky K, Zemel R. Prototypical Networks for Few-shot Learning. NeurIPS 30, 2017. https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html',
        '[3] Yang S, Liu L, Xu M. Free Lunch for Few-shot Learning: Distribution Calibration. ICLR, 2021. https://openreview.net/forum?id=JWOiYxMG92s',
        '[4] Ravi S, Larochelle H. Optimization as a Model for Few-Shot Learning. ICLR, 2017. https://openreview.net/pdf?id=rJY0-Kcll',
        '[5] Liu Y, Lee J, Park M, et al. Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning. ICLR, 2019. https://arxiv.org/abs/1805.10002',
        '以上为原报告保留的参考条目，用于方法背景，不声称复现对应论文完整流程或公布分数。mini-ImageNet文件组织参考：https://github.com/yaoyao-liu/mini-imagenet-tools 。']
    report.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    evidence={'generated_utc':datetime.now(timezone.utc).isoformat(),'artifact_revision':audit['artifact_revision'],'paired_module_differences':pairs,
              'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.iterdir() if p.is_file() and p.suffix in ('.json','.npz','.pt')},
              'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'unpublished_candidates_included':False}
    (output/'report_evidence.json').write_bytes((json.dumps(evidence,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    if office:export_office(report,report_title='小样本分类与特征校准：当前整合报告',footer_text='实验四 · 当前整合报告 / 2026-11-25')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifacts',default=str(SOURCE/'outputs/lab04'))
    parser.add_argument('--output',default=str(ROOT/'deliverables/lab4/当前版本'));parser.add_argument('--markdown-only',action='store_true')
    args=parser.parse_args();print(build(args.artifacts,args.output,not args.markdown_only))


if __name__=='__main__':main()
