"""Build editable report, slides and speaker notes exclusively from saved measured results."""
from __future__ import annotations
import json,statistics,math
from pathlib import Path
from .dataset import ROOT
OUT=ROOT/'outputs/lab03';DEL=ROOT/'deliverables/lab03'

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def pct(n):return f'{100*n:.2f}%'

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'figure.dpi':140,'savefig.dpi':190})
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    rows=read('summary.json');methods=['bfs','astar_fixed','astar_ga','mcts','hybrid','hybrid_no_expert']
    labels=['BFS','固定 A*','GA A*','MCTS','混合','无专家']
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    for ax,split,title in zip(axes,['test','ood'],['独立测试图','分布变化图']):
        rs=[next(r for r in rows if r['split']==split and r['budget']==8 and r['method']==m) for m in methods]
        bars=ax.bar(range(len(rs)),[r['success_rate']*100 for r in rs],color=['#B5A18C','#A7512C','#286A72','#9B8155','#455B68','#C3B6A5'])
        ax.set_xticks(range(len(rs)),labels,rotation=15);ax.set_ylim(0,112);ax.set_ylabel('有效答案率 / %');ax.set_title(title+' · 扩展预算8')
        ax.spines[['top','right']].set_visible(False)
        for b,r in zip(bars,rs):ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{r["success_rate"]*100:.1f}',ha='center',fontsize=9)
    fig.savefig(folder/'01_success.png',facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    rs=[next(r for r in rows if r['split']=='test' and r['budget']==8 and r['method']==m) for m in methods]
    xs=list(range(len(rs)));ax.bar([x-.18 for x in xs],[r['mean_expanded'] for r in rs],width=.36,label='树扩展',color='#9A4728')
    ax.bar([x+.18 for x in xs],[r['mean_total_work'] for r in rs],width=.36,label='含rollout总工作量',color='#426D80')
    ax.set_xticks(xs,labels);ax.set_ylabel('每题平均操作数');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'02_work.png',facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    for seed,ga in read('ga_history.json').items():ax.plot([r['generation'] for r in ga['history']],[r['fitness'] for r in ga['history']],marker='o',label='种子 '+seed)
    ax.set(xlabel='代数',ylabel='验证集适应度',title='仅在验证图优化；测试集不参与选择');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'03_ga.png',facecolor='white');plt.close(fig)
    example=read('search_examples.json')[0];result=example['result'];nodes={n['id']:n['name'] for n in example['graph']['nodes']};unique={}
    for s in result['trace']:unique[tuple(s['edge_ids'])]=s
    vals=list(unique.values())[:38];levels={}
    for s in vals:levels.setdefault(s['depth'],[]).append(s)
    pos={tuple(s['edge_ids']):(d,-i) for d,states in levels.items() for i,s in enumerate(states)}
    fig,ax=plt.subplots(figsize=(11,max(3.5,max(map(len,levels.values()))*.4)),layout='constrained')
    for s in vals:
        k=tuple(s['edge_ids']);x,y=pos[k]
        if k[:-1] in pos and k:xp,yp=pos[k[:-1]];ax.plot([xp,x],[yp,y],color='#D9CFC3',zorder=0)
        ax.text(x,y,nodes[s['node']][:9],ha='center',va='center',fontsize=8,bbox={'boxstyle':'round,pad=.4','fc':'#F2DCCF' if s['status']=='solution' else '#F8F5F0','ec':'#A7512C' if s['status']=='solution' else '#C4B6A4'})
    ax.axis('off');ax.set_xlim(-.4,4.4);fig.savefig(folder/'04_search_tree.png',facecolor='white');plt.close(fig)
    return folder


def build_report():
    from docx import Document
    from docx.shared import Pt,Cm
    from docx.oxml.ns import qn
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DEL.mkdir(parents=True,exist_ok=True);fig=figures();summary=read('summary.json');manifest=read('manifest.json');ga=read('ga_history.json');learning=read('learning_metrics.json');llm=read('llm_summary.json');audit=read('llm_audit.json');expl=read('llm_explanations.json');paired=read('paired_effects.json');brief=json.loads((ROOT/'data/lab03/brief.json').read_text(encoding='utf-8'))
    metric=lambda split,m:next(r for r in summary if r['split']==split and r['budget']==8 and r['method']==m)
    base=metric('test','bfs');fixed=metric('test','astar_fixed');opt=metric('test','astar_ga');mix=metric('test','hybrid');mcts=metric('test','mcts');noexpert=metric('test','hybrid_no_expert')
    llmb,llma=llm['results'];doc=Document();sec=doc.sections[0];sec.top_margin=sec.bottom_margin=Cm(2);sec.left_margin=sec.right_margin=Cm(2.3)
    for name in ['Normal','Title','Heading 1','Heading 2']:
        st=doc.styles[name];st.font.name='Microsoft YaHei';st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'微软雅黑')
    doc.styles['Normal'].font.size=Pt(10.5);doc.styles['Normal'].paragraph_format.line_spacing=1.35
    doc.core_properties.title=brief['title'];doc.core_properties.author='第31组'
    doc.add_heading('人工智能导论 · 课堂报告三',0);doc.add_heading('大模型辅助的企业股权知识图谱\n多跳路径搜索与智能优化',1)
    doc.add_paragraph('A* / MCTS 搜索 · 遗传算法 · 路径价值学习 · 证据专家规则')
    doc.add_paragraph('华南农业大学 · 数学与信息学院、软件学院\n软件工程5班 · 第31组\n组长：梁镇贤\n组员：陈宗波、温文辉、陈宇冲、刘亦达\n2026年11月8日')
    doc.add_paragraph('本报告由本次代码真实运行结果生成。扩展图与标签均为教学构造；LLM使用本机Qwen2.5-0.5B-Instruct真实推断，不以模拟结果冒充模型调用。')
    def para(t):doc.add_paragraph(t)
    def table(headers,rows):
        t=doc.add_table(rows=1,cols=len(headers));t.style='Light Shading Accent 1'
        for c,v in zip(t.rows[0].cells,headers):c.text=str(v)
        for row in rows:
            for c,v in zip(t.add_row().cells,row):c.text=str(v)
        return t
    def picture(file,caption):doc.add_picture(str(fig/file),width=Cm(16));p=doc.add_paragraph(caption);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading('摘要',1)
    para(f'本项目沿用企业股权场景，将2—4跳关系问答建模为带证据约束的有限状态空间搜索，独立实现BFS、A*和UCT/MCTS。设计固定启发式，并仅利用验证图优化其参数；用训练图成功/失败路径前缀训练逻辑回归价值分类器；引入本地指令模型作行动合法性辅助、路径评分和搜索后解释。405题划分为162道训练题、81道验证题、108道测试题和54道分布变化题，实体在不同图与划分间完全隔离。三种子、三预算、八方法共运行3,888次。预算8时，BFS有效答案率为{pct(base["success_rate"])}，固定A*为{pct(fixed["success_rate"])}，GA A*为{pct(opt["success_rate"])}。分布变化下GA可能退化；MCTS的额外模拟开销需单独计入。9题LLM配对实测中，加入模型前后答案率分别为{pct(llmb["success_rate"])}和{pct(llma["success_rate"])}，不据此宣称总体增益。')
    para('关键词：知识图谱；启发式搜索；蒙特卡洛树搜索；遗传算法；路径分类；大模型幻觉；证据核验')
    doc.add_heading('1 场景与任务定义',1)
    para('金融尽调与股权关系核查经常需要穿过多层持股、担保和交易记录，寻找能解释关联关系的证据链。ICIJ Offshore Leaks公开数据库展示了人员、企业与中介关系组织和追踪的行业形式[6]。本项目仅借鉴关系追踪流程，不导入其真实人物数据，不声称真实部署或法律判断。')
    para('与实验一、二衔接：共享案例直接读取原CSV，包含16个实体及24条本任务支持的持股/担保/交易边；亲属关系未纳入此次关系词表。扩展基准为独立生成的企业图，用于让搜索分支、干扰与证据问题可控。问答采用明确关系序列与终点属性，不宣称实现开放域自然语言解析。')
    para('问题示例：从王建国出发，连续经过三层持股关系，能到达哪家上市企业？搜索得到P1→C1→C2→C3及每条原始边。该结果仅表示关系路径；路径乘积18.2%也不能代替控制结论或把两条路径总权益34.2%混为一谈。')
    table(['建模项','精确定义'],[
      ['状态 s','(当前节点v，已走关系数d，已访问节点序列P，边序列E)。保留路径可避免循环并保留解释。'],
      ['动作 a','从v出发、关系类型等于问题第d+1项、且不重复访问节点的图中实际出边。'],
      ['转移','追加现存关系及目标节点；每条边代价1。模型不得创建动作或添加实体。'],
      ['目标','恰好走完2—4跳关系序列，终点满足公司、行业与上市属性。返回一条有效证据路径。'],
      ['终止','找到目标 / 候选耗尽 / 扩展预算耗尽。后两者分别记录，不混称答案为假。'],
      ['专家证据','verified=True、active=True且confidence≥0.55才允许通过。阈值为预先规定的教学假设。']])
    doc.add_heading('2 搜索算法与启发式',1)
    para('BFS使用FIFO队列逐层扩展；A*使用优先队列按f(s)=g(s)+h(s)排序。下界版h0=剩余跳数，在单位边代价和固定关系序列任务上是可采纳的，作为正确性对照。完整路径状态而非单一节点用于去重，避免不同前缀被不当合并。')
    para('参数化启发式：hθ(s)=(L−d)+θ1(1−主题匹配)+θ2(1−后继可选性)+θ3(1−路径最低置信度)+θ4(1−终点类型匹配)+θ5·死路提示+λ(1−pML)。在目标状态h=0。固定参数θ=[1.7,0.5,0.6,0.2,1.5]；GA只优化前5项，混合版λ=1.5为事先固定配置。LLM版额外加2(1−pLLM)，未评分边使用0.5。')
    para('GA与学习启发式可能高估剩余代价，不声明A*的最优性保证。因为本任务要求固定跳数，找到的有效解都满足给定L；实验关注有限预算成功率和搜索成本，不用该基准证明变长最短路径最优性。')
    para('MCTS独立实现选择、扩展、模拟与回传。选择分数为Q/N+√2·sqrt(ln(Nparent)/N)；未访问子节点优先。模拟在合法出边上随机采样，到终点给1奖励，否则用中性叶估值。为保证纯MCTS对照不偷用训练模型，其基线不加载路径分类器。每个树节点扩展及每次rollout转移分别计数。')
    doc.add_heading('3 遗传算法与路径价值学习',1)
    para('GA使用种群14、10代、精英保留2、锦标赛选择3、算术交叉、高斯变异(σ=0.65、概率0.3)，权重限制[0,5]。每个种子独立优化，适应度J=验证图有效答案率−0.12×平均扩展数/12；验证预算12。固定参数作为初始个体，最优验证适应度不会因代际替换下降。测试结果不参与个体选择。')
    picture('03_ga.png','图1 三个种子的验证适应度曲线；上升不等于泛化改善。')
    table(['种子','初代J','末代J','验证题数','最终θ'],[[s,f'{v["history"][0]["fitness"]:.4f}',f'{v["history"][-1]["fitness"]:.4f}',v['validation_queries'],', '.join(f'{w:.3f}' for w in v['weights'][:5])] for s,v in ga.items()])
    para('价值分类器使用6维逻辑回归，正标签表示训练图中该路径前缀存在一条有效目标延伸。独立穷举仅在离线标签生成和评估中使用，搜索内核不调用oracle。训练采用类别平衡权重、L2正则0.012、学习率0.12、700次梯度迭代，配置在测试前固定。输入只含局部特征，没有节点ID、目标答案、到答案的真实距离。')
    table(['种子','测试前缀数','Precision','Recall','F1','Brier'],[[r['seed'],r['samples'],f'{r["precision"]:.4f}',f'{r["recall"]:.4f}',f'{r["f1"]:.4f}',f'{r["brier"]:.4f}'] for r in learning])
    doc.add_heading('4 LLM辅助与权限边界',1)
    para('本机使用Qwen2.5-0.5B-Instruct，固定revision 7ae557604adf67be50417f59c2c2f167def9a775[5]，以Transformers在CPU运行。模型文件保存在用户缓存，不随项目打包。另实现Responses API结构化远程评分适配器，凭据从环境变量读取；本报告不把未经真实调用的远程模式算作实验结果。')
    para('模型对候选边执行两次受限下一token评估：A/B/C表示有希望/不确定/较差，pvalue=P(A)+0.5P(B)；Y/N表示证据是否符合要求。模型认为不合法时，评分缩至原值的0.25，但不直接删边；确定性专家规则仍拥有最终裁决权。这样即使模型错判，也不能覆盖明确无效的边。')
    para('LLM子集固定为每个种子的第一张测试图、前3道题，共9题且覆盖2/3/4跳。只评分起点两层内最多8条现存边，未评分边采用中性分数。预评分开销单独记录为scored_edges和LLM毫秒数，不混入核心搜索计时。选择顺序在模型结果出现前固定，不挑选有利案例。')
    para(f'真实评分共{llm.get("legal_checks",0)}条边；合法性判断与确定性规则一致率为{pct(llm.get("legal_check_accuracy",0))}。模型前向推断累计约{llm["total_inference_ms"]/1000:.1f}秒，不含加载与搜索后文本解释。CPU小模型开销明显大于本地符号搜索，只有更复杂、信息不完整的任务才可能抵消此成本，当前结果不支持生产效率优势。')
    table(['子集方法','题数','有效答案率','平均扩展','搜索毫秒/题'],[[r['label'],r['n'],pct(r['success_rate']),f'{r["mean_expanded"]:.3f}',f'{r["mean_ms"]:.3f}'] for r in llm['results']])
    para('路径解释在程序搜索成功后才生成；模型收到已核验路径，不参与答案选择。保存原始输出并检查是否遗漏路径实体。解释文本仍可能出现改名、漏名或含糊表述，界面将其标为模型原文并保留证据对照。这与LATS/Think-on-Graph关于模型参与搜索、利用外部图证据的思路相关[3–4]，本作业是受限教学实现，不声称复现论文全量基准。')
    for e in expl:
        para(f'解释案例 {e["query_id"]}：{e["text"]}');para('实体一致性检查：'+('全部路径实体可定位。' if e.get('grounded') else '未完整出现的路径实体：'+str(e.get('missing_path_entities'))+'。不能把这段文本当作新的证据。'))
    doc.add_heading('5 实验设计与指标口径',1)
    table(['划分','图数','问题数','用途'],[['训练',18,162,'分类器学习；不作GA适应度'],['验证',9,81,'每种子27题选择GA参数'],['测试',12,108,'固定参数后评估'],['分布变化',6,54,'节点数48→64、额外干扰边、主题同配性下降']])
    para('随机种子为7、19、43。每图9题，均为2/3/4跳各3题；不同图使用不同实体ID。生成时保留至少一条有效证据路径，但向搜索引擎只提供图与公开问题谓词，不提供被植入路径。问题都可答，因而无解检出能力另以单元测试验证，不据此宣称开放世界召回率。')
    para('八种方法为BFS、下界A*、固定A*、GA A*、仅学习A*、MCTS、GA+学习+专家、去掉专家的混合版。预算4/8/16统一限制树扩展数；MCTS额外模拟步数另计，因此只看扩展数不构成公平总成本结论。')
    table(['指标','定义'],[['有效答案率','返回端点满足问题，且完整边序列存在、顺序正确、无重复节点、证据全部有效。'],['扩展节点数','BFS/A*弹出处理状态数；MCTS树中首次扩展状态数。'],['总工作量','扩展节点 + MCTS rollout边转移；另提供候选边检查次数。'],['推理步数','成功路径的边数，失败记缺失；固定问题跳数应一致。'],['冗余路径占比','已完成候选路径中，重复到达既有终点的路径数/已完成候选路径数。'],['重复状态比例','扩展过程重复(node,depth)次数/总扩展；与冗余完整路径分开。'],['无效答案率','返回了目标但证据路径未通过校验的比例。'],['不确定性','按整张图bootstrap配对差异，避免把同图相关问题当独立样本。']])
    doc.add_heading('6 结果与对比分析',1)
    picture('01_success.png','图2 独立测试与分布变化有效答案率；扩展预算为8。')
    for split,title in [('test','独立测试'),('ood','分布变化')]:
        doc.add_heading(title+'（预算8）',2)
        table(['方法','有效答案率','扩展节点','含模拟总工作量','无效答案率'],[[r['label'],pct(r['success_rate']),f'{r["mean_expanded"]:.2f}',f'{r["mean_total_work"]:.2f}',pct(r['invalid_answer_rate'])] for r in summary if r['split']==split and r['budget']==8])
    para(f'测试图上，固定A*相对BFS改善{(fixed["success_rate"]-base["success_rate"])*100:.2f}个百分点；GA相对固定A*变化{(opt["success_rate"]-fixed["success_rate"])*100:+.2f}个百分点。混合版为{pct(mix["success_rate"])}，不应把多个模块叠加后所有改善都归因于某一个组件。去专家后的无效答案率为{pct(noexpert["invalid_answer_rate"])}，显示证据约束具有独立作用。')
    for r in paired:
        if r['budget']==8 and r['method']=='astar_ga':para(f'{r["split"]}上GA相对固定A*的配对成功率差为{r["success_delta"]*100:+.2f}个百分点，图级bootstrap 95%区间[{r["graph_bootstrap_low"]*100:+.2f}, {r["graph_bootstrap_high"]*100:+.2f}]个百分点。若区间含0，不将观察到的均值差描述为显著优势。')
    picture('02_work.png','图3 MCTS树扩展与包括模拟的总工作量；不能省略模拟成本。')
    para(f'MCTS平均树扩展为{mcts["mean_expanded"]:.2f}，包含模拟后为{mcts["mean_total_work"]:.2f}。其rollout可在较少树节点扩展时发现答案，但每次模拟仍进行实际图遍历。壁钟数据是单次串行测量，只用于本机描述。')
    doc.add_heading('7 搜索树与失败案例',1)
    picture('04_search_tree.png','图4 一次真实混合搜索的路径状态树（节选）；完整轨迹见网页与JSON。')
    for case in read('failure_cases.json')[:4]:
        para(f'案例 {case["query"]["id"]} / {case["method"]}：{case["query"]["text"]}。状态={case["status"]}，独立oracle有{case["oracle_paths"]}条有效路径。{case["diagnosis"]}。')
    para('失败来源包括：低预算在到达正确深度前耗尽；主题同配启发式在跨行业关系图上失效；未经核验的高诱惑路径在无专家消融中被返回；LLM的证据判断或自然语言解释不可靠。没有把失败题删除或仅展示最优种子。')
    stress=read('hallucination_stress.json');para(f'额外受控实验故意把无效边评分设高、有效边评分设低。在{len(stress["cases"])}题中，开启专家规则返回的无效答案为{stress["invalid_answers"]}。该试验是人为注入错误分数，不能写成模型真实幻觉发生率。模型真实错误应从llm_audit与解释原文中单独统计。')
    doc.add_heading('8 创新、自评与局限',1)
    table(['评分维度','本次可核验依据','仍需避免的夸大'],[['完成度','六项要求有代码、网页、实验记录和报告对应；三套搜索内核独立。','未自动完成课程平台提交。'],['行业结合','复用股权CSV，使用持股/担保/交易证据路径，结合尽调核查流程。','教学构造不是实测企业业务数据。'],['优化迭代','固定h→GA→路径学习→专家约束→LLM；消融与OOD检验。','GA验证提升不意味着任何分布都提升。'],['创新度','证据约束优先；LLM评分与程序答案分离；原文解释与图证据对照。','是课程系统设计创新，不是首次提出新搜索算法。']])
    para('建议以B档90分为有证据的自评目标，由组员在实际演示后确认。按所给通知，90分及以上自评可进入争优演示；85/80档可报名并经教师核实。A档95分和额外奖励属于教师评价，不在报告中预先保证。进一步争取高档评价应扩展真实授权数据、多模型和更大问题集、更多实验重复，而不是只提高自评分。')
    para('局限：固定关系模板、固定2—4跳、所有基准题可答；结构同配性为人为分布设定；LLM为0.5B小模型且仅9题，模型解释没有形式化事实核查；专家约束依赖输入证据标注；GA只在有限搜索空间调参。迁移到定理证明或规划时，需将节点替换为证明/规划状态，动作替换为可验证推理规则，终止条件替换为目标公式或规划目标，并确保动作可由外部验证器独立检查。')
    doc.add_heading('8.1 可靠性与交互改进', 2)
    para('本轮优化保持搜索排序与评分协议不变，重点修复失败状态和证据展示：预算耗尽、候选穷尽和无效证据分别给出解释；低置信度边不再被显示为有效；BFS与MCTS不展示未参与计算的A*启发式。GA或路径分类器缺失、损坏、种子不匹配时明确拒绝运行，不用固定权重或中性概率冒充模型。LLM评分还需通过图、问题、边编号与数值范围校验。')
    para('网站提供模块内导航、请求超时、停止等待、恢复重试、逐步播放、播放速度、缩放、定位当前与步骤列表。取消请求不会把旧响应写回新任务；停止等待并不保证后端立即终止。手机和键盘用户可独立查看完整路径名称与证据；超过轨迹记录上限时明确标注缺失事件。GA曲线附每代数值表，算法配对结果展示图级重采样区间，LLM推断耗时与搜索工作量分列。')
    doc.add_heading('9 复现与提交',1)
    para('网站入口：双击 experiments/lab03/打开网站.cmd，或总首页选择实验三。四个页面为搜索工作台、遗传优化与路径学习、LLM辅助与证据约束、对比实验与失败分析。全部任务从当前代码运行，原两次实验保留独立语义。')
    for cmd in ['python -B -X utf8 main.py lab03 demo','python -B -X utf8 main.py lab03 benchmark','python -B -X utf8 scripts/prepare_lab03_model.py','python -B -X utf8 main.py lab03 llm','python -B -X utf8 main.py lab03 report','python -B -X utf8 -m unittest discover -s tests -t .']:
        para(cmd)
    para('网页默认重放保存的真实LLM评分，无需下载权重即可演示；重新生成真实模型结果需下载约0.99GB权重并安装requirements-lab03.txt。远程接口通过环境变量配置。运行benchmark会重新生成数值结果，运行llm会重新生成对应模型记录，报告再从这些文件读取。')
    qa=OUT/'acceptance.json'
    if qa.exists():
        a=read('acceptance.json')
        para(f'本轮验收：{a["unit_tests"]}项单元测试通过，前端构建与路由检查通过；3,888次基准和18次LLM配对结果逐项回放一致。320、390、768、920、1366像素视口的四个实验三模块无横向溢出，网页搜索、轨迹播放、导出、预算耗尽及前两次实验运行均已验证。Demo为60秒、2560×1440、30fps，完整解码通过。')
    para('AI使用说明：本次实现、测试和文稿整理使用AI编程助手协助；报告数值由代码运行生成，模型评分来自本地真实推断。小组应理解算法、核对材料并承担提交与演示责任。成员名单沿用前两次作业，具体人工分工应由小组确认，未虚构成员实施工作。')
    doc.add_heading('参考资料',1)
    for r in brief['references']:para(f'[{r["id"]}] {r["title"]} {r["url"]} （核对日期：2026-11-08）')
    from docx.oxml import OxmlElement
    for t in doc.tables:
        head=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(head)
        for row in t.rows:
            no_split=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(no_split)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('第 ');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field);footer.add_run(' 页')
    dest=DEL/'第31组_实验三_课程报告.docx';doc.save(dest)
    build_slides(summary,llm,ga,fig)
    build_plan(brief,manifest)
    print('Generated',dest,flush=True)


def build_slides(summary,llm,ga,fig):
    from pptx import Presentation
    from pptx.util import Inches,Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    bg='F8F5F0';ink='292724';muted='71685F';accent='9A4728'
    def box(s,x,y,w,h,text,size=22,bold=False,color=ink):
        shape=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=shape.text_frame;tf.word_wrap=True
        for i,line in enumerate(text.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Microsoft YaHei';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(10)
        return shape
    def page(title,kicker='实验三 · 多跳路径搜索'):
        s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string(bg)
        box(s,.55,.3,12,.35,kicker,12,False,accent);box(s,.55,.85,12.2,1,title,30,True);box(s,.55,7.08,12,.23,f'第31组  ·  教学数据与真实运行结果  |  {len(prs.slides):02d}',10,False,muted);return s
    def notes(s,text):s.notes_slide.notes_text_frame.text=text
    m=lambda split,method:next(r for r in summary if r['split']==split and r['budget']==8 and r['method']==method)
    s=page('让模型辅助找路，让证据决定答案');box(s,.65,2.1,11.8,1.3,'企业股权知识图谱多跳推理智能体',36,True);box(s,.65,3.7,11.8,2,'BFS / A* / MCTS  ·  GA优化  ·  路径价值学习\n共享场景延续实验一、二，形成统一课程项目',25);notes(s,'实验三把多跳问答建模为状态空间搜索。我们继续使用股权场景，让模型辅助搜索，但由真实图路径和证据规则决定答案。')
    s=page('同一张关系图，回答2—4跳问题');box(s,.65,2,12,3.7,'状态：当前节点 + 深度 + 已走路径\n动作：下一类关系对应的现存出边\n目标：关系顺序完成，终点属性匹配\n终止：找到路径 / 搜索耗尽 / 预算耗尽',26);notes(s,'问题给定起点、关系顺序和终点属性。每次动作必须来自图中已有关系，区分没有答案和预算耗尽。控制与关系可达性仍是不同概念。')
    s=page('三套搜索内核，共享验证标准');box(s,.65,2,12,4.4,'BFS：逐层扩展，提供盲目搜索基线\nA*：f=g+h；下界版作正确性对照\nMCTS：UCT选择 → 扩展 → 模拟 → 回传\n单独记录rollout，防止用少量树节点掩盖开销',25);notes(s,'BFS、A星与MCTS独立实现。A星下界版本用于核对正确性，学习启发式不保证可采纳。MCTS必须同时报告模拟开销。')
    s=page('遗传优化只看验证图');s.shapes.add_picture(str(fig/'03_ga.png'),Inches(.7),Inches(1.9),width=Inches(8.2));box(s,9.1,2.15,3.6,3.7,'14个体 × 10代\n3个随机种子\n成功率 − 搜索代价\n测试集不参与选择',22);notes(s,'遗传算法在验证图优化五个启发式权重，适应度结合成功率与扩展成本。训练图只用于路径分类器，测试图从未参与参数选择。')
    s=page('规则、学习与LLM各司其职');box(s,.65,2,12,4.5,'专家规则：无效、过期、低置信度证据直接拦截\n路径分类器：6个局部特征预测前缀延伸价值\nLLM：评估候选边，不能生成答案或修改图\n解释在搜索完成后生成，保留原文供核对',24);notes(s,'我们的混合设计把排序和合法性分开。分类器和大模型帮助排序，专家约束控制证据边界，最终答案仍由程序验证。')
    s=page('405题、3,888次运行，检查分布变化');s.shapes.add_picture(str(fig/'01_success.png'),Inches(.6),Inches(1.8),width=Inches(12.1));notes(s,f'在预算8的测试图上，BFS为{pct(m("test","bfs")["success_rate"])}，固定A星为{pct(m("test","astar_fixed")["success_rate"])}，GA为{pct(m("test","astar_ga")["success_rate"])}。分布变化后收益可能消失，不能宣称全面优胜。')
    s=page('树扩展少，不代表总成本低');s.shapes.add_picture(str(fig/'02_work.png'),Inches(1),Inches(1.85),width=Inches(11.2));notes(s,'MCTS通过模拟提前发现路径，但模拟也会读取和检查边。因此报告总工作量、候选检查与耗时，不只看扩展节点。')
    s=page('真实小模型实测：如实报告没有收益');box(s,.65,2,12,4.2,f'Qwen2.5-0.5B-Instruct，本地真实推断\n9题配对：{pct(llm["results"][0]["success_rate"])} → {pct(llm["results"][1]["success_rate"])}\n合法性一致率：{pct(llm.get("legal_check_accuracy",0))}\n小样本与小模型，不外推到全面性能',26);notes(s,'我们真实调用本地小模型，并保存原始评分、提示词和解释。九题对子实验不足以证明普遍收益，模型还可能错误改写实体，界面会给出核验提示。')
    s=page('搜索树可回放，失败可以定位');s.shapes.add_picture(str(fig/'04_search_tree.png'),Inches(.65),Inches(1.8),width=Inches(12));notes(s,'网站支持逐步播放真实搜索树，查看路径证据和规则拦截。失败包括低预算、分布变化和模型误判，均保留记录而非只挑成功样例。')
    s=page('形成可复现的课程项目');box(s,.65,2,12,4.1,'总首页 → 实验三 → 四个可操作模块\n代码、图级隔离数据、原始结果与Word报告完整对应\n下一步：更大模型、授权真实数据、开放式问题\n课程设计创新，结论保持边界',26);notes(s,'实验三已经接入统一网站，材料与运行记录可对应复现。当前完成的是可核验教学系统，后续再扩展真实数据、更大模型和开放式问题。')
    prs.save(DEL/'第31组_实验三_汇报.pptx')
    from docx import Document
    from docx.shared import Pt
    d=Document();d.add_heading('实验三汇报讲稿（约3分钟）',0)
    d.add_paragraph('对应10页PPT；不含开场自我介绍与结尾致谢。时长以实际排练为准。')
    for i,s in enumerate(prs.slides,1):d.add_heading(f'第{i}页',2);d.add_paragraph(s.notes_slide.notes_text_frame.text)
    d.save(DEL/'第31组_实验三_汇报讲稿.docx')


def build_plan(brief,manifest):
    lines=['# 实验三实施计划、要求对应与自评依据','','本次按六项作业要求实现，并接入原统一平台；评分由教师最终决定。','','## 已完成的实现阶段','']
    lines+=['- '+s for s in brief['stages']]
    lines+=['','## 要求到代码与证据','', '| 要求 | 代码 | 验证依据 |','| --- | --- | --- |']
    for requirement in brief['requirements']:
        implementations = [p.strip() for p in requirement['implementation'].split('/')]
        paths = ["web/src/components/LabThree.tsx" if p == "LabThree.tsx" else "experiments/lab03/" + p for p in implementations]
        lines.append(f'| {requirement["id"]}：{requirement["requirement"]} | ' + ' / '.join('`'+p+'`' for p in paths) + f' | {requirement["evidence"]} |')
    lines+=['','## 自评建议','','建议B档90分。依据是六项任务均有实现、实测与可视化，含GA、ML、LLM和专家约束；保留失败案例与泛化边界。A档95分不应仅凭功能数量自评，需要老师认可创新深度、真实行业价值和演示完成度。','','评分截图给出的档位为A95/B90/C85/D80/E75，90及以上进入争优演示；85/80档需报名并由教师核实。额外奖励由教师按展示表现决定，不预先承诺。','','## 运行与复现','','- 双击 `experiments/lab03/打开网站.cmd`。','- 重跑：`python -B -X utf8 main.py lab03 benchmark`。','- 本地模型：`python -B -X utf8 scripts/prepare_lab03_model.py`，然后运行 `python -B -X utf8 main.py lab03 llm`。','- 生成材料：`python -B -X utf8 main.py lab03 report`。','- 网站默认提供真实评分记录重放；标注清楚，不冒充实时调用。','','## 提交前核对','','- 核对小组成员与课程平台分组。','- 全员排练并能解释BFS/A*/MCTS、GA适应度、训练/测试隔离和模型权限边界。','- 检查报告、PPT、网页和实测记录一致。','- 云端/课程平台提交需要自行按课程要求操作；本次未替你提交。','','## 局限','','合成且均可答的固定关系问题；0.5B模型9题小样本；不主张SOTA或真实企业风险准确率。GA在OOD可能退化，MCTS有额外模拟成本，LLM解释可能出现实体错误。']
    (DEL/'实验三_实施计划与验收.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')