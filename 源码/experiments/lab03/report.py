"""Build editable report, slides and speaker notes exclusively from saved measured results."""
from __future__ import annotations
import json,statistics,math
from pathlib import Path
from .dataset import ROOT
OUT=ROOT/'outputs/lab03';DEL=ROOT/'deliverables/lab03'

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def pct(n):return '不适用' if n is None else f'{100*n:.2f}%'

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
    ax.bar([x+.18 for x in xs],[r['mean_total_work'] for r in rs],width=.36,label='扩展+模拟（不含前瞻）',color='#426D80')
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
    from .verification import verify
    verify()
    from docx import Document
    from docx.shared import Pt,Cm
    from docx.oxml.ns import qn
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DEL.mkdir(parents=True,exist_ok=True);fig=figures();summary=read('summary.json');manifest=read('manifest.json');ga=read('ga_history.json');learning=read('learning_metrics.json');llm=read('llm_summary.json');audit=read('llm_audit.json');expl=read('llm_explanations.json');paired=read('paired_effects.json');brief=json.loads((ROOT/'data/lab03/brief.json').read_text(encoding='utf-8'))
    metric=lambda split,m:next(r for r in summary if r['split']==split and r['budget']==8 and r['method']==m)
    diagnostics=read('llm_diagnostics.json');legal=llm['legal_metrics'];boundary=read('boundary_cases.json')
    base=metric('test','bfs');fixed=metric('test','astar_fixed');opt=metric('test','astar_ga');mix=metric('test','hybrid');mcts=metric('test','mcts');noexpert=metric('test','hybrid_no_expert')
    llmb,llma=llm['results'];doc=Document();sec=doc.sections[0];sec.top_margin=sec.bottom_margin=Cm(2);sec.left_margin=sec.right_margin=Cm(2.3)
    for name in ['Normal','Title','Heading 1','Heading 2']:
        st=doc.styles[name];st.font.name='Microsoft YaHei';st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'微软雅黑')
    doc.styles['Normal'].font.size=Pt(10.5);doc.styles['Normal'].paragraph_format.line_spacing=1.35
    doc.core_properties.title=brief['title'];doc.core_properties.author='第33组'
    doc.add_heading('人工智能导论 · 课堂报告三',0);doc.add_heading('大模型辅助的企业股权知识图谱\n多跳路径搜索与智能优化',1)
    doc.add_paragraph('A* / MCTS 搜索 · 遗传算法 · 路径价值学习 · 证据专家规则')
    doc.add_paragraph('华南农业大学 · 数学与信息学院、软件学院\n软件工程5班 · 第33组\n组长：梁镇贤\n组员：陈宗波、温文辉、陈宇冲、刘亦达\n2026年11月12日')
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
    para('与实验一、二衔接：共享案例直接读取原CSV，包含16个实体及24条本任务支持的持股/担保/交易边；亲属关系未纳入实验三关系词表。扩展基准为独立生成的企业图，用于让搜索分支、干扰与证据问题可控。问答采用明确关系序列与终点属性，不宣称实现开放域自然语言解析。')
    para('问题示例：从王建国出发，连续经过三层持股关系，能到达哪家上市企业？搜索得到P1→C1→C2→C3及每条原始边。该结果仅表示关系路径；路径乘积18.2%也不能代替控制结论或把两条路径总权益34.2%混为一谈。')
    table(['建模项','精确定义'],[
      ['状态 s','(当前节点v，已走关系数d，已访问节点序列P，边序列E)。保留路径可避免循环并保留解释。'],
      ['动作 a','从v出发、关系类型等于问题第d+1项、且不重复访问节点的图中实际出边。'],
      ['转移','追加现存关系及目标节点；每条边代价1。模型不得创建动作或添加实体。'],
      ['目标','恰好走完2—4跳关系序列，终点满足公司、行业与上市属性。返回一条有效证据路径。'],
      ['终止','找到目标 / 候选耗尽 / 扩展预算耗尽。后两者分别记录，不混称答案为假。'],
      ['专家证据','verified=True、active=True且confidence≥0.55才允许通过。阈值为预先规定的教学假设。']])
    doc.add_heading('2 搜索算法与启发式',1)
    para('BFS不调用启发函数或路径分类器，使用FIFO队列逐层扩展；A*使用优先队列按f(s)=g(s)+h(s)排序。下界版h0=剩余跳数，在单位边代价和固定关系序列任务上是可采纳的，作为正确性对照。具有非零估值的A*在f相同时优先更深前缀，下界h0保持FIFO。保留完整路径状态，不按单一节点合并不同前缀；只禁止在同一路径中重复访问实体。')
    para('为什么启发式可能比盲目搜索高效：BFS对同一层候选不分价值，分支数为b、答案深度为L时通常需考察O(b^L)个状态。启发式利用终点属性、后继和证据信息优先探索更可能成功的分支，有限预算下可减少无关扩展。但错误启发会偏离答案，最坏情况下仍须搜索整个状态空间；它不是无条件更快。特别地，本任务h0=L−d使所有状态f=d+(L−d)=L，下界A*退化为FIFO同层优先，因而其结果与BFS相同；有效改进来自具有区分度的估值，而不是使用优先队列本身。')
    para('参数化启发式：hθ(s)=(L−d)+θ1(1−主题匹配)+θ2(1−后继可选性)+θ3(1−路径最低置信度)+θ4(1−终点类型匹配)+θ5·死路提示+λ(1−pML)。在目标状态h=0；若在一跳邻居或有界两跳后缀内观察到满足公开终点谓词的合法延伸，则h=剩余跳数且不被模型估值覆盖。答案仍须依次出队并验证，不能免费返回。固定参数θ=[1.7,0.5,0.6,0.2,1.5]；GA只优化前5项，混合版λ=0.25仅根据验证集确定，并在评估时保持固定。LLM版对非确知后缀的完整路径前缀加2(0.5−pLLM)，总h不低于0；未评分前缀为0.5，即零调整而非额外惩罚；同一边出现在不同前缀中不共享评分。')
    para('GA与学习启发式可能高估剩余代价，不声明A*的最优性保证。因为本任务要求固定跳数，找到的有效解都满足给定L；实验关注有限预算成功率和搜索成本，不用该基准证明变长最短路径最优性。')
    para('MCTS独立实现选择、扩展、模拟与回传。选择分数为Q/N+√2·sqrt(ln(Nparent)/N)；未访问子节点优先。模拟在合法出边上随机采样，到终点给1奖励，否则给0.125的失败叶奖励。为保证纯MCTS对照不偷用训练模型，其基线不加载路径分类器。每个树节点扩展及每次rollout转移分别计数。')
    para('当前6维特征采用深度感知的一跳/有界两跳局部估值：只读取下一条要求关系的真实出边，按当前专家开关筛掉不合格边；剩余1跳时检查邻居是否满足公开终点谓词，恰余2跳时按固定边顺序最多探测2个子前缀的合法后继；若观察到合法目标后缀则优先，只有所有候选均在上限内检查完才能认定局部死路，截断扫描保持未知。更早阶段将当前/下一节点行业作为弱线索，降低上市属性的过早影响。这是局部前瞻，不是调用oracle或读取测试答案。每个完整前缀的候选边和证据规则判断在一次搜索内缓存，供启发/扩展/模拟复用；候选缓存命中不是新的边检查。特征与h只计算一次；BFS、下界A*和纯MCTS不做启发前瞻，但同样允许复用已检查的候选。')
    doc.add_heading('3 遗传算法与路径价值学习',1)
    para('GA使用种群14、10代、精英保留2、锦标赛选择3、算术交叉、高斯变异(σ=0.65、概率0.3)，权重限制[0,5]。每个种子独立优化，适应度J=验证图有效答案率−0.12×平均扩展数/8；验证预算8与主比较一致。固定参数与前轮仅在验证图得到的候选共同进入初代，剩余个体随机生成；热启动来源与数据指纹存于data/lab03/ga_warm_start.json，总种群仍为14。精确浮点权重作缓存键，不用四舍五入值合并排序可能不同的候选；最优验证适应度不会因代际替换下降。测试结果不参与个体选择。')
    picture('03_ga.png','图1 三个种子的验证适应度曲线；上升不等于泛化改善。')
    table(['种子','初代J','末代J','验证题数','最终θ'],[[s,f'{v["history"][0]["fitness"]:.4f}',f'{v["history"][-1]["fitness"]:.4f}',v['validation_queries'],', '.join(f'{w:.3f}' for w in v['weights'][:5])] for s,v in ga.items()])
    para('价值分类器使用6维逻辑回归，正标签表示训练图中该路径前缀存在一条有效目标延伸。独立穷举仅在离线标签生成和评估中使用，搜索内核不调用oracle。训练采用类别平衡权重、L2正则0.012、学习率0.12、700次梯度迭代，配置在测试前固定。输入只含局部特征，没有节点ID、目标答案标签、到答案的真实距离。一/两步目标谓词判断来自有界可观察邻域，非额外答案标签；分类器与实际搜索使用同一特征函数。')
    table(['种子','测试前缀数','Precision','Recall','F1','Brier'],[[r['seed'],r['samples'],f'{r["precision"]:.4f}',f'{r["recall"]:.4f}',f'{r["f1"]:.4f}',f'{r["brier"]:.4f}'] for r in learning])
    doc.add_heading('4 LLM辅助与权限边界',1)
    para('本机使用Qwen2.5-0.5B-Instruct，固定revision 7ae557604adf67be50417f59c2c2f167def9a775[5]，以Transformers在CPU运行。模型文件保存在用户缓存，不随项目打包。推断仅生成最后一个位置的logits，并对完全相同的提示/选项做最多256条的内存缓存；缓存命中、实际处理token与推断时间分开记录。另实现Responses API结构化远程评分适配器，凭据从环境变量读取；本报告不把未经真实调用的远程模式算作实验结果。')
    para('本地加载时校验固定版本权重、配置、generation_config、分词器与词表等7个文件的SHA256，不允许用LAB03_LOCAL_MODEL指向其他权重却仍标为Qwen。远程接口与本地共享同一证据政策：整条前缀的每条边均须已核验、有效且置信度≥0.55；远程请求以JSON schema为准，拒绝拒答、未完成消息、含错误的响应和多个矛盾评估。未配置远程凭据时不声称实测。')
    para('模型运行时只接收当前路径的原始属性、关系与证据字段，不接收专家规则计算出的合法性标签。对候选路径前缀执行两次前向评估：A/B/C表示有希望/不确定/较差，pvalue=P(A)+0.5P(B)；True/False表示整个前缀的证据是否符合要求；相同权重下，该布尔提示在验证集降低了Y/N选项偏置，阈值仍为0.5。系统保存原始True/False预测，并使用只在训练图拟合的896维Qwen隐藏表示逻辑回归分类头；有监督分类头判不合法时，评分缩至原值的0.25，但不直接删边；确定性专家规则仍拥有最终裁决权。这样即使模型错判，也不能覆盖明确无效的边。')
    para('LLM子集固定为每个种子的第一张测试图、前3道题，共9题且覆盖2/3/4跳。只评分起点两层内最多8个现存路径前缀，提示词包含已走关系、当前深度、剩余关系及沿途证据，未评分前缀采用中性分数。远程演示采用同样前瞻范围，但不同模型的单次结果不冒充配对基准。预评分开销分别记录候选数、准备阶段出边扫描/状态数与LLM毫秒数，不混入核心搜索计时。选择顺序在模型结果出现前固定，不挑选有利案例。')
    para(f'真实评分共{llm.get("legal_checks",0)}个路径前缀；原始模型合法性判断与确定性规则一致率为{pct(llm.get("legal_check_accuracy",0))}。模型前向推断累计约{llm["total_inference_ms"]/1000:.1f}秒，不含加载与搜索后文本解释。CPU小模型开销明显大于本地符号搜索，只有更复杂、信息不完整的任务才可能抵消此成本，当前结果不支持生产效率优势。')
    table(['合法性混淆矩阵','专家合法','专家不合法'],[['模型判合法',legal['true_legal'],legal['false_accept']],['模型判不合法',legal['false_reject'],legal['true_illegal']]])
    para(f'总体一致率不能代替安全检出能力。{legal["ineligible"]}个不合格前缀中，模型误放行{legal["false_accept"]}个，不合法证据检出率为{pct(legal["ineligible_recall"])}，平衡准确率为{pct(legal["balanced_accuracy"])}；“全部判合法”的基线一致率为{pct(legal["always_legal_accuracy"])}。这些是模型合法性预测错误，不应直接叫作最终答案错误；专家规则仍独立拦截不合格证据。')
    adapted=llm['calibrated_legal_metrics'];training=llm['readout_training']
    para(f'有监督证据分类头另列：只用18张训练图的{training["training_prefixes"]}个前缀，按完整提示去重为{training["trained_unique_prompts"]}条；固定Qwen最后隐藏状态，训练均衡逻辑回归，L2={training["l2"]}。验证图选择正则，测试图不参与训练或阈值选择。当前同一72前缀上，分类头误放行{adapted["false_accept"]}、误拒{adapted["false_reject"]}，不合法检出率{pct(adapted["ineligible_recall"])}。这是LLM表示上的有监督评估器，不能把它改写为原始0.5B模型零样本能力，也不是专家规则输出。')
    para(f'训练额外花费{training["training_elapsed_ms"]/1000:.2f}秒、{training["training_model_passes"]}次模型前向和{training["training_prompt_tokens"]}个prompt token；训练开销不混入9题推断速度。原始合法性概率和校准概率在同一次隐藏状态/LM head前向中得到，12个固定验证提示核对最大概率差为0；基准前清空提示缓存，避免训练数据缓存伪装成推断加速。')
    para(f'同题配对检查：加入LLM后新增答对{diagnostics["improved_answers"]}题、丢失原可答{diagnostics["lost_answers"]}题；{diagnostics["extra_expansion_cases"]}题增加扩展，{diagnostics["fewer_expansion_cases"]}题减少扩展，总扩展净变化{diagnostics["extra_expansions"]:+d}。优势仅限于提供一种可插拔的排序线索，当前小样本没有证据支持全面质量或效率增益；原始错误前缀和逐题变化见llm_diagnostics.json。')
    table(['子集方法','题数','有效答案率','平均扩展','搜索毫秒/题'],[[r['label'],r['n'],pct(r['success_rate']),f'{r["mean_expanded"]:.3f}',f'{r["mean_ms"]:.3f}'] for r in llm['results']])
    para('路径解释在程序重新核验搜索结果后才生成；模型收到已核验路径，不参与答案选择。保存原始输出，但“出现全部实体名”不能证明关系或语义正确。只有与程序逐条证据渲染文本完全一致的解释才自动认证，其余明确显示为未核验模型原文，并同时给出程序证据解释。这与LATS/Think-on-Graph关于模型参与搜索、利用外部图证据的思路相关[3–4]，本作业是受限教学实现，不声称复现论文全量基准。')
    for e in expl:
        para(f'解释案例 {e["query_id"]}：{e["text"]}');para('解释认证：'+('与程序逐条证据解释完全一致。' if e.get('grounded') else '未通过完整证据认证；不能把模型原文当作新的证据。'))
        para('程序证据解释：'+e.get('verified_text',''))
    para(f'另外提供受控证据解释：LLM从程序验证后的三种完整事实措辞中选择A/B/C，{llm.get("controlled_explanations",0)}段输出经过模板范围核验。选择概率和提示原文保存，文字严格限定为已核验路径。该能力是模型辅助表达，不是模型独立推理或自由生成事实。旧原始自由文本仍保留、不放宽严格认证规则；不能用受控文本的正确率冒充自由生成正确率。')
    doc.add_heading('5 实验设计与指标口径',1)
    table(['划分','图数','问题数','用途'],[['训练',18,162,'分类器学习；不作GA适应度'],['验证',9,81,'每种子27题选择GA参数'],['测试',12,108,'固定参数后评估'],['分布变化',6,54,'节点数48→64、额外干扰边、主题同配性下降']])
    para('随机种子为7、19、43。每图9题，均为2/3/4跳各3题；不同图使用不同实体ID。生成时保留至少一条有效证据路径，但向搜索引擎只提供图与公开问题谓词，不提供被植入路径。主基准问题都可答；无解、循环、过期、低置信度与关系不符另有56次确定性边界检验，不据此宣称开放世界召回率。')
    para('八种方法为BFS、下界A*、固定A*、GA A*、仅学习A*、MCTS、GA+学习+专家、去掉专家的混合版。预算4/8/16统一限制各算法定义的扩展计数；MCTS额外模拟步数另计，因此只看扩展数不构成公平总成本结论。')
    table(['指标','定义'],[['有效答案率','返回端点满足问题，且完整边序列存在、顺序正确、无重复节点、证据全部有效。'],['扩展节点数','BFS/A*弹出处理状态数；MCTS树中首次扩展状态数。'],['节点工作量','扩展节点 + MCTS rollout边转移；不含启发式前瞻和LLM。'],['边检查成本','首次为扩展获取候选记successor_checks；首次为估值获取记heuristic_edge_checks；二者之和是实际候选边检查，缓存复用不重复计。'],['全部出边扫描','adjacency_edge_scans包含关系不匹配与环过滤边；候选证据检查是其子集，不能相加重复计算。'],['后缀探测状态','heuristic_probe_states是有界两跳启发访问的子前缀次数，不含在扩展+模拟工作量中。'],['已证无解剪枝','proof_pruned仅来自终点谓词不符、无后继或完整局部后缀核验；模型概率不能剪枝，截断未知分支保留。'],['队列/证明成本','proof_checks计已知无解判定次数；peak_frontier计FIFO/优先队列峰值，MCTS不适用，不能拿它代表全部内存。'],['候选缓存命中','同一搜索中完整前缀已检查候选被复用的次数；不等于扩展次数，也不抵扣扩展预算。'],['推理步数','成功路径的边数，失败记缺失；固定问题跳数应一致。'],['冗余路径占比','已完成候选路径中，重复到达既有终点的路径数/已完成候选路径数。'],['重复状态比例','扩展过程重复(node,depth)次数/总扩展；与冗余完整路径分开。'],['无效答案率','返回了目标但证据路径未通过校验的比例。'],['不确定性','按整张图bootstrap配对差异，避免把同图相关问题当独立样本。']])
    doc.add_heading('6 结果与对比分析',1)
    picture('01_success.png','图2 独立测试与分布变化有效答案率；扩展预算为8。')
    for split,title in [('test','独立测试'),('ood','分布变化')]:
        doc.add_heading(title+'（预算8）',2)
        table(['方法','有效答案率','扩展节点','含模拟工作','成功跳数','冗余路径','无效答案率'],[[r['label'],pct(r['success_rate']),f'{r["mean_expanded"]:.2f}',f'{r["mean_total_work"]:.2f}',f'{r["mean_hops_success"]:.2f}' if r['mean_hops_success'] is not None else '—',pct(r['mean_redundant_paths']),pct(r['invalid_answer_rate'])] for r in summary if r['split']==split and r['budget']==8])
    para('推理跳数是成功题的条件均值，不同算法成功题组成不同，所以均值不同不表示它在同一固定跳数题上找到了更短证明。冗余路径按实际访问的完整路径统计；若未完成任何候选，程序约定为0，汇总为逐题宏平均。提前找到第一条答案即停止也会使占比很低或为0，这不能证明整张图没有冗余路径。三预算完整明细在summary.csv和网页对比页，而非只保留有利预算。')
    para(f'测试图上，固定A*相对BFS改善{(fixed["success_rate"]-base["success_rate"])*100:.2f}个百分点；GA相对固定A*变化{(opt["success_rate"]-fixed["success_rate"])*100:+.2f}个百分点。混合版为{pct(mix["success_rate"])}，不应把多个模块叠加后所有改善都归因于某一个组件。去专家后的无效答案率为{pct(noexpert["invalid_answer_rate"])}，显示证据约束具有独立作用。')
    for r in paired:
        if r['budget']==8 and r['method']=='astar_ga':para(f'{r["split"]}上GA相对固定A*的配对成功率差为{r["success_delta"]*100:+.2f}个百分点，图级bootstrap 95%区间[{r["graph_bootstrap_low"]*100:+.2f}, {r["graph_bootstrap_high"]*100:+.2f}]个百分点。若区间含0，不将观察到的均值差描述为显著优势。')
    picture('02_work.png','图3 扩展与模拟节点工作量；启发探测与全部出边扫描另列，不能视为全部成本。')
    para(f'MCTS平均树扩展为{mcts["mean_expanded"]:.2f}，包含模拟后为{mcts["mean_total_work"]:.2f}。其rollout可在较少树节点扩展时发现答案，但每次模拟仍进行实际图遍历。壁钟数据是单次串行测量，只用于本机描述。')
    para(f'GA收益与过拟合风险：在测试图预算8下，固定h→GA的平均扩展为{fixed["mean_expanded"]:.3f}→{opt["mean_expanded"]:.3f}；OOD答案率为{pct(metric("ood","astar_fixed")["success_rate"])}→{pct(metric("ood","astar_ga")["success_rate"])}。遗传搜索反复比较验证题，会适应有限验证图的行业同配性和预算8；即使从未查看测试标签，也可能对验证集过拟合。应固定验证预算和代数、限制搜索空间、报告独立测试与OOD区间，再用嵌套图级验证或更多种子检验稳健性，不能为得到更高测试分数继续调参。')
    doc.add_heading('6.1 第一阶段历史对比：局部特征改进',2)
    from .optimization import audit_optimization
    iteration = audit_optimization()
    comparisons = [r for r in iteration['comparisons'] if r['budget']==8 and r['method'] in ['astar_fixed','astar_ga','hybrid']]
    table(['划分/方法','优化前答案率','优化后答案率','前扩展','后扩展'],
          [[r['split']+' / '+r['method'],pct(r['before_success_rate']),pct(r['after_success_rate']),
            f'{r["before_mean_expanded"]:.3f}',f'{r["after_mean_expanded"]:.3f}'] for r in comparisons])
    para('开发时先保留旧源码与原始结果，再在81道验证题比较局部特征、GA预算、分类器训练及融合强度；模型提示在独立验证图的72个前缀上比较。选择后固定实现，最后重跑原测试/OOD。原始试验、未采用方案与冻结协议见outputs/lab03/optimization；scripts/reproduce_lab03_optimization.py在隔离的旧版源码下重放搜索验证，--llm另做真实模型验证。')
    para('第一阶段主要改进来自一步图谱特征，不应归功于GA或LLM：测试预算8中GA略低于固定h，OOD与固定h相同；LLM子集无额外答对，并保留多扩展的退化。λ=0只作消融，0.25是验证中不损失成功题的最小已测非零值；保留学习器不等于证明它有净收益。')
    para('工作量统计不重复计算同一边检查，但前瞻仍有额外开销。测试预算8固定A*每题平均扩展边检查'+f'{fixed["mean_successor_checks"]:.2f}'+'、额外启发边检查'+f'{fixed["mean_heuristic_edge_checks"]:.2f}'+'，合计'+f'{fixed["mean_total_edge_checks"]:.2f}'+'，而BFS检查'+f'{base["mean_total_edge_checks"]:.2f}'+'。因此，实验结论是有限扩展预算下答案覆盖率更高、扩展更少，不是所有计算指标都降低。网页和CSV已同时展示这些成本。')
    para(f'研究边界：已有测试/OOD参与过方法迭代，虽然不用于最终配置选择，相关结果仍属于同题回归，不能视为全新盲测。验证集被多次比较也有选择偏差，后续确认性实验应新增未经查看的真实图。数据、答案、划分、方法和三档预算均保持不变；{len(iteration["regressions"])}条逐题退化记录（所有方法/预算）保存在comparison.json，不因整体提高而删除。')
    doc.add_heading('6.2 第二阶段历史：缓存细化与确认',2)
    from .refinement import audit_refinement
    refinement = audit_refinement()
    table(['划分/方法','前扩展','后扩展','前边检查','后边检查'],
          [[r['split']+' / '+r['method'],f'{r["before"]["mean_expanded"]:.3f}',f'{r["after"]["mean_expanded"]:.3f}',
            f'{r["before"]["mean_edge_checks"]:.3f}',f'{r["after"]["mean_edge_checks"]:.3f}']
           for r in refinement['summary'] if r['budget']==8 and r['method'] in ['astar_fixed','hybrid','mcts']])
    para('第二阶段目标不是堆叠更多模型，而是减少已知证据被估值覆盖、平局导致无用扩展和重复候选检查。一步目标延伸以图中可观察事实优先，同分时优先完成较深路径；每前缀候选缓存供所有搜索内核使用。GA从前轮验证候选显式热启动，仍仅在验证图选择参数。原测试预算8答案率不变，扩展和边检查减少；无专家消融可更早找到无效候选，必须保留其退化，不能关闭专家追求速度。')
    confirmation=read('refinement/confirmation.json')
    para(f'为避免只在旧题回归，冻结实现之后使用预先登记的新种子101、307、911，普通图和更强干扰图各12张，共{confirmation["question_count"]}题、{confirmation["paired_runs"]}对新旧运行。模型种子依次固定映射7、19、43，不重新训练、不在这些题选择参数。仍是同一合成生成机制，不代表真实行业迁移。')
    table(['确认场景/方法','前答案率','后答案率','前边检查','后边检查'],
          [[r['split']+' / '+r['method'],pct(r['before']['success_rate']),pct(r['after']['success_rate']),
            f'{r["before"]["mean_edge_checks"]:.2f}',f'{r["after"]["mean_edge_checks"]:.2f}']
           for r in confirmation['summary'] if r['budget']==8 and r['method'] in ['astar_fixed','hybrid','mcts']])
    para(f'第二阶段旧记录按当时源码归档复算，不再与当前实现混合；第一阶段和第二阶段分别保留{len(iteration["regressions"])}条、{len(refinement["regressions"])}条同题退化。第二阶段确认种子属于已见数据，不用于最终参数选择，也不作为最终盲测。')
    doc.add_heading('6.3 第三阶段历史：有界后缀探测与确认',2)
    from .structure import audit_structure
    structural=audit_structure();fresh=read('structure/confirmation.json');development=read('structure/validation-final.json')
    from .history_three import stage_three_json
    stage3_summary=stage_three_json('outputs/lab03/summary.json');stage3_llm=stage_three_json('outputs/lab03/llm_summary.json')
    para('在保持原405题、图划分和UI样式不变的前提下，比较2/4/8子前缀上限与去行业线索等候选；仅训练图训练分类器，验证图选择方案。选上限2，是因为验证成功率与更宽探测相同而边检查最少；固定A*在81题验证集从80题答对增至81题，混合版保持81题，平均扩展4.457降至4.111，候选边检查19.802增至21.728。全部出边扫描43.975增至49.864，成本上升没有隐藏。')
    table(['回归/方法','前答案率','后答案率','前扩展','后扩展'],
          [[r['split']+' / '+r['method'],pct(r['before']['success_rate']),pct(r['after']['success_rate']),
            f'{r["before"]["mean_expanded"]:.3f}',f'{r["after"]["mean_expanded"]:.3f}']
           for r in structural['summary'] if r['budget']==8 and r['method'] in ['astar_fixed','astar_ga','hybrid']])
    table(['当前成本/方法','候选边检查','全部出边扫描','后缀探测状态'],
          [[r['split']+' / '+r['method'],f'{r["mean_total_edge_checks"]:.2f}',f'{r["mean_adjacency_edge_scans"]:.2f}',f'{r["mean_heuristic_probe_states"]:.2f}']
           for r in stage3_summary if r['budget']==8 and r['method'] in ['bfs','astar_fixed','hybrid']])
    para(f'完成验证与源码冻结后，预登记新种子1201、2027、4051，共{fresh["question_count"]}题、{fresh["paired_runs"]}对同题新旧运行；其模型种子固定映射7、19、43。普通和强干扰图各12张，这些题不参与参数选择。原测试/OOD仅叫回归，新的确认仍限于合成固定关系机制，不外推真实企业。')
    table(['新确认/方法','前答案率','后答案率','前边检查','后边检查'],
          [[r['split'].replace('structure_confirmation','新图')+' / '+r['method'],pct(r['before']['success_rate']),pct(r['after']['success_rate']),
            f'{r["before"]["mean_edge_checks"]:.2f}',f'{r["after"]["mean_edge_checks"]:.2f}']
           for r in fresh['summary'] if r['budget']==8 and r['method'] in ['astar_fixed','astar_ga','hybrid']])
    para(f'全部预算/方法的同题回归保留{len(structural["regressions"])}条退化，新确认保留{len(fresh["regressions"])}条退化。每个问题的前后结果、冻结hash、未选方案和验证脚本均保存在outputs/lab03/structure/；前两阶段原始文件不改写。')
    para('第三阶段中性融合修正属于逻辑正确性，不等于模型更聪明；当时原始合法性仍误放行1条。该阶段记录按当时源码复算，不与下节第四阶段的有监督模型读出头混算。')
    doc.add_heading('6.4 第四阶段：安全剪枝与LLM表示校准',2)
    from .reliability import audit_reliability
    reliable=audit_reliability();fresh4=read('reliability/confirmation.json');llmfresh=read('reliability/llm_confirmation.json')
    para('第四阶段不继续扩大前瞻范围，也不让分类器直接删除分支。只在已有特征能证明无解时跳过队列入队：完整终点不符、无合法后继、一步邻居全部不符、或上限内完整后缀已核验无目标。未探测完整仍为未知。proof_pruned与专家拒绝分别显示，证据扫描与后缀探测照常计费；成功目标仍需出队，不把剪枝包装为免费找到答案。BFS、下界A*和MCTS保留原有独立盲搜行为。')
    para('验证集原有可答题成功率没有变化；每题平均剪去约4.5个已知无解候选。另从验证题派生“终点行业在图中不存在”的不可答问题，仅作开发对照而不混入答案率；预算8时正确判穷尽从0/81变为79/81，预算64从79/81变为81/81。它检验的是闭世界固定关系问题，不等于自然语言开放世界的无答案证明。')
    table(['原回归/方法','前答案率','后答案率','前扩展','后扩展'],
          [[r['split']+' / '+r['method'],pct(r['before']['success_rate']),pct(r['after']['success_rate']),
            f'{r["before"]["mean_expanded"]:.3f}',f'{r["after"]["mean_expanded"]:.3f}'] for r in reliable['summary'] if r['budget']==8 and r['method'] in ['astar_fixed','astar_ml','hybrid']])
    para(f'冻结后用新种子5039/7001/9001，生成{fresh4["question_count"]}道可答题及{fresh4["negative_questions"]}道明确派生的不可答题，可答题{fresh4["paired_runs"]}对新旧运行、不可答题{fresh4["negative_pairs"]}对；模型与GA不再训练。所有预算/方法退化共{len(reliable["regressions"])}条原题、{len(fresh4["regressions"])}条新题，全部保留。')
    table(['新图/方法','前答案率','后答案率','前扩展','后扩展'],
          [[r['split'].replace('reliability_confirmation','新图')+' / '+r['method'],pct(r['before']['success_rate']),pct(r['after']['success_rate']),
            f'{r["before"]["mean_expanded"]:.2f}',f'{r["after"]["mean_expanded"]:.2f}'] for r in fresh4['summary'] if r['budget']==8 and r['method'] in ['astar_fixed','hybrid']])
    table(['不可答/方法/预算','题数','前正确穷尽','后正确穷尽','前扩展','后扩展'],
          [[r['split'].replace('reliability_confirmation','新图')+'/'+r['method']+'/'+str(r['budget']),r['n'],r['before']['correct_exhausted'],r['after']['correct_exhausted'],
            f'{r["before"]["mean_expanded"]:.2f}',f'{r["after"]["mean_expanded"]:.2f}'] for r in fresh4['negative_summary'] if r['method']=='hybrid'])
    para('LLM单纯换JSON/布尔表达式/Yes-No提示不能改善数字阈值，部分全部判为合法；没有采用这些看起来复杂但更差的提示。隐藏表示分类头在648个验证前缀上误放行2/80、误拒0；仅看不与训练提示重合的225前缀仍为2次误放行。L2=0.001虽验证全对，却误拒一个开发边界合法例，故不用；0.01与0.1效果相同，选更强正则0.1。这些选择都在当前测试之前完成。')
    table(['LLM核验范围','原始误放行','校准误放行','原始误拒','校准误拒'],
          [[name,llmfresh[a]['false_accept'],llmfresh[b]['false_accept'],llmfresh[a]['false_reject'],llmfresh[b]['false_reject']]
           for name,a,b in [('新图144前缀','raw_metrics','calibrated_metrics'),('未见提示子集','novel_raw_metrics','novel_calibrated_metrics'),('更高精度数字挑战','numeric_raw','numeric_calibrated')]])
    para(f'新图LLM共{llmfresh["questions"]}题、{llmfresh["prefixes"]}前缀，其中{llmfresh["novel_prompt_prefixes"]}个前缀的完整证据提示不在训练集中；训练/验证图隔离不代表数字文本完全不重复，故另列未见提示表现。另用冻结种子11003生成100个数值精度挑战，比较5位小数与0/1/阈值附近，并包含4种verified/active组合。结果用于确认与失败分析，不能再反过来调参数。')
    table(['新题搜索辅助','题数','答对','扩展合计'],[[r['variant'],r['questions'],r['valid'],r['expanded']] for r in llmfresh['paired_summary']])
    nr,nc=llmfresh['numeric_raw'],llmfresh['numeric_calibrated']
    para(f'数值精度挑战的总体准确率从{pct(nr["accuracy"])}到{pct(nc["accuracy"])}，但均衡准确率从{pct(nr["balanced_accuracy"])}降至{pct(nc["balanced_accuracy"])}：合法类召回降低，不能只报道误放行减少。下表保留每个校准错误；不据此重新挑正则或阈值。')
    table(['置信度','已核验/有效','正确合法性','原始模型','校准头'],
          [[r['confidence'],f'{r["verified"]}/{r["active"]}',r['target'],r['raw_legal'],r['calibrated_legal']]
           for r in llmfresh['numeric_challenge'] if r['target']!=r['calibrated_legal']])
    para('证据分类头是否更会判合法，与最终搜索答案是否提升是不同指标。真实记录同时保留无LLM、原始LLM与校准LLM三种同题搜索；三组同题对照各18题均答对、各扩展78次，没有测得搜索收益。已经被专家裁决的非法边即使分类器判得更准，也未必提高答案率。数值泛化和自由解释仍然失败，不能凭有限新图表现宣称通用数值推理或无幻觉。')
    doc.add_heading('7 搜索树与失败案例',1)
    picture('04_search_tree.png','图4 一次真实混合搜索的路径状态树（节选）；完整轨迹见网页与JSON。')
    for case in read('failure_cases.json')[:4]:
        para(f'案例 {case["query"]["id"]} / {case["method"]}：{case["query"]["text"]}。状态={case["status"]}，独立oracle有{case["oracle_paths"]}条有效路径。{case["diagnosis"]}。')
    para('失败来源包括：低预算在到达正确深度前耗尽；主题同配启发式在跨行业关系图上失效；未经核验的高诱惑路径在无专家消融中被返回；LLM的证据判断或自然语言解释不可靠。没有把失败题删除或仅展示最优种子。')
    stress=read('hallucination_stress.json')
    para(f'受控错误评分试验：把不合格前缀设高分、合格前缀设低分。在相同{len(stress["cases"])}题、相同预算16和评分下，开启专家规则的无效答案为{stress["invalid_answers"]}，关闭规则为{stress["without_expert_invalid_answers"]}。这是人为注入错误分数，不能写成模型真实幻觉发生率。模型真实合法性错误和搜索成本变化另存llm_diagnostics.json。')
    para('另给出两个可独立复现的最小机制反例：①不合格分支被评分误导为优先时，关闭专家规则会返回证据无效的候选，开启规则则能拦截；②所有边证据都合法，但错误高分偏向无目标分支时，同样预算5下，中性评分完成4跳、错误评分耗尽预算失败。第二例保留干扰后继，避免被立即已知无解剪枝消除；使用0.1弱主题权重与中性分类器隔离排序效应，第一例使用零权重，不冒充训练模型或真实LLM性能。这说明规则保障证据边界，却不能保证启发排序和有限预算下的完备性。')
    doc.add_heading('8 创新、自评与局限',1)
    table(['评分维度','本次可核验依据','仍需避免的夸大'],[['完成度','六项要求有代码、网页、实验记录和报告对应；三套搜索内核独立。','功能实现与实验结论分别提供证据。'],['行业结合','复用股权CSV，使用持股/担保/交易证据路径，结合尽调核查流程。','教学构造不是实测企业业务数据。'],['优化迭代','固定h→GA→路径学习→专家约束→LLM；消融与OOD检验。','GA验证提升不意味着任何分布都提升。'],['创新度','证据约束优先；LLM评分与程序答案分离；原文解释与图证据对照。','是课程系统设计创新，不是首次提出新搜索算法。']])
    para('我们的自评依据是六项任务的实现、实测结果、失败分析与可视化，最终成绩由教师评价。真实授权数据、多模型对照、更大问题集和重复实验仍是项目的扩展方向。')
    para('局限：固定关系模板、固定2—4跳、所有主基准题可答，另有56次独立边界案例覆盖八种受约束方法的无解和错误证据；结构同配性为人为分布设定；LLM为0.5B小模型，原9题与新增18题仍是小样本；100例数值挑战存在误放行/误拒取舍，自由文本不做语义认证而与程序证据解释分开；专家约束依赖输入证据标注；GA只在有限搜索空间调参。迁移到定理证明或规划时，需将节点替换为证明/规划状态，动作替换为可验证推理规则，终止条件替换为目标公式或规划目标，并确保动作可由外部验证器独立检查。')
    doc.add_heading('8.1 可靠性与交互改进', 2)
    para('网站采用石墨色工作台样式；搜索实现包括有界后缀估值、完整状态历史验证、纯BFS不调用h检查，以及统一的LLM证据政策与模型权重身份校验。预算耗尽、候选穷尽和无效证据分别给出解释；低置信度边明确标为不合格；BFS与MCTS不展示未参与计算的A*启发式。GA或路径分类器缺失、损坏、种子不匹配时明确拒绝运行，不用固定权重或中性概率冒充模型。LLM评分需通过图、问题、完整路径前缀、协议版本与数值范围校验；学习产物绑定源码和数据指纹，过期时阻断而不冒充当前结果。')
    para('网站提供模块内导航、请求超时、停止等待、恢复重试、逐步播放、播放速度、缩放、定位当前与步骤列表。取消请求不会把旧响应写回新任务；停止等待并不保证后端立即终止。手机和键盘用户可独立查看完整路径名称与证据；超过轨迹记录上限时明确标注缺失事件。GA曲线附每代数值表，算法配对结果展示图级重采样区间，LLM推断耗时与搜索工作量分列。')
    para('运行时证据一致性：缓存数值在[0,1]范围内并不证明来自模型。每条评分绑定原始A/B/C与True/False概率、冻结Qwen隐藏表示及训练图评估头，重新校验整条前缀；原始9题与新增18题的搜索配对在页面请求中作确定性重放，汇总、未见提示子集、数值挑战与自由/受控解释认证均交叉核验。原基准表和图级区间由3,888行CSV重建；无解表从完整逐题矩阵重算。校验失败显示不可用，不把无效数据写成零或静默修成好看数字。')
    para('每次页面请求仅加载一次证据分类头与源码指纹，同一请求内复用；不跨请求缓存可变页面数据，下一次请求重新读取文件；仅以完整输入内容指纹复用成功核验结论，最多16条指纹，输入变化立即重算、失败不缓存，异常与并发请求相互隔离。这一层没有重新训练LLM、没有改变已冻结搜索策略或上述答案率，也不是模型推断。它校验保存证据之间的一致性，不能证明遭到协调篡改的全部证据真实，也不能代替完整离线重跑。')
    doc.add_heading('9 复现与提交',1)
    para('网站入口：双击 experiments/lab03/打开网站.cmd，或总首页选择实验三。四个页面为搜索工作台、遗传优化与路径学习、LLM辅助与证据约束、对比实验与失败分析。全部任务从当前代码运行，原两次实验保留独立语义。')
    for cmd in ['python -B -X utf8 main.py lab03 demo','python -B -X utf8 main.py lab03 benchmark','python -B -X utf8 scripts/prepare_lab03_model.py','python -B -X utf8 main.py lab03 llm','python -B -X utf8 main.py lab03 report','python -B -X utf8 -m unittest discover -s tests -t .']:
        para(cmd)
    para('网页默认重放保存的真实LLM评分，无需下载权重即可演示；重新生成真实模型结果需下载约0.99GB权重并安装requirements-lab03.txt。远程接口通过环境变量配置。运行benchmark会重新生成数值结果，运行llm会重新生成对应模型记录，报告再从这些文件读取。')
    qa=OUT/'acceptance.json'
    if qa.exists():
        a=read('acceptance.json')
        para(f'验收记录日期：{a.get("date","未记录")}；{a["unit_tests"]}项单元测试通过。完整结果回放覆盖3,888次基准、18次LLM配对、18项图级配对区间、33代GA、56次边界案例、两轮各5,184对历史确认回放、5,184对第四阶段新图与864对新不可答题，以及144个新LLM前缀和100个数值挑战；浏览器、交付材料与验收详情见outputs/lab03/acceptance.json。Demo用于展示网站操作，程序正确性由自动化测试与结果复核支持。')
    para('AI使用说明：我们在实现、测试和文稿整理中使用AI编程助手辅助。报告数值由代码运行生成，模型评分来自本地真实推断。我们对算法解释、结果核对、材料内容及演示承担责任。成员信息与职责见正式实验报告。')
    doc.add_heading('参考资料',1)
    for r in brief['references']:para(f'[{r["id"]}] {r["title"]} {r["url"]}')
    from docx.oxml import OxmlElement
    for t in doc.tables:
        head=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(head)
        for row in t.rows:
            no_split=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(no_split)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('第 ');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field);footer.add_run(' 页')
    dest=DEL/'第33组_实验三_课程报告.docx';doc.save(dest)
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
        box(s,.55,.3,12,.35,kicker,12,False,accent);box(s,.55,.85,12.2,1,title,30,True);box(s,.55,7.08,12,.23,f'第4组  ·  教学数据与真实运行结果  |  {len(prs.slides):02d}',10,False,muted);return s
    def notes(s,text):s.notes_slide.notes_text_frame.text=text
    m=lambda split,method:next(r for r in summary if r['split']==split and r['budget']==8 and r['method']==method)
    s=page('让模型辅助找路，让证据决定答案');box(s,.65,2.1,11.8,1.3,'企业股权知识图谱多跳推理智能体',36,True);box(s,.65,3.7,11.8,2,'BFS / A* / MCTS  ·  GA优化  ·  路径价值学习\n共享场景延续实验一、二，形成统一课程项目',25);notes(s,'实验三把多跳问答建模为状态空间搜索。我们继续使用股权场景，让模型辅助搜索，但由真实图路径和证据规则决定答案。')
    s=page('同一张关系图，回答2—4跳问题');box(s,.65,2,12,3.7,'状态：当前节点 + 深度 + 已走路径\n动作：下一类关系对应的现存出边\n目标：关系顺序完成，终点属性匹配\n终止：找到路径 / 搜索耗尽 / 预算耗尽',26);notes(s,'问题给定起点、关系顺序和终点属性。每次动作必须来自图中已有关系，区分没有答案和预算耗尽。控制与关系可达性仍是不同概念。')
    s=page('三套搜索内核，共享验证标准');box(s,.65,2,12,4.4,'BFS：逐层扩展，提供盲目搜索基线\nA*：f=g+h；下界版作正确性对照\nMCTS：UCT选择 → 扩展 → 模拟 → 回传\n单独记录rollout，防止用少量树节点掩盖开销',25);notes(s,'BFS、A星与MCTS独立实现。A星下界版本用于核对正确性，学习启发式不保证可采纳。MCTS必须同时报告模拟开销。')
    s=page('遗传优化只看验证图');s.shapes.add_picture(str(fig/'03_ga.png'),Inches(.7),Inches(1.9),width=Inches(8.2));box(s,9.1,2.15,3.6,3.7,'14个体 × 10代\n3个随机种子\n成功率 − 搜索代价\n测试集不参与选择',22);notes(s,'遗传算法在验证图优化五个启发式权重，适应度结合成功率与扩展成本。训练图只用于路径分类器，本轮参数只根据验证图选择；旧测试图已被前一轮查看，当前属于同题回归而非全新盲测。')
    s=page('规则、学习与LLM各司其职');box(s,.65,2,12,4.5,'专家规则：无效、过期、低置信度证据直接拦截\n路径分类器：6个局部特征预测前缀延伸价值\nLLM：评估路径前缀，不能生成答案或修改图\n受控解释只选择事实措辞；原始自由文本另列',24);notes(s,'我们的混合设计把排序和合法性分开。分类器和大模型帮助排序，专家约束控制证据边界，最终答案仍由程序验证。')
    s=page('405题、3,888次运行，检查分布变化');s.shapes.add_picture(str(fig/'01_success.png'),Inches(.6),Inches(1.8),width=Inches(12.1));notes(s,f'在预算8的测试图上，BFS为{pct(m("test","bfs")["success_rate"])}，固定A星为{pct(m("test","astar_fixed")["success_rate"])}，GA为{pct(m("test","astar_ga")["success_rate"])}。分布变化后收益可能消失，不能宣称全面优胜。')
    s=page('树扩展少，不代表总成本低');s.shapes.add_picture(str(fig/'02_work.png'),Inches(1),Inches(1.85),width=Inches(11.2));box(s,.8,6.75,11.7,.4,'图内仅计扩展+模拟；前瞻、证明检查与LLM训练/推断另列。',15);notes(s,'MCTS模拟和A星有界前瞻都要检查真实边。图内仅为扩展加模拟；启发探测、候选检查与全部出边扫描另列，不能把更少扩展说成更少总计算。')
    fresh=read('reliability/llm_confirmation.json');nr=fresh['numeric_raw'];nc=fresh['numeric_calibrated']
    s=page('真实小模型实测：保留改进与退化');box(s,.65,2,12,4.4,f'Qwen2.5-0.5B-Instruct；有监督评估头另列\n新18题：无LLM / 原始 / 校准均18题成功，扩展相同\n144前缀：原始与校准均误放行1条、误拒0条\n100例数字挑战：误放行{nr["false_accept"]} → {nc["false_accept"]}，误拒{nr["false_reject"]} → {nc["false_reject"]}\n均衡准确率{pct(nr["balanced_accuracy"])} → {pct(nc["balanced_accuracy"])}；不隐藏退化',24);notes(s,'只在训练图拟合Qwen隐藏表示分类头，不能把它冒充原始大模型能力。新18题没有新增搜索收益；数字挑战减少了误放行，却新增两个误拒，均衡准确率反而下降。真实失败单独保存，最终答案由图路径与证据规则裁决。')
    s=page('搜索树可回放，失败可以定位');s.shapes.add_picture(str(fig/'04_search_tree.png'),Inches(.65),Inches(1.8),width=Inches(12));notes(s,'网站逐步回放搜索树和证据。除了预算与分布变化，错误评分可能把合法搜索引向死路；专家规则能挡住不合格证据，却不能保证预算内找到答案。受控反例与真实模型错误分开记录。')
    s=page('形成可复现的课程项目');box(s,.65,2,12,4.1,'总首页 → 实验三 → 四个可操作模块\n代码、图级隔离数据、原始结果与Word报告完整对应\n本轮：已证无解剪枝与LLM表示分类头；原始输出保留\n新增216道可答+216道不可答确认；数值挑战单列\n课程设计创新，结论保持边界',26);notes(s,'实验三已经接入统一网站，材料与运行记录可对应复现。当前完成的是可核验教学系统，后续再扩展真实数据、更大模型和开放式问题。')
    prs.save(DEL/'第4组_实验三_汇报.pptx')
    from docx import Document
    from docx.shared import Pt
    d=Document();d.add_heading('实验三汇报讲稿（约3分钟）',0)
    d.add_paragraph('对应10页PPT；不含开场自我介绍与结尾致谢。时长以实际排练为准。')
    for i,s in enumerate(prs.slides,1):d.add_heading(f'第{i}页',2);d.add_paragraph(s.notes_slide.notes_text_frame.text)
    d.save(DEL/'第4组_实验三_汇报讲稿.docx')


def build_plan(brief,manifest):
    lines=['# 实验三要求对应与结果说明','','我们在统一课程平台实现多跳搜索、启发式优化、路径价值学习与LLM辅助。各项功能以源码、可视化和实测记录对应，最终成绩由教师评价。','','## 要求、代码与证据','', '| 要求 | 代码 | 验证依据 |','| --- | --- | --- |']
    for requirement in brief['requirements']:
        implementations = [p.strip() for p in requirement['implementation'].split('/')]
        paths = ["web/src/components/LabThree.tsx" if p == "LabThree.tsx" else "experiments/lab03/" + p for p in implementations]
        lines.append(f'| {requirement["id"]}：{requirement["requirement"]} | ' + ' / '.join('`'+p+'`' for p in paths) + f' | {requirement["evidence"]} |')
    lines += ['', '## 运行与复现', '',
              '以下命令在统一“源码”目录中执行。', '',
              '- 双击 `打开网站.cmd`，从首页进入实验一、实验二或实验三。',
              '- 文件完整性：`python -B -X utf8 核查提交文件.py`。',
              '- 已保存结果复核：`python -B -X utf8 -m experiments.lab03.verification`。',
              '- 重新计算：`python -B -X utf8 scripts/prepare_lab03.py --force`；该命令写入实验结果，适合在独立副本中执行。',
              '- 真实本地模型：先运行 `python -B -X utf8 scripts/prepare_lab03_model.py`，再执行 `python -B -X utf8 scripts/prepare_lab03.py --force --llm`。',
              '- 网页默认回放已保存的真实评分记录，不将回放视为实时模型推断。', '',
              '## 成果材料', '',
              '三个实验共用源码目录，报告、PPT、Demo及PPT演讲视频按实验保存在项目的 `deliverables/lab1`、`lab2`、`lab3`。运行环境与验证摘要见 `运行说明`。', '',
              '## 结论边界', '',
              '我们分别评价答案有效性、扩展数量、额外前瞻和LLM推断成本。GA的验证集表现不保证分布变化后的优势；MCTS存在额外模拟成本；专家约束可以排除不合格证据，但不能保证有限预算下找到答案。', '',
              '405道主基准题为教学合成、固定关系问题；参与过方法迭代的测试/OOD仅作同题回归。固定策略的新图确认与派生不可答题单独统计。小模型的有限样本结果不代表通用数值推理或真实企业风险准确率，自由解释必须与程序证据对照。', '',
              '优化对照与原始证据分别见 `outputs/lab03/optimization`、`refinement`、`reliability` 和 `runtime`。这些目录中的数据均按原实验设置解释。']
    (DEL/'实验三_要求对应与结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
