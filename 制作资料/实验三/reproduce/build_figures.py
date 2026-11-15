"""Generate publication assets from frozen metrics and complete search traces."""
from pathlib import Path
import json, html, math, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BASE=Path(__file__).resolve().parents[1]
FIG=BASE/'figures'; EVI=BASE/'evidence'
MET=json.loads((EVI/'report_metrics.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#64748b','text.color':'#172b43','axes.labelcolor':'#334155','xtick.color':'#475569',
                     'ytick.color':'#475569','savefig.dpi':240,'svg.fonttype':'none'})
COLORS={'bfs':'#718096','astar_fixed':'#287ca6','astar_ga':'#7954b3','hybrid':'#18836a','mcts':'#c48728'}
LABELS={'bfs':'BFS','astar_admissible':'A* 下界','astar_fixed':'A* 固定','astar_ga':'A* GA','astar_ml':'A* ML',
        'hybrid':'混合','hybrid_no_expert':'无专家','mcts':'MCTS'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def case(cid):return read(EVI/'cases'/f'{cid}.json')
def row(split,method,budget=8):return next(x for x in MET['summary'] if x['split']==split and x['method']==method and x['budget']==budget)
def save_plot(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=240,facecolor='white',bbox_inches='tight',pad_inches=.18)
    fig.savefig(FIG/(name+'.svg'),facecolor='white',bbox_inches='tight',pad_inches=.18)
    plt.close(fig)

# GA: true near-flat curve, baseline remains visible, no fake interpolation.
ga=read(EVI/'raw/ga_history.json')
fig,axs=plt.subplots(1,3,figsize=(10.4,3.2),sharey=True)
for ax,(seed,record),stat in zip(axs,ga.items(),MET['ga_validation']):
    hs=record['history']; x=[v['generation'] for v in hs]; y=[v['fitness'] for v in hs]
    ax.plot(x,y,'o-',color=COLORS['astar_ga'],lw=1.8,ms=3.5,label='本代最优')
    ax.axhline(stat['fixed_fitness'],color=COLORS['astar_fixed'],ls='--',lw=1.3,label='固定 h')
    ax.set_title(f'种子 {seed} · 27 道验证题',fontsize=11); ax.set_xlabel('代数（含第 0 代）')
    ax.set_xticks([0,2,4,6,8,10]); ax.set_ylim(.9362,.9399); ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.grid(axis='y',alpha=.18); ax.text(.98,.06,f'最终 {y[-1]:.6f}',ha='right',transform=ax.transAxes,fontsize=10)
axs[0].set_ylabel('验证适应度 J');axs[-1].legend(loc='upper right',fontsize=9,frameon=False)
fig.tight_layout();save_plot(fig,'02_ga_validation')
# Main comparison (all four required metrics are also tabulated in report).
methods=['bfs','astar_fixed','astar_ga','mcts','hybrid']
fig,axs=plt.subplots(1,2,figsize=(10.4,4.0))
for ax,split in zip(axs,['test','ood']):
    vals=[row(split,m)['success_rate']*100 for m in methods]
    y=np.arange(len(methods));bars=ax.barh(y,vals,color=[COLORS[m] for m in methods],height=.58)
    ax.set_yticks(y,[LABELS[m] for m in methods]);ax.invert_yaxis();ax.set_xlim(0,118);ax.set_xlabel('有效答案率（%）')
    ax.set_title(('测试集 · 108 题' if split=='test' else 'OOD · 54 题')+' / 预算 8',fontsize=12)
    for b,v in zip(bars,vals):ax.text(v+1.5,b.get_y()+b.get_height()/2,f'{v:.2f}',va='center',fontsize=10)
    ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();save_plot(fig,'03_accuracy_budget8')
# Budget sensitivity.
fig,axs=plt.subplots(1,2,figsize=(10.4,3.4),sharey=True)
for ax,split in zip(axs,['test','ood']):
    for m in ['bfs','astar_fixed','astar_ga','mcts']:
        vals=[row(split,m,b)['success_rate']*100 for b in [4,8,16]]
        ax.plot([4,8,16],vals,marker='o' if m!='astar_ga' else 's',ms=5,lw=1.6,
                ls='--' if m=='astar_ga' else '-',label=LABELS[m],color=COLORS[m])
    ax.set_xticks([4,8,16]);ax.set_ylim(-3,107);ax.set_title('测试集' if split=='test' else 'OOD')
    ax.set_xlabel('扩展预算');ax.grid(alpha=.15);ax.legend(fontsize=9,frameon=False,loc='lower right')
axs[0].set_ylabel('有效答案率（%）');fig.tight_layout();save_plot(fig,'04_budget_sensitivity')
# Costs: no misleading total-compute label.
fig,axs=plt.subplots(1,3,figsize=(10.4,3.8))
for ax,key,title in zip(axs,['mean_expanded','mean_total_work','mean_total_edge_checks'],['树节点扩展','扩展 + rollout 转移','候选证据检查']):
    vals=[row('test',m)[key] for m in methods];y=np.arange(len(methods))
    ax.barh(y,vals,color=[COLORS[m] for m in methods],height=.58);ax.invert_yaxis()
    ax.set_yticks(y,[LABELS[m] for m in methods]);ax.set_title(title,fontsize=11);ax.set_xlabel('每题均值')
    ax.set_xlim(0,max(vals)*1.23);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for i,v in enumerate(vals):ax.text(v+max(vals)*.02,i,f'{v:.2f}',va='center',fontsize=9)
fig.tight_layout();save_plot(fig,'05_cost_breakdown')
# Numerical-shift confusion matrices, columns prediction; rows rule truth.
fresh=MET['llm_fresh'];fig,axs=plt.subplots(1,2,figsize=(8.8,3.9))
for ax,k,title in zip(axs,['numeric_raw','numeric_calibrated'],['原始 Qwen 输出','隐藏表示分类头']):
    m=fresh[k];a=np.array([[m['true_legal'],m['false_reject']],[m['false_accept'],m['true_illegal']]])
    ax.imshow(a,cmap='Blues',vmin=0,vmax=90)
    for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',fontsize=20,color='white' if v>45 else '#173149')
    ax.set_xticks([0,1],['预测合法','预测不合法']);ax.set_yticks([0,1],['实际合法','实际不合法'])
    ax.set_title(f'{title}\n准确率 {100*m["accuracy"]:.1f}% / 均衡 {100*m["balanced_accuracy"]:.2f}%',fontsize=11)
    ax.tick_params(length=0)
fig.tight_layout();save_plot(fig,'06_llm_numeric_confusion')
# Negative-fixture before / after.
fig,axs=plt.subplots(1,2,figsize=(9.8,3.4))
for ax,split,title in zip(axs,['reliability_confirmation','reliability_confirmation_shift'],['常规新图派生无解题','变化新图派生无解题']):
    rs=[r for r in MET['fresh_negative_summary'] if r['split']==split and r['method']=='hybrid']
    x=np.arange(2); w=.33
    ax.bar(x-w/2,[100*r['before']['correct_exhausted']/r['n'] for r in rs],w,label='完整证明剪枝前',color='#94a3b8')
    ax.bar(x+w/2,[100*r['after']['correct_exhausted']/r['n'] for r in rs],w,label='完整证明剪枝后',color='#18836a')
    ax.set_xticks(x,[f'预算 {r["budget"]}' for r in rs]);ax.set_ylim(0,115);ax.set_title(title,fontsize=11)
    ax.set_ylabel('正确穷尽率（%）');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for i,r in enumerate(rs):
        for dx,s in [(-w/2,'before'),(w/2,'after')]:
            v=100*r[s]['correct_exhausted']/r['n'];ax.text(i+dx,v+2,f'{v:.1f}',ha='center',fontsize=9)
axs[0].legend(frameon=False,fontsize=9,loc='upper left');fig.tight_layout();save_plot(fig,'07_negative_confirmation')

# Self-contained dark SVG diagrams; every tree node is a full edge-prefix state.
PAL={'solution':'#34d399','expanded':'#22d3ee','frontier':'#94a3b8','rejected':'#fb7185','proof_pruned':'#fbbf24','rollout':'#a78bfa','invalid':'#fb7185'}
SVG_META=[]
class Svg:
    def __init__(self,w,h,title,subtitle):
        self.w=w;self.h=h;self.lines=[]
        self.add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">')
        self.add('<defs><style>text{font-family:"Microsoft YaHei","Noto Sans SC","PingFang SC",sans-serif;fill:#e2e8f0}</style><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1e293b" stroke-width="0.5"/></pattern>')
        for k,c in PAL.items():self.add(f'<marker id="arr-{k}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="{c}"/></marker>')
        self.add('</defs>');self.add(f'<rect width="{w}" height="{h}" fill="#0f172a"/><rect width="{w}" height="{h}" fill="url(#grid)"/>')
        self.text(30,38,title,23,'#f8fafc',weight=700);self.text(30,68,subtitle,14,'#94a3b8')
    def add(self,s):self.lines.append(s)
    def text(self,x,y,t,size=15,color='#e2e8f0',anchor='start',weight=400):
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" style="fill:{color}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(t))}</text>')
    def box(self,x,y,w,h,color=PAL['expanded'],fill='#16283a',dash=False):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="#0f172a"/>')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{color}" stroke-width="1.5"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')
    def line(self,x1,y1,x2,y2,status='frontier',dash=False):
        self.add(f'<path d="M{x1},{y1} C{(x1+x2)/2},{y1} {(x1+x2)/2},{y2} {x2},{y2}" fill="none" stroke="{PAL[status]}" stroke-width="1.6" marker-end="url(#arr-{status})"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')
    def save(self,name,meta=None):
        self.add('</svg>');(FIG/(name+'.svg')).write_text('\n'.join(self.lines),encoding='utf-8')
        SVG_META.append({'name':name,'viewBox':[self.w,self.h],**(meta or {})})

def short(s):return s.rsplit('-',1)[-1]

def tree(cid,name,title):
    c=case(cid);r=c['result'];events=r['trace'];nodes={};children=collections.defaultdict(list);exp=0
    for e in events:
        key=tuple(e['edge_ids'])
        if key not in nodes:
            nodes[key]={'events':[],'path':e['path'],'first':e['id'],'expanded_order':None}
            if key:children[key[:-1]].append(key)
        nodes[key]['events'].append(e)
        if e['status']=='expanded':exp+=1;nodes[key]['expanded_order']=exp
    # Tree-state traces carry ancestors; MCTS rollouts use same full-prefix identity.
    assert () in nodes
    assert all(k[:-1] in nodes for k in nodes if k)
    leaf=0;ys={}
    def place(k):
        nonlocal leaf
        if children[k]:ys[k]=sum(place(ch) for ch in children[k])/len(children[k])
        else:ys[k]=leaf;leaf+=1
        return ys[k]
    place(())
    depth=max(len(k) for k in nodes);bw=148 if depth==4 else 164;bh=69;col=bw+86
    w=max(900,60+depth*col+bw);h=165+max(1,leaf)*91
    status_text={'solution':'有效解','expanded':'已扩展','frontier':'待扩展','rejected':'专家拒绝','proof_pruned':'证明剪枝','rollout':'模拟路径','invalid':'证据无效'}
    subtitle=f'{c["query"]["id"]}  |  {r["method"]}  |  预算 {r["budget"]}  |  全部 {len(nodes)} 个路径状态 / {len(events)} 个事件'
    s=Svg(w,h,title,subtitle)
    for k,n in nodes.items():
        sts={e['status'] for e in n['events']}
        st=next((a for a in ['solution','rejected','proof_pruned','expanded','rollout'] if a in sts),'frontier')
        if st=='solution' and not r['valid']:st='invalid'
        n['status']=st;n['x']=30+len(k)*col;n['y']=110+ys[k]*91
    solution=set(tuple(r['edge_ids'][:i]) for i in range(1,len(r['edge_ids'])+1)) if r['valid'] else set()
    for k,n in nodes.items():
        if not k:continue
        p=nodes[k[:-1]];st='solution' if k in solution else n['status']
        s.line(p['x']+bw,p['y']+bh/2,n['x']-3,n['y']+bh/2,st,st in ['rejected','proof_pruned','rollout'])
        # Edge IDs only, relationship sequence appears in caption and raw JSON.
        midx=(p['x']+bw+n['x'])/2
        s.text(midx,n['y']+bh/2-7,(short(k[-1]) if '-R' in k[-1] else k[-1]),11,PAL[st],anchor='middle')
    for k,n in nodes.items():
        st=n['status'];s.box(n['x'],n['y'],bw,bh,PAL[st],fill='#122535' if st!='rejected' else '#321c30',dash=st=='frontier')
        suffix=f'   #{n["expanded_order"]}' if n['expanded_order'] else ''
        s.text(n['x']+12,n['y']+23,short(n['path'][-1])+suffix,18,PAL[st],weight=700)
        s.text(n['x']+12,n['y']+44,status_text[st]+f' · d={len(k)}',13)
        vals=next((e for e in reversed(n['events']) if e.get('h') is not None),None)
        line=f'h={vals["h"]:.3f}  f={vals["f"]:.3f}' if vals else f'事件 {n["first"]} / 前缀状态'
        s.text(n['x']+12,n['y']+60,line,11,'#94a3b8')
    # Legend outside main area; lines fit width regardless of depth.
    y=h-24;x=30
    legend=[('solution','解路径'),('expanded','已扩展 #序号'),('frontier','待扩展'),('proof_pruned','已证无解'),('rejected','拒绝 / 无效'),('rollout','模拟')]
    step=(w-60)/len(legend)
    for i,(st,txt) in enumerate(legend):
        s.add(f'<circle cx="{x+i*step}" cy="{y-5}" r="5" fill="{PAL[st]}"/>');s.text(x+i*step+11,y,txt,12,'#cbd5e1')
    s.save(name,{'case_id':cid,'full_trace':True,'states':len(nodes),'events':len(events),'layout':'full edge-prefix tree; no clipping or invented nodes',
                 'nodes': [{'edge_prefix':list(k),'x':n['x'],'y':n['y'],'w':bw,'h':bh,'status':n['status']} for k,n in nodes.items()]})

tree('S01_two_hop','08_tree_two_hop','两跳搜索：证据约束后的有效路径')
tree('S02_three_hop','09_tree_three_hop','三跳搜索：成功路径、拒绝与剪枝')
tree('S03_four_hop','10_tree_four_hop','四跳搜索：5 次扩展到达目标')
tree('S04_mcts_four_hop','11_tree_mcts','MCTS：树扩展与 rollout 分开计数')
tree('F01_bfs_budget','12_tree_bfs_budget','BFS：目标已生成，但预算内未出队')
tree('F03_no_expert','13_tree_no_expert','关闭专家：找到候选不等于证据有效')
tree('F05_ood_ga','14_tree_ood_ga','OOD：一次额外分支消耗关键预算')
tree('F09_proved_no_answer','15_tree_no_answer','无解样例：完整证据支持候选穷尽')
tree('F10_controlled_neutral','16_tree_controlled_neutral','受控反例：中性评分在预算内成功')
tree('F11_controlled_wrong','17_tree_controlled_wrong','受控反例：错误评分导致预算失败')
# Architecture with strict data and authority separation.
s=Svg(1080,635,'规则约束 + 学习估值 + 启发搜索','实线表示数据流；LLM 只能评估真实路径前缀，不能创建实体、边或答案。')
# Layout: input and search in middle; offline assets top; audit below.
boxes=[(30,118,255,91,'输入图谱与关系序列',['2—4 跳问题 / 终点公开谓词','实体、真实边与证据属性'],'expanded'),
       (365,118,295,91,'离线训练与验证',['训练图 → 6 维路径分类器','验证图 → GA 参数；测试不回流'],'rollout'),
       (745,118,305,91,'本地 Qwen 评估',['合法性 True / False + 节点价值','受控解释在搜索后生成'],'rollout'),
       (30,292,255,104,'动作与专家约束',['匹配关系 / 禁止路径内循环','核验 + 有效 + 置信度 ≥ 0.55'],'proof_pruned'),
       (365,292,295,104,'独立搜索内核',['BFS / A* / MCTS','A*：固定、GA、ML、混合'],'expanded'),
       (745,292,305,104,'路径前缀估值',['h = 剩余步数 + 特征罚项','pML / 有界 pLLM 仅影响排序'],'rollout'),
       (365,480,295,91,'返回与独立核验',['found 与 valid 分离','完整路径 → 图证据检查'],'solution'),
       (745,480,305,91,'可复核证据',['trace / 指标 / 原始 JSON','自由解释与证据文本分列'],'solution')]
s.line(155,209,155,292,'expanded');s.line(285,344,365,344,'proof_pruned');s.line(512,209,512,292,'rollout')
s.line(897,209,897,292,'rollout');s.line(745,344,660,344,'rollout');s.line(512,396,512,480,'solution');s.line(660,525,745,525,'solution')
for x,y,w,h,title,lines,col in boxes:
    s.box(x,y,w,h,PAL[col]);s.text(x+18,y+29,title,19,PAL[col],weight=700)
    for i,t in enumerate(lines):s.text(x+18,y+54+22*i,t,14,'#cbd5e1')
s.text(30,610,'硬约束决定能否使用证据；统计模型只决定优先看哪条合法路径。',16,'#94a3b8')
s.save('01_architecture',{'layout':'3 columns, separated offline assets, runtime search and output audit'})
# OOD queue order: actual expansions only, no invented causal trace.
a=case('F04_ood_fixed');b=case('F05_ood_ga');s=Svg(1010,629,'同一 OOD 问题：固定 A* 与 GA A* 的扩展顺序','ood-19-0-Q8 / 预算 8；表格直接取 trace.status = expanded，Nxx 为原实体 ID 后缀。')
for x,title,c in [(30,'固定 A*：第 8 次扩展找到目标',a),(520,'GA A*：预算 8 耗尽，目标尚待出队',b)]:
    s.text(x,110,title,18,'#22d3ee',weight=700)
    ev=[e for e in c['result']['trace'] if e['status']=='expanded']
    for i,e in enumerate(ev):
        y=132+i*49;path=' → '.join(short(n) for n in e['path']);col=PAL['proof_pruned'] if short(e['node'])=='N34' else PAL['solution'] if i==7 and c is a else PAL['frontier']
        s.box(x,y,460,41,col);s.text(x+12,y+27,f'{i+1:02d}  {path}',15,col)
s.text(30,581,'GA 先扩展 N34，再转向 N51；预算增至 16 后，第 9 次扩展成功。',17,'#fbbf24')
s.text(30,609,'结论：证据规则未失效；失败来自分布变化下的排序及预算边界。',15,'#94a3b8')
s.save('18_ood_expansion_order',{'cases':['F04_ood_fixed','F05_ood_ga'],'source_events':'expanded only'})
(EVI/'figure_index.json').write_text(json.dumps(SVG_META,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Generated',len(SVG_META),'diagram SVGs and 6 statistical chart pairs')