"""Read-only collection and deterministic case replays for the Lab 3 report.
Only writes under deliverables/lab3; never regenerates the benchmark or calls an LLM.
"""
from __future__ import annotations
from pathlib import Path
import sys, json, csv, hashlib, datetime, platform, collections, statistics, shutil
from dataclasses import asdict
BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.lab03.dataset import load_dataset, unpack, shared_graph, make_graph
from experiments.lab03.domain import Graph, Node, Edge, Query, State, enumerate_paths, path_key, expert_legal, digest
from experiments.lab03.search import search, HYBRID_ML_WEIGHT
OUT = ROOT / 'outputs/lab03'
EVIDENCE = BASE / 'evidence'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    source_files = [
        'manifest.json','runs.csv','summary.json','paired_effects.json','ga_history.json','models.json',
        'learning_metrics.json','llm_summary.json','llm_diagnostics.json','llm_explanations.json',
        'llm_audit.json','hallucination_stress.json','boundary_cases.json',
        'optimization/comparison.json','refinement/comparison.json','structure/comparison.json',
        'reliability/comparison.json','reliability/confirmation.json','reliability/protocol.json',
        'reliability/llm_confirmation.json','reliability/failure_analysis.json'
    ]
    sources = []
    for f in source_files:
        src = OUT/f; dst = EVIDENCE/'raw'/f
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        sources.append({'source': src.relative_to(ROOT).as_posix(), 'attachment': dst.relative_to(BASE).as_posix(),
                        'bytes':src.stat().st_size, 'sha256':sha(src)})
    audit_dir = Path((ROOT/'outputs/runtime/latest-code-audit.txt').read_text(encoding='utf-8-sig').strip())
    audit_src = audit_dir/'代码核查与修正验收.json'
    shutil.copy2(audit_src, EVIDENCE/'code_audit_20261113.json')
    sources.append({'source':audit_src.relative_to(ROOT).as_posix(), 'attachment':'evidence/code_audit_20261113.json', 'sha256':sha(audit_src)})
    for f in ['data/lab03/benchmark.json','data/lab03/ga_warm_start.json','data/lab03/llm_model_manifest.json',
              'outputs/lab03/llm_readout_training.json','outputs/lab03/llm_evidence_readout.json']:
        src=ROOT/f; sources.append({'source':f, 'attachment':None, 'sha256':sha(src), 'bytes':src.stat().st_size})
    models=read(OUT/'models.json'); ga=read(OUT/'ga_history.json'); records=load_dataset()
    record_map={r['graph']['id']:r for r in records}
    runrows=list(csv.DictReader((OUT/'runs.csv').open(encoding='utf-8-sig',newline='')))
    run_map={(r['query_id'],r['method'],int(r['budget'])):r for r in runrows}
    cases=[]; case_dir=EVIDENCE/'cases'; case_dir.mkdir(exist_ok=True)
    def save_case(cid,g,q,method,budget,seed,*,weights=None,model=None,expert=True,llm_scores=None,kind='benchmark_replay',reason='',source='data/lab03/benchmark.json'):
        w=list(weights if weights is not None else ga[str(seed)]['weights'])
        if weights is None: w[5]=HYBRID_ML_WEIGHT
        model=model if model is not None else models[str(seed)]
        result=search(g,q,method,budget,w,model,seed,expert=expert,llm_scores=llm_scores,trace=True)
        # Independent exhaustive oracle only AFTER the algorithm finished.
        paths=enumerate_paths(g,q)
        if result['found']:
            s=State(tuple(result['path']),tuple(result['edge_ids']))
            assert g.valid_state(s)
            assert result['valid'] == g.valid_path(s,q)
        assert not result['trace_truncated']
        for ev in result['trace']:
            s=State(tuple(ev['path']),tuple(ev['edge_ids']))
            assert g.valid_state(s), (cid,ev)
            assert list(s.edges[:-1]) == ev['parent_path']
        benchmark=run_map.get((q.id,method,budget))
        matched=[]
        if kind=='benchmark_replay' and benchmark:
            for k,v in benchmark.items():
                if k not in result or k=='elapsed_ms':continue
                actual=result[k]
                if actual is None: assert v==''
                elif type(actual) is bool: assert v==str(actual),(cid,k,v,actual)
                elif type(actual) is int: assert int(v)==actual,(cid,k,v,actual)
                elif type(actual) is float: assert abs(float(v)-actual)<1e-10,(cid,k,v,actual)
                else: assert v==actual,(cid,k,v,actual)
                matched.append(k)
        obj={'case_id':cid, 'kind':kind,'selection_reason':reason,'source':source,
             'graph_sha256':digest(g.to_dict()),'query_sha256':digest(asdict(q)),
             'parameters':{'method':method,'budget':budget,'seed':seed,'weights':w,'model':model,
                           'expert':expert,'llm_scores':llm_scores},
             'graph':g.to_dict(),'query':asdict(q),'result':result,
             'independent_oracle':{'used_only_after_search':True,'valid_path_count':len(paths),
                                   'valid_paths':[{'nodes':list(s.nodes),'edge_ids':list(s.edges)} for s in paths]},
             'replay_validation':{'all_trace_states_valid':True,'trace_complete':True,
                                  'matches_saved_run_fields':matched,'elapsed_ms_not_compared':True}}
        write(case_dir/(cid+'.json'),obj)
        cases.append({'case_id':cid,'file':f'evidence/cases/{cid}.json','kind':kind,'query_id':q.id,
                      'method':method,'budget':budget,'status':result['status'],'found':result['found'],
                      'valid':result['valid'],'expanded':result['expanded'],'rollout_steps':result['rollout_steps'],
                      'trace_events':len(result['trace']),'oracle_paths':len(paths),'selection_reason':reason})
        return obj
    g,qs=unpack(record_map['test-7-0'])
    specs=[('S01_two_hop',0,'hybrid',8,'固定选择第一个测试图的2跳题，展示安全证据路径。'),
           ('S02_three_hop',7,'hybrid',8,'沿用归档搜索树问题，展示成功、专家拒绝、确定性剪枝与未扩展分支。'),
           ('S03_four_hop',2,'hybrid',8,'固定选择同图首个4跳题，覆盖要求的最高跳数。'),
           ('S04_mcts_four_hop',2,'mcts',8,'同题展示真实均匀rollout；不把模拟步数混入树节点扩展。'),
           ('F01_bfs_budget',7,'bfs',8,'与S02同图同题同预算，只更换搜索方法。'),
           ('F02_minimum_budget',2,'hybrid',4,'与S03同题，仅将预算由8降为4，展示目标生成但尚未出队。'),
           ('F03_no_expert',0,'hybrid_no_expert',8,'与S01同题，去掉专家约束后返回未经核验且已失效的边。')]
    for cid,qi,method,budget,reason in specs:save_case(cid,g,qs[qi],method,budget,7,reason=reason)
    g,qs=unpack(record_map['ood-19-0'])
    for cid,method,budget in [('F04_ood_fixed','astar_fixed',8),('F05_ood_ga','astar_ga',8),('F06_ood_ga_recovery','astar_ga',16)]:
        save_case(cid,g,qs[8],method,budget,19,reason='原OOD预算8中GA相对固定A*的唯一失题；预算16用于机制复核，不重新调参。')
    protocol=read(OUT/'reliability/protocol.json'); p=protocol['confirmation_plan']
    seed_model=dict(zip(p['seeds'],p['model_seeds']))
    graph,qs=make_graph(7001,2,'reliability_confirmation_shift',n=64,ood=True)
    for cid,budget in [('F07_fresh_shift',8),('F08_fresh_shift_recovery',16)]:
        save_case(cid,graph,qs[2],'astar_ga',budget,seed_model[7001],kind='frozen_confirmation_replay',
                  reason='冻结后的新图确认仍存在预算失败；保留预算扩大后的对照。',source='outputs/lab03/reliability/protocol.json')
    graph,qs=make_graph(p['seeds'][0],0,p['splits'][0],n=48,ood=False)
    q=qs[0];negative=Query(q.id+'-negative',q.start,q.relations,'confirmation-absent-sector')
    save_case('F09_proved_no_answer',graph,negative,'hybrid',8,p['model_seeds'][0],kind='derived_negative_fixture',
              reason='按冻结协议将目标行业改为图中不存在的行业，展示exhausted而非budget_exhausted。',
              source='outputs/lab03/reliability/confirmation.json')
    # Exact reproduction of diagnostics.controlled_counterexamples; not model predictions.
    model={'seed':7,'coef':[0.]*6,'intercept':0.}
    ns=[Node('p','起点',kind='Person'),Node('a','安全中间'),Node('b','干扰中间'),
        Node('z','目标公司',sector='制造',listed=True),Node('c','第二层',sector='制造'),Node('d','第三层',sector='制造')]
    graph=Graph(ns,[Edge('a-live','p','a','HOLDS'),Edge('z-dead','p','b','HOLDS'),
        Edge('middle','a','c','HOLDS'),Edge('next','c','d','HOLDS'),Edge('end','d','z','HOLDS'),Edge('bad-next','b','z','HOLDS')])
    q=Query('controlled-budget','p',('HOLDS',)*4,'制造')
    for cid,scores in [('F10_controlled_neutral',{path_key(['a-live']):.5,path_key(['z-dead']):.5}),
                       ('F11_controlled_wrong',{path_key(['a-live']):0.,path_key(['z-dead']):1.})]:
        save_case(cid,graph,q,'hybrid_llm',5,7,weights=[.1,0.,0.,0.,0.,0.],model=model,llm_scores=scores,
                  kind='controlled_injected_scores_not_real_llm',reason='精确复现固定微型反例，隔离错误评分对有限预算排序的影响。',
                  source='experiments/lab03/diagnostics.py::controlled_counterexamples')
    # Index and reporting inputs, computed rather than typed by hand.
    split_stats=[]
    for split in ['train','validation','test','ood']:
        rr=[r for r in records if r['split']==split]; edges=[len(r['graph']['edges']) for r in rr]
        split_stats.append({'split':split,'graphs':len(rr),'questions':sum(len(r['queries']) for r in rr),
                            'nodes':sum(len(r['graph']['nodes']) for r in rr),'edges':sum(edges),
                            'edge_range':[min(edges),max(edges)],
                            'hops':dict(collections.Counter(len(q['relations']) for r in rr for q in r['queries']))})
    sg,sq=shared_graph()
    ga_validation=[]
    for seed in [7,19,43]:
        pairs=[(g,q) for r in records if r['seed']==seed and r['split']=='validation' for g,qq in [unpack(r)] for q in qq]
        rs=[search(g,q,'astar_fixed',8,trace=False) for g,q in pairs]
        fit=statistics.mean(r['valid'] for r in rs)-.12*statistics.mean(r['expanded']/8 for r in rs)
        h=ga[str(seed)]
        ga_validation.append({'seed':seed,'queries':len(pairs),'fixed_fitness':fit,'fixed_success_rate':statistics.mean(r['valid'] for r in rs),
           'fixed_mean_expanded':statistics.mean(r['expanded'] for r in rs),'initial_champion_fitness':h['history'][0]['fitness'],
           'final_fitness':h['history'][-1]['fitness'],'final_mean_expanded':h['history'][-1]['expanded_fraction']*8,
           'final_weights':h['weights'],'unique_candidates':h['unique_candidates']})
    fresh=read(OUT/'reliability/llm_confirmation.json')
    write(EVIDENCE/'case_index.json',{'selection_policy':'Mechanism-based fixed representative selection; examples do not estimate frequency. All test/OOD questions remain in full statistical tables.',
                                     'presentation':'Node display names use short IDs; raw evidence is byte-preserved. Figures merge events by full edge prefix, never by endpoint alone.',
                                     'cases':cases})
    write(EVIDENCE/'report_metrics.json',{'split_stats':split_stats,'shared':{'nodes':len(sg.nodes),'edges':len(sg.edges),'questions':len(sq)},
          'ga_validation':ga_validation,'summary':read(OUT/'summary.json'),'paired_effects':read(OUT/'paired_effects.json'),
          'learning_metrics':read(OUT/'learning_metrics.json'),
          'fresh_search_summary':read(OUT/'reliability/confirmation.json')['summary'],
          'fresh_negative_summary':read(OUT/'reliability/confirmation.json')['negative_summary'],
          'llm_fresh':{k:fresh[k] for k in ['questions','prefixes','raw_metrics','calibrated_metrics','novel_prompt_prefixes',
                'novel_raw_metrics','novel_calibrated_metrics','inference_ms','paired_summary','numeric_raw','numeric_calibrated']},
          'cases':cases})
    write(EVIDENCE/'source_manifest.json',{'created_at':datetime.datetime.now().isoformat(timespec='seconds'),
         'report_date':'2026-11-13','root_path_at_generation':str(ROOT),'python':sys.version,'platform':platform.platform(),
         'policy':'Only copies and replays; original outputs and code unchanged. No new LLM inference, no data regeneration.',
         'sources':sources,'core_hashes_at_report_time':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'experiments/lab03').glob('*.py'))}})
    print(json.dumps({'cases':len(cases),'source_copies':len(source_files),'ga_validation':ga_validation,'shared':{'nodes':len(sg.nodes),'edges':len(sg.edges)},
                      'recovery':[(r['case_id'],r['status'],r['expanded']) for r in cases if 'fresh' in r['case_id']]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()