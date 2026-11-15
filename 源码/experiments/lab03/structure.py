"""Audit the preserved THIRD-stage improvement; never rebind old results to live code.

Selection reads training/validation only. Existing test/OOD and stage-two seeds
are regression sets. Confirmation seeds below are fixed before generating data.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from .dataset import ROOT,make_graph
from .domain import digest
from .artifacts import source_hashes,provenance
from .history import stage_two_search,stage_two_file,stage_two_bytes
from .optimization import csv_rows
from .history_three import stage_three_bytes,stage_three_file,stage_three_sources,stage_three_search
FOLDER=ROOT/'outputs/lab03/structure'
OUT=ROOT/'outputs/lab03'
METHODS=('bfs','astar_admissible','astar_fixed','astar_ga','astar_ml','mcts','hybrid','hybrid_no_expert')
FIELDS=('valid','found','expanded','total_work','total_edge_checks','status','hops')

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_frozen_strategy():
    protocol=read(FOLDER/'protocol.json')
    assert protocol['dataset_sha256']==provenance()['dataset_sha256']
    assert all(hashlib.sha256(stage_three_bytes(path)).hexdigest()==sha for path,sha in protocol['source_sha256'].items()), '第三阶段源码与冻结协议失配；不能用新算法冒充冻结证据'
    for path,sha in protocol['evaluation_source_sha256'].items():
        assert hashlib.sha256(stage_three_bytes(path)).hexdigest()==sha, ('确认算法失配',path)
    for name,sha in protocol['validation_sha256'].items():
        assert hashlib.sha256((FOLDER/name).read_bytes()).hexdigest()==sha, ('验证集选择证据失配',name)
    for name,sha in protocol['script_sha256'].items():
        assert hashlib.sha256((FOLDER/'validation_scripts'/name).read_bytes()).hexdigest()==sha
    stage_two_file('runs.csv')
    return protocol

def paired_summary(rows):
    groups={}
    for row in rows:groups.setdefault((row['split'],row['budget'],row['method']),[]).append(row)
    output=[]
    for (split,budget,method),rs in groups.items():
        output.append({'split':split,'budget':budget,'method':method,'n':len(rs),
                       **{stage:{'success_rate':sum(r[stage]['valid'] for r in rs)/len(rs),
                                 'mean_expanded':sum(r[stage]['expanded'] for r in rs)/len(rs),
                                 'mean_edge_checks':sum(r[stage]['total_edge_checks'] for r in rs)/len(rs),
                                 'invalid_answers':sum(r[stage]['found'] and not r[stage]['valid'] for r in rs)} for stage in ('before','after')},
                       'gained':sum(not r['before']['valid'] and r['after']['valid'] for r in rs),
                       'lost':sum(r['before']['valid'] and not r['after']['valid'] for r in rs)})
    return output

def regressions(rows):
    return [{k:r[k] for k in ('query_id','split','budget','method')} for r in rows if r['before']['valid'] and not r['after']['valid']]

def audit_structure():
    protocol=validate_frozen_strategy()
    ga=read(stage_three_file('ga_history.json'));selection=read(FOLDER/'validation-final.json')
    assert {s:g['weights'] for s,g in ga.items()}==selection['ga_weights']
    old,new=csv_rows(stage_two_file('runs.csv')),csv_rows(stage_three_file('runs.csv'))
    assert len(old)==len(new)==3888 and set(old)==set(new)
    rows=[]
    def convert(row):
        return {k:(row[k]=='True' if k in ('valid','found') else row[k] if k=='status' else int(row[k]) if row[k] else None) for k in FIELDS}
    for key,before in old.items():
        after=new[key]
        assert all(before[k]==after[k] for k in ('seed','split','graph_id','hops_requested','answerable'))
        if key[2] in ('bfs','astar_admissible','mcts'):
            assert all(before[k]==after[k] for k in FIELDS)
        assert int(after['adjacency_edge_scans'])>=int(after['total_edge_checks'])
        assert int(after['total_edge_checks'])==int(after['successor_checks'])+int(after['heuristic_edge_checks'])
        rows.append({'query_id':key[0],'budget':key[1],'method':key[2],'split':after['split'],
                     'before':convert(before),'after':convert(after)})
    llm=read(stage_three_file('llm_summary.json'));old_llm=read(stage_two_file('llm_summary.json'))
    keys=('legal_metrics','paired_diagnostics','total_inference_ms','controlled_explanations','prediction_cache_hits','processed_prompt_tokens')
    return {'status':'passed','protocol':protocol['name'],'source_sha256':stage_three_sources(),'pairs':len(rows),
            'summary':paired_summary(rows),'regressions':regressions(rows),
            'llm':{'before':{k:old_llm[k] for k in keys},'after':{k:llm[k] for k in keys}},
            'limits':protocol['limits']}

def confirm(*,write_result=True):
    preserved=stage_three_search();search=preserved.search;HYBRID_ML_WEIGHT=preserved.HYBRID_ML_WEIGHT
    domain=sys.modules[preserved.__package__+'.domain']
    protocol=validate_frozen_strategy();plan=protocol['confirmation_plan']
    models,ga=read(stage_three_file('models.json')),read(stage_three_file('ga_history.json'))
    old_models,old_ga=read(stage_two_file('models.json')),read(stage_two_file('ga_history.json'))
    old=stage_two_search();old_domain=sys.modules[old.__package__+'.domain']
    rows=[];graphs=[]
    for seed,model_seed in zip(plan['seeds'],plan['model_seeds']):
        for split in plan['splits']:
            shifted=split.endswith('_shift')
            for index in range(plan['graphs_per_seed_per_split']):
                graph,queries=make_graph(seed,index,split,n=64 if shifted else 48,queries=plan['queries_per_graph'],ood=shifted)
                graphs.append({'id':graph.id,'sha256':digest(graph.to_dict()),'seed':seed,'model_seed':model_seed,'split':split,'questions':len(queries)})
                old_graph=old_domain.Graph.from_dict(graph.to_dict())
                for query in queries:
                    old_query=old_domain.Query(**query.__dict__)
                    for budget in plan['budgets']:
                        for method in METHODS:
                            w=ga[str(model_seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
                            old_w=old_ga[str(model_seed)]['weights'][:];old_w[5]=old.HYBRID_ML_WEIGHT
                            before=old.search(old_graph,old_query,method,budget,old_w,old_models[str(model_seed)],model_seed,trace=False)
                            after=search(domain.Graph.from_dict(graph.to_dict()),domain.Query(**query.__dict__),method,budget,w,models[str(model_seed)],model_seed,trace=False)
                            if method in ('bfs','astar_admissible','mcts'):assert all(before[k]==after[k] for k in FIELDS)
                            rows.append({'query_id':query.id,'graph_id':graph.id,'split':split,'budget':budget,'method':method,
                                         'before':{k:before[k] for k in FIELDS},'after':{k:after[k] for k in FIELDS},
                                         'after_probe_states':after['heuristic_probe_states'],'after_adjacency_edge_scans':after['adjacency_edge_scans']})
    result={'protocol':protocol['name'],'source_sha256':stage_three_sources(),'question_count':sum(g['questions'] for g in graphs),
            'graphs':graphs,'paired_runs':len(rows),'summary':paired_summary(rows),'regressions':regressions(rows),'rows':rows,
            'note':'Seeds fixed before generation after code/validation freeze. Synthetic fixed-relation retrieval only. These outcomes cannot be reused for tuning and still called unseen.'}
    if write_result:write(FOLDER/'confirmation.json',result)
    return result

def rebuild():
    comparison=audit_structure();write(FOLDER/'comparison.json',comparison)
    confirmation=confirm()
    return {'pairs':comparison['pairs'],'new_questions':confirmation['question_count'],'new_pairs':confirmation['paired_runs']}

if __name__=='__main__':
    print(json.dumps(rebuild(),ensure_ascii=False,indent=2))
