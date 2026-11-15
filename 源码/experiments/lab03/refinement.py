"""Audit the preserved SECOND stage and replay its historical confirmation.

These supplementary questions never enter GA/ML training. Existing test/OOD
remain labelled regression, and the prior stage stays preserved for comparison.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import types
import zipfile
from pathlib import Path

from .dataset import ROOT, make_graph
from .domain import digest
from .artifacts import source_hashes, provenance
from .optimization import csv_rows
from .history import stage_two_bytes, stage_two_json, stage_two_file, stage_two_source_hashes, stage_two_search

OUT = ROOT/'outputs/lab03'
FOLDER = OUT/'refinement'
METHODS = ('bfs','astar_admissible','astar_fixed','astar_ga','astar_ml','mcts','hybrid','hybrid_no_expert')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def validate_frozen_strategy():
    protocol=read(FOLDER/'protocol.json')
    assert protocol['dataset_sha256']==provenance()['dataset_sha256']
    for path,sha in protocol['frozen_core_sha256'].items():
        assert hashlib.sha256(stage_two_bytes(path)).hexdigest()==sha, ('冻结策略被更改',path)
    assert hashlib.sha256((FOLDER/'stage_one_baseline.zip').read_bytes()).hexdigest()==protocol['previous_archive_sha256']
    for category,folder in [('validation_hashes',FOLDER),('baseline_hashes',FOLDER),('script_hashes',FOLDER/'validation_scripts')]:
        for name,sha in protocol[category].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha, ('细化证据失配',name)
    return protocol


def archived_search():
    """Load verified, project-owned baseline modules into a separate namespace.

    No extraction, sys.path override or change to current experiment modules.
    Byte strings are from the hash-checked archive produced before refinement.
    """
    validate_frozen_strategy()
    package='lab03_preserved_stage_one'
    if package+'.search' in sys.modules:
        return sys.modules[package+'.search']
    module=types.ModuleType(package);module.__path__=[];module.__package__=package
    sys.modules[package]=module
    # Baseline training provenance is never used here; only inference code runs.
    artifacts=types.ModuleType(package+'.artifacts');artifacts.provenance=lambda: {}
    sys.modules[artifacts.__name__]=artifacts
    with zipfile.ZipFile(FOLDER/'stage_one_baseline.zip') as archive:
        for name in ('domain','dataset','learning','search'):
            source=archive.read('experiments/lab03/'+name+'.py').decode('utf-8-sig')
            full=package+'.'+name;module=types.ModuleType(full);module.__package__=package
            module.__file__=str(ROOT/'experiments/lab03'/f'{name}.py')
            sys.modules[full]=module
            exec(compile(source,module.__file__+' [preserved]', 'exec'),module.__dict__)
    return sys.modules[package+'.search']


def confirm(*, write=True):
    preserved=stage_two_search();search=preserved.search;HYBRID_ML_WEIGHT=preserved.HYBRID_ML_WEIGHT
    domain=sys.modules[preserved.__package__+'.domain']
    protocol=validate_frozen_strategy();plan=protocol['confirmation_plan']
    models,ga=read(stage_two_file('models.json')),read(stage_two_file('ga_history.json'))
    before_models,before_ga=read(FOLDER/'previous_models.json'),read(FOLDER/'previous_ga_history.json')
    baseline=archived_search();old_domain=sys.modules[baseline.__package__+'.domain']
    rows=[];graphs=[]
    for seed,model_seed in zip(plan['seeds'],plan['model_seeds']):
        for split in plan['splits']:
            shift=split.endswith('_shift')
            for index in range(plan['graphs_per_seed_per_split']):
                graph,queries=make_graph(seed,index,split,n=64 if shift else 48,ood=shift)
                graphs.append({'id':graph.id,'sha256':digest(graph.to_dict()),'seed':seed,'model_seed':model_seed,
                               'questions':len(queries),'split':split})
                old_graph=old_domain.Graph.from_dict(graph.to_dict())
                for query in queries:
                    old_query=old_domain.Query(**query.__dict__)
                    for budget in plan['budgets']:
                        for method in METHODS:
                            w=ga[str(model_seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
                            old_w=before_ga[str(model_seed)]['weights'][:];old_w[5]=HYBRID_ML_WEIGHT
                            old=baseline.search(old_graph,old_query,method,budget,old_w,before_models[str(model_seed)],model_seed,trace=False)
                            new=search(domain.Graph.from_dict(graph.to_dict()),domain.Query(**query.__dict__),method,budget,w,models[str(model_seed)],model_seed,trace=False)
                            fields=('valid','found','expanded','total_work','total_edge_checks','status','hops')
                            if method in ('bfs','astar_admissible','mcts'):
                                assert all(old[k]==new[k] for k in fields if k!='total_edge_checks')
                            rows.append({'query_id':query.id,'graph_id':graph.id,'split':split,'budget':budget,'method':method,
                                         'before':{k:old[k] for k in fields},'after':{k:new[k] for k in fields}})
    groups={}
    for row in rows:groups.setdefault((row['split'],row['budget'],row['method']),[]).append(row)
    summary=[]
    for (split,budget,method),group in groups.items():
        summary.append({'split':split,'budget':budget,'method':method,'n':len(group),
                        **{stage:{'success_rate':sum(r[stage]['valid'] for r in group)/len(group),
                                  'mean_expanded':sum(r[stage]['expanded'] for r in group)/len(group),
                                  'mean_edge_checks':sum(r[stage]['total_edge_checks'] for r in group)/len(group),
                                  'invalid_answers':sum(r[stage]['found'] and not r[stage]['valid'] for r in group)}
                           for stage in ('before','after')}})
    result={'protocol':protocol['name'],'source_sha256':stage_two_source_hashes(),'question_count':sum(g['questions'] for g in graphs),
            'graphs':graphs,'paired_runs':len(rows),'summary':summary,'rows':rows,
            'note':'Fresh seeds generated after strategy freeze, not used to choose parameters. Synthetic fixed-relation task; not real-world generalization.'}
    if write:(FOLDER/'confirmation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def audit_refinement():
    protocol=validate_frozen_strategy()
    ga=read(stage_two_file('ga_history.json'));selection=read(FOLDER/'validation-final-strategy.json')
    warm=stage_two_json('data/lab03/ga_warm_start.json')
    for seed,weights in selection['ga'].items():
        assert ga[seed]['weights']==weights
        assert ga[seed]['initial_weights']==[warm['seed_records'][seed]['weights']]
    old,new=csv_rows(FOLDER/'previous_runs.csv'),csv_rows(stage_two_file('runs.csv'))
    assert len(old)==len(new)==3888 and set(old)==set(new)
    groups={};regressions=[]
    for key,before in old.items():
        after=new[key]
        assert all(before[k]==after[k] for k in ('seed','split','graph_id','hops_requested','answerable'))
        if key[2] in ('bfs','astar_admissible','mcts'):
            assert all(before[k]==after[k] for k in ('valid','found','expanded','total_work','status','hops'))
        assert int(after['total_edge_checks'])==int(after['successor_checks'])+int(after['heuristic_edge_checks'])
        groups.setdefault((after['split'],key[1],key[2]),[]).append((before,after))
        if before['valid']=='True' and after['valid']!='True':
            regressions.append({'query_id':key[0],'budget':key[1],'method':key[2],'split':after['split']})
    summary=[]
    for (split,budget,method),rows in groups.items():
        summary.append({'split':split,'budget':budget,'method':method,'n':len(rows),
                        **{stage:{'success_rate':sum(r[i]['valid']=='True' for r in rows)/len(rows),
                                  'mean_expanded':sum(int(r[i]['expanded']) for r in rows)/len(rows),
                                  'mean_edge_checks':sum(int(r[i]['total_edge_checks']) for r in rows)/len(rows)}
                           for i,stage in enumerate(('before','after'))}})
    llm=read(stage_two_file('llm_summary.json'));old_llm=read(FOLDER/'previous_llm_summary.json')
    return {'status':'passed','protocol':protocol['name'],'source_sha256':stage_two_source_hashes(),'pairs':len(old),
            'summary':summary,'regressions':regressions,
            'llm':{'before':{k:old_llm[k] for k in ('legal_metrics','paired_diagnostics','total_inference_ms')},
                   'after':{k:llm[k] for k in ('legal_metrics','paired_diagnostics','total_inference_ms','controlled_explanations','prediction_cache_hits','processed_prompt_tokens')}},
            'limits':protocol['limits']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--confirm',action='store_true');args=parser.parse_args()
    if args.confirm:
        result=confirm();print(json.dumps({k:result[k] for k in ('question_count','paired_runs','summary')},ensure_ascii=False,indent=2))
    else:
        result=audit_refinement();(FOLDER/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'status':result['status'],'pairs':result['pairs'],'regressions':len(result['regressions'])},ensure_ascii=False))
