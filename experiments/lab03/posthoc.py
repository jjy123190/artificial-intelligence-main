"""Post-freeze failure analysis; never feeds labels back into search or tuning."""
from .dataset import ROOT,make_graph
from .domain import enumerate_paths
import sys
from .history_three import stage_three_search,stage_three_file
from .structure import read,write,validate_frozen_strategy
OUT=ROOT/'outputs/lab03'

def analyze_structural_failures(*,write_result=True):
    validate_frozen_strategy()
    preserved=stage_three_search();search=preserved.search;HYBRID_ML_WEIGHT=preserved.HYBRID_ML_WEIGHT
    domain=sys.modules[preserved.__package__+'.domain']
    conf=read(OUT/'structure/confirmation.json');models=read(stage_three_file('models.json'));ga=read(stage_three_file('ga_history.json'))
    selected=[r for r in conf['rows'] if r['budget']==8 and r['method'] in ('astar_fixed','hybrid') and not r['after']['valid']]
    records=[];graphs={}
    for row in selected:
        info=next(g for g in conf['graphs'] if g['id']==row['graph_id'])
        index=int(info['id'].rsplit('-',1)[1]);key=(info['seed'],index,info['split'])
        if key not in graphs:
            graphs[key]=make_graph(*key,n=64 if info['split'].endswith('_shift') else 48,ood=info['split'].endswith('_shift'))
        graph,queries=graphs[key];query=next(q for q in queries if q.id==row['query_id']);seed=info['model_seed']
        graph=domain.Graph.from_dict(graph.to_dict());query=domain.Query(**query.__dict__)
        weights=ga[str(seed)]['weights'][:];weights[5]=HYBRID_ML_WEIGHT
        result=search(graph,query,row['method'],8,weights,models[str(seed)],seed)
        # Oracle is called AFTER frozen search for explanatory labels only.
        solutions=domain.enumerate_paths(graph,query)
        prefixes={s.edges[:i] for s in solutions for i in range(query.hops+1)}
        expanded=[e for e in result['trace'] if e['status']=='expanded']
        extended=search(graph,query,row['method'],16,weights,models[str(seed)],seed,trace=False)
        records.append({'query_id':query.id,'method':row['method'],'split':graph.split,'hops':query.hops,
                        'status':result['status'],'oracle_paths':len(solutions),
                        'oracle_use':'post-freeze failure explanation only, never h/search input',
                        'expanded_prefixes':[{'edge_ids':e['edge_ids'],'depth':e['depth'],'h':e['h'],
                                              'on_any_valid_path':tuple(e['edge_ids']) in prefixes} for e in expanded],
                        'budget8':{k:result[k] for k in ('valid','expanded','heuristic_probe_states','total_edge_checks','adjacency_edge_scans')},
                        'budget16':{k:extended[k] for k in ('valid','expanded','total_edge_checks')},
                        'diagnosis':'Bounded local evidence does not certify a full 3/4-hop continuation. Incomplete probe/earlier noisy proxy may rank a nonproductive prefix first; budget exhausted is not no answer.'})
    value={'scope':'All expert-on astar_fixed/hybrid failures at budget8 in the new confirmation, not cherry-picked',
           'count':len(records),'cases':records,'no_expert_regressions_retained':len(conf['regressions']),
           'note':'No parameter was changed after examining these failures. New seeds now become historical for any future iteration.'}
    if write_result:write(OUT/'structure/failure_analysis.json',value)
    return value

if __name__=='__main__':
    result=analyze_structural_failures();print({'posthoc_failure_cases':result['count']})
