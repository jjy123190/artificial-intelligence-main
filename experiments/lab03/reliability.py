"""Fourth-stage measured reliability audit and post-freeze confirmation."""
from __future__ import annotations
import hashlib,json,sys,random
from .dataset import ROOT,make_graph
from .domain import Query,State,digest,path_key,expert_legal
from .artifacts import source_hashes,provenance
from .history_three import stage_three_file,stage_three_bytes,stage_three_search
from .optimization import csv_rows
from .structure import paired_summary,regressions
OUT=ROOT/'outputs/lab03';FOLDER=OUT/'reliability'
METHODS=('bfs','astar_admissible','astar_fixed','astar_ga','astar_ml','mcts','hybrid','hybrid_no_expert')
FIELDS=('valid','found','expanded','total_work','total_edge_checks','status','hops')

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_frozen_strategy():
    p=read(FOLDER/'protocol.json')
    assert p['source_sha256']==source_hashes(), '第四阶段冻结源码失配'
    assert p['dataset_sha256']==provenance()['dataset_sha256']
    for name,sha in p['evaluation_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    for name,sha in p['validation_sha256'].items():assert hashlib.sha256((FOLDER/name).read_bytes()).hexdigest()==sha
    for name,sha in p['script_sha256'].items():assert hashlib.sha256((FOLDER/'validation_scripts'/name).read_bytes()).hexdigest()==sha
    stage_three_file('runs.csv')
    return p

def audit_reliability():
    p=validate_frozen_strategy();ga=read(OUT/'ga_history.json')
    assert {s:r['weights'] for s,r in ga.items()}==read(FOLDER/'validation-final.json')['ga_weights']
    old,new=csv_rows(stage_three_file('runs.csv')),csv_rows(OUT/'runs.csv')
    assert set(old)==set(new) and len(new)==3888
    rows=[]
    def convert(r):return {k:(r[k]=='True' if k in ('valid','found') else r[k] if k=='status' else int(r[k]) if r[k] else None) for k in FIELDS}
    for key,before in old.items():
        after=new[key]
        assert all(before[k]==after[k] for k in ('graph_id','seed','split','hops_requested','answerable'))
        if key[2] in ('bfs','astar_admissible','mcts'):assert all(before[k]==after[k] for k in FIELDS)
        assert int(after['total_edge_checks'])==int(after['successor_checks'])+int(after['heuristic_edge_checks'])
        assert int(after['proof_pruned'])>=0
        rows.append({'query_id':key[0],'budget':key[1],'method':key[2],'split':after['split'],'before':convert(before),'after':convert(after)})
    llm=read(OUT/'llm_summary.json');old_llm=read(stage_three_file('llm_summary.json'))
    keys=('legal_metrics','paired_diagnostics','total_inference_ms','controlled_explanations')
    return {'status':'passed','protocol':p['name'],'source_sha256':source_hashes(),'pairs':len(rows),
            'summary':paired_summary(rows),'regressions':regressions(rows),
            'llm':{'before':{k:old_llm[k] for k in keys},'after':{k:llm[k] for k in keys},
                   'calibrated_after':llm['calibrated_legal_metrics'],'readout_training':llm['readout_training']},'limits':p['limits']}

def fresh_tasks():
    p=validate_frozen_strategy()['confirmation_plan']
    for seed,model_seed in zip(p['seeds'],p['model_seeds']):
        for split in p['splits']:
            shift=split.endswith('_shift')
            for index in range(p['graphs_per_seed_per_split']):
                g,qs=make_graph(seed,index,split,n=64 if shift else 48,queries=p['queries_per_graph'],ood=shift)
                yield seed,model_seed,index,g,qs

def confirm_search(*,write_result=True):
    from .search import search,HYBRID_ML_WEIGHT
    plan=validate_frozen_strategy()['confirmation_plan'];models,ga=read(OUT/'models.json'),read(OUT/'ga_history.json')
    old_models,old_ga=read(stage_three_file('models.json')),read(stage_three_file('ga_history.json'))
    old=stage_three_search();domain=sys.modules[old.__package__+'.domain']
    rows=[];negative=[];graphs=[]
    for seed,model_seed,index,g,qs in fresh_tasks():
        graphs.append({'id':g.id,'sha256':digest(g.to_dict()),'split':g.split,'questions':len(qs),'seed':seed,'model_seed':model_seed})
        old_g=domain.Graph.from_dict(g.to_dict());w=ga[str(model_seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
        ow=old_ga[str(model_seed)]['weights'][:];ow[5]=old.HYBRID_ML_WEIGHT
        for q in qs:
            old_q=domain.Query(**q.__dict__)
            for budget in plan['budgets']:
                for method in METHODS:
                    before=old.search(old_g,old_q,method,budget,ow,old_models[str(model_seed)],model_seed,trace=False)
                    after=search(g,q,method,budget,w,models[str(model_seed)],model_seed,trace=False)
                    if method in ('bfs','astar_admissible','mcts'):assert all(before[k]==after[k] for k in FIELDS)
                    rows.append({'query_id':q.id,'graph_id':g.id,'split':g.split,'budget':budget,'method':method,
                                 'before':{k:before[k] for k in FIELDS},'after':{k:after[k] for k in FIELDS},
                                 'proof_pruned':after['proof_pruned'],'proof_checks':after['proof_checks'],'peak_frontier':after['peak_frontier']})
            # Explicitly derived negative questions, never pooled into answer accuracy.
            nq=Query(q.id+'-negative',q.start,q.relations,'confirmation-absent-sector')
            assert all(n.sector!=nq.sector for n in g.nodes.values())
            for budget in plan['negative_budgets']:
                for method in ('astar_fixed','hybrid'):
                    before=old.search(old_g,domain.Query(**nq.__dict__),method,budget,ow,old_models[str(model_seed)],model_seed,trace=False)
                    after=search(g,nq,method,budget,w,models[str(model_seed)],model_seed,trace=False)
                    negative.append({'query_id':nq.id,'graph_id':g.id,'split':g.split,'budget':budget,'method':method,
                                     'before':{k:before[k] for k in FIELDS},'after':{k:after[k] for k in FIELDS},
                                     'proof_pruned':after['proof_pruned'],'peak_frontier':after['peak_frontier']})
    groups={}
    for r in negative:groups.setdefault((r['split'],r['budget'],r['method']),[]).append(r)
    negatives=[]
    for (split,budget,method),rs in groups.items():
        negatives.append({'split':split,'budget':budget,'method':method,'n':len(rs),
                          **{stage:{'correct_exhausted':sum(r[stage]['status']=='exhausted' and not r[stage]['found'] for r in rs),
                                    'false_found':sum(r[stage]['found'] for r in rs),
                                    'mean_expanded':sum(r[stage]['expanded'] for r in rs)/len(rs),
                                    'mean_edge_checks':sum(r[stage]['total_edge_checks'] for r in rs)/len(rs)} for stage in ('before','after')}})
    result={'source_sha256':source_hashes(),'question_count':sum(g['questions'] for g in graphs),'paired_runs':len(rows),
            'graphs':graphs,'rows':rows,'summary':paired_summary(rows),'regressions':regressions(rows),
            'negative_questions':sum(g['questions'] for g in graphs),'negative_pairs':len(negative),'negative_rows':negative,'negative_summary':negatives,
            'note':'Parameters frozen before fresh seeds; explicit absent-sector negative fixtures separately reported. No claim of open-world no-answer reasoning or equal total compute.'}
    if write_result:write(FOLDER/'confirmation.json',result)
    return result

def numeric_challenge():
    p=validate_frozen_strategy()['llm_confirmation'];rng=random.Random(p['numeric_seed'])
    values=[0.,.549,.55,.551,1.]+[round(rng.random(),5) for _ in range(p['random_confidences'])]
    for c in values:
        for verified,active in ((True,True),(False,True),(True,False),(False,False)):
            text=('Is every condition true? verified must be True; active must be True; minimum_confidence must be at least 0.55.\n'
                  +f'verified={verified}; active={active}; minimum_confidence={c}'
                  +'\nReturn True if all three conditions hold, otherwise False. Result:')
            yield {'confidence':c,'verified':verified,'active':active,'prompt':text,
                   'target':verified and active and c>=.55}

def confirm_llm():
    from .llm import score_actions,candidate_actions,clear_prediction_caches,evidence_prediction,bound_scores
    from .calibration import load_readout,probability
    from .diagnostics import legality_metrics
    import time
    from .search import search,HYBRID_ML_WEIGHT
    validate_frozen_strategy();readout=load_readout();clear_prediction_caches();audit=[];numeric=[];paired=[]
    models,ga=read(OUT/'models.json'),read(OUT/'ga_history.json')
    for seed,model_seed,index,g,qs in fresh_tasks():
        if index:continue
        for q in qs[:3]:
            records=score_actions(g,q,candidate_actions(g,q))
            audit.append({'query_id':q.id,'graph_id':g.id,'graph_sha256':digest(g.to_dict()),'query_sha256':digest(q.__dict__),'records':records})
            raw_scores={r['path_key']:r['score'] if r['legal'] else .25*r['score'] for r in records}
            calibrated_scores=bound_scores(records)
            w=ga[str(model_seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
            for variant,method,scores in [('no_llm','hybrid',None),('raw','hybrid_llm',raw_scores),('calibrated','hybrid_llm',calibrated_scores)]:
                result=search(g,q,method,8,w,models[str(model_seed)],model_seed,llm_scores=scores,trace=False)
                paired.append({'query_id':q.id,'variant':variant,'budget':8,
                               **{k:result[k] for k in ('valid','found','status','expanded','total_edge_checks','proof_pruned')}})

            print('Fresh LLM',q.id,flush=True)
    clear_prediction_caches()
    for entry in numeric_challenge():
        started=time.perf_counter();hidden,raw,n,hit=evidence_prediction(entry['prompt']);p=probability(readout,hidden)
        numeric.append({**entry,'hidden':hidden,'raw_probabilities':raw,'raw_legal':raw[0]>=raw[1],
                        'calibrated_probability':p,'calibrated_legal':p>=.5,'prompt_tokens':n,'cache_hit':hit,
                        'elapsed_ms':(time.perf_counter()-started)*1000})
    records=[r for a in audit for r in a['records']]
    training=read(OUT/'llm_readout_training.json');known={r['prompt'] for r in training['unique_prompts']}
    novel=[r for r in records if r['legality_prompt'] not in known]
    result={'source_sha256':source_hashes(),'readout_sha256':digest(readout),'questions':len(audit),'prefixes':len(records),'audit':audit,
            'raw_metrics':legality_metrics(records),'calibrated_metrics':legality_metrics([{**r,'legal':r['calibrated_legal']} for r in records]),
            'novel_prompt_prefixes':len(novel),'novel_raw_metrics':legality_metrics(novel),
            'novel_calibrated_metrics':legality_metrics([{**r,'legal':r['calibrated_legal']} for r in novel]),
            'inference_ms':sum(r['inference_ms'] for r in records),'paired_search':paired,
            'paired_summary':[{'variant':variant,'questions':sum(r['variant']==variant for r in paired),
                               'valid':sum(r['valid'] for r in paired if r['variant']==variant),
                               'expanded':sum(r['expanded'] for r in paired if r['variant']==variant)}
                              for variant in ('no_llm','raw','calibrated')],
            'numeric_challenge':numeric,
            'numeric_raw':legality_metrics([{'legal':r['raw_legal'],'expert_legal':r['target']} for r in numeric]),
            'numeric_calibrated':legality_metrics([{'legal':r['calibrated_legal'],'expert_legal':r['target']} for r in numeric]),
            'note':'Fresh post-freeze model evidence; numeric precision shift explicitly reported separately. No expert labels provided to model or readout. No claim of universal numerical reasoning.'}
    write(FOLDER/'llm_confirmation.json',result);return result

def rebuild():
    comparison=audit_reliability();write(FOLDER/'comparison.json',comparison)
    search=confirm_search();llm=confirm_llm()
    return {'pairs':comparison['pairs'],'new_questions':search['question_count'],'new_pairs':search['paired_runs'],'new_llm_prefixes':llm['prefixes']}

if __name__=='__main__':print(json.dumps(rebuild(),ensure_ascii=False,indent=2))
