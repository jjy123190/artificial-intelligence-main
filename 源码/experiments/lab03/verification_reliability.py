"""Recompute saved fresh-model evidence without treating replay as new inference."""
from __future__ import annotations
import math
from .reliability import read,FOLDER,OUT,validate_frozen_strategy,fresh_tasks,numeric_challenge
from .llm import candidate_actions,prompt_for,legality_prompt_for,bound_scores,SCORE_PROTOCOL
from .domain import digest,path_key,expert_legal
from .calibration import load_readout,probability
from .diagnostics import legality_metrics
from .search import search,HYBRID_ML_WEIGHT
from .artifacts import source_hashes

def verify_llm_confirmation():
    validate_frozen_strategy();saved=read(FOLDER/'llm_confirmation.json');model=load_readout()
    assert saved['source_sha256']==source_hashes() and saved['readout_sha256']==digest(model)
    tasks={q.id:(g,q,seed) for _,seed,index,g,qs in fresh_tasks() if index==0 for q in qs[:3]}
    assert set(tasks)=={r['query_id'] for r in saved['audit']} and len(saved['audit'])==len(tasks)
    models,ga=read(OUT/'models.json'),read(OUT/'ga_history.json');paired=[]
    for task in saved['audit']:
        graph,query,seed=tasks[task['query_id']]
        assert task['graph_sha256']==digest(graph.to_dict()) and task['query_sha256']==digest(query.__dict__)
        candidates=candidate_actions(graph,query);records=task['records'];assert len(records)==len(candidates)
        for row,(state,edge) in zip(records,candidates):
            prompt=prompt_for(graph,query,edge,state);assert row['prompt']==prompt and row['prompt_sha256']==digest(prompt)
            assert row['legality_prompt']==legality_prompt_for(prompt)
            assert row['path_key']==path_key(state.edges+(edge.id,))
            assert row['expert_legal']==all(expert_legal(graph.edges[e]) for e in state.edges+(edge.id,))
            assert row['protocol']==SCORE_PROTOCOL and row['readout_sha256']==digest(model)
            raw=row['legality_probabilities'];ps=row['probabilities']
            assert set(raw)=={'True','False'} and set(ps)=={'A','B','C'}
            for values in (raw,ps):
                assert all(math.isfinite(v) and 0<=v<=1 for v in values.values()) and math.isclose(sum(values.values()),1,abs_tol=1e-6)
            assert row['legal']==(raw['True']>=raw['False']) and row['score']==ps['A']+.5*ps['B']
            p=probability(model,row['evidence_hidden']);assert math.isclose(p,row['calibrated_confidence'],abs_tol=1e-7)
            assert row['calibrated_legal']==(p>=.5)
        raw_scores={r['path_key']:r['score'] if r['legal'] else .25*r['score'] for r in records}
        w=ga[str(seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
        for variant,method,scores in [('no_llm','hybrid',None),('raw','hybrid_llm',raw_scores),('calibrated','hybrid_llm',bound_scores(records))]:
            r=search(graph,query,method,8,w,models[str(seed)],seed,llm_scores=scores,trace=False)
            paired.append({'query_id':query.id,'variant':variant,'budget':8,**{k:r[k] for k in ('valid','found','status','expanded','total_edge_checks','proof_pruned')}})
    assert paired==saved['paired_search']
    records=[r for a in saved['audit'] for r in a['records']]
    assert saved['raw_metrics']==legality_metrics(records)
    assert saved['calibrated_metrics']==legality_metrics([{**r,'legal':r['calibrated_legal']} for r in records])
    known={r['prompt'] for r in read(OUT/'llm_readout_training.json')['unique_prompts']}
    novel=[r for r in records if r['legality_prompt'] not in known]
    assert saved['novel_prompt_prefixes']==len(novel)
    assert saved['novel_raw_metrics']==legality_metrics(novel)
    assert saved['novel_calibrated_metrics']==legality_metrics([{**r,'legal':r['calibrated_legal']} for r in novel])
    expected=list(numeric_challenge());numeric=saved['numeric_challenge'];assert len(expected)==len(numeric)
    for prompt,row in zip(expected,numeric):
        assert all(row[k]==v for k,v in prompt.items())
        raw=row['raw_probabilities'];assert len(raw)==2 and math.isclose(sum(raw),1,abs_tol=1e-6)
        assert row['raw_legal']==(raw[0]>=raw[1])
        p=probability(model,row['hidden']);assert math.isclose(p,row['calibrated_probability'],abs_tol=1e-7)
        assert row['calibrated_legal']==(p>=.5)
    assert saved['numeric_raw']==legality_metrics([{'legal':r['raw_legal'],'expert_legal':r['target']} for r in numeric])
    assert saved['numeric_calibrated']==legality_metrics([{'legal':r['calibrated_legal'],'expert_legal':r['target']} for r in numeric])
    assert saved['inference_ms']==sum(r['inference_ms'] for r in records)
    return {'fresh_questions':len(tasks),'fresh_prefixes':len(records),'numeric_cases':len(numeric),'paired_searches':len(paired)}
