"""Artifact replay verification independent of report generation; no fabricated scores."""
import csv,json,hashlib,statistics
from pathlib import Path
from .dataset import ROOT,load_dataset,unpack
from .domain import digest
from .search import search
from .evaluate import summarize
OUT=ROOT/'outputs/lab03'

def verify():
    data=load_dataset();models=json.loads((OUT/'models.json').read_text(encoding='utf-8'));ga=json.loads((OUT/'ga_history.json').read_text(encoding='utf-8'))
    tasks={q.id:(g,q,r['seed']) for r in data for g,qs in [unpack(r)] for q in qs}
    with (OUT/'runs.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==3888
    expected_fields=['valid','found','status','expanded','total_work','successor_checks','hops','redundancy_ratio','redundant_path_ratio','expert_rejections']
    for row in rows:
        g,q,seed=tasks[row['query_id']];w=ga[str(seed)]['weights'][:];w[5]=1.5
        actual=search(g,q,row['method'],int(row['budget']),w,models[str(seed)],seed,trace=False)
        for k in expected_fields:
            target=row[k]
            if isinstance(actual[k],bool):assert str(actual[k])==target,(q.id,k)
            elif actual[k] is None:assert target=='',(q.id,k)
            elif isinstance(actual[k],str):assert actual[k]==target,(q.id,k)
            else:assert abs(actual[k]-float(target))<1e-10,(q.id,k)
    cache=json.loads((OUT/'llm_cache.json').read_text(encoding='utf-8'));llm_rows=json.loads((OUT/'llm_runs.json').read_text(encoding='utf-8'))
    for row in llm_rows:
        g,q,seed=tasks[row['query_id']];record=cache[q.id]
        assert record['graph_sha256']==digest(g.to_dict()) and record['query_sha256']==digest(q.__dict__)
        w=ga[str(seed)]['weights'][:];w[5]=1.5
        actual=search(g,q,row['method'],row['budget'],w,models[str(seed)],seed,llm_scores=record['scores'])
        for k in expected_fields[:-1]:assert actual[k]==row[k],(q.id,k)
    # Fresh summaries must agree with serialized table values, excluding elapsed time.
    converted=[]
    for row in rows:
        r=dict(row)
        for k in ['seed','budget','expanded','total_work','successor_checks','expert_rejections']:r[k]=int(r[k])
        for k in ['valid','found']:r[k]=r[k]=='True'
        for k in ['redundancy_ratio','redundant_path_ratio','elapsed_ms']:r[k]=float(r[k])
        r['hops']=int(r['hops']) if r['hops'] else None
        converted.append(r)
    actual=summarize(converted);saved=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert actual==saved
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['dataset_sha256']==hashlib.sha256((ROOT/'data/lab03/benchmark.json').read_bytes()).hexdigest()
    for n in ['domain.py','dataset.py','search.py','learning.py','genetic.py','evaluate.py','llm.py']:
        rel='experiments/lab03/'+n
        assert manifest['source_sha256'].get(rel)==hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),rel
    return {'benchmark_runs_replayed':len(rows),'llm_paired_runs_replayed':len(llm_rows),'tables_recomputed':len(saved),'dataset_and_core_hashes':'matched','status':'passed'}

if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))