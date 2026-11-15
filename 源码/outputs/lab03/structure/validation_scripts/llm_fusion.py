"""Evaluate neutral-centred, bounded LLM influence on validation only."""
from pathlib import Path
import json,sys,types
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.domain import path_key
from experiments.lab03.search import HYBRID_ML_WEIGHT
OUT=ROOT/'outputs/lab03/structure'
# Execute definitions only, not the candidate sweep.
source=(OUT/'validation_scripts/structure_variants.py').read_text(encoding='utf-8')
ns={'__file__':str(OUT/'validation_scripts/structure_variants.py')};exec(source[:source.index('results=[]')],ns)
variant=next(v for v in json.loads((OUT/'validation-search-variants.json').read_text(encoding='utf-8')) if v['name']=='suffix2-cap2')
def f(*a,**k):return ns['suffix_features'](*a,**k,cap=2)
rows=json.loads((ROOT/'outputs/lab03/optimization/validation-llm-values.json').read_text(encoding='utf-8'))['records']
assert all(r['query'].startswith('validation-') for r in rows)
caches={}
for r in rows:
    caches.setdefault(r['query'],{})[path_key(r['edges'])]=r['value_score'] if r['prediction'] else .25*r['value_score']
results=[]
for style,weight in [('uncentered',2.),('centered',.25),('centered',.5),('centered',1.),('centered',2.)]:
    code=ns['SOURCE'].replace('query, expert, candidates=candidates)','query, expert, candidates=candidates, prepare=prepare_successors)')
    code=code.replace('q.hops-s.depth == 1 and x[1] == 1','q.hops-s.depth in (1,2) and x[1] == 1').replace('return 1.0\n        vals=','return float(base)\n        vals=')
    if style=='centered':
        code=code.replace("if llm and s.edges:base += 2.0*(1-llm.get(path_key(s.edges),.5))",f"if llm and s.edges:base = max(0.,base+{weight}*(.5-llm.get(path_key(s.edges),.5)))")
    m=types.ModuleType('experiments.lab03._llm_validation');m.__package__='experiments.lab03';exec(compile(code,style,'exec'),m.__dict__);m.features=f
    outcomes=[]
    for record in load_dataset():
        if record['split']!='validation':continue
        graph,qs=unpack(record);seed=record['seed'];model=variant['models'][str(seed)]
        for q in qs:
            if q.id not in caches:continue
            w=variant['ga'][str(seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
            before=m.search(graph,q,'hybrid',8,w,model,seed,trace=False)
            after=m.search(graph,q,'hybrid_llm',8,w,model,seed,llm_scores=caches[q.id],trace=False)
            fields=('valid','expanded','total_edge_checks')
            outcomes.append({'query':q.id,'before':{k:before[k] for k in fields},'after':{k:after[k] for k in fields}})
    v={'style':style,'weight':weight,'questions':len(outcomes),'before_valid':sum(r['before']['valid'] for r in outcomes),'after_valid':sum(r['after']['valid'] for r in outcomes),'before_expanded':sum(r['before']['expanded'] for r in outcomes),'after_expanded':sum(r['after']['expanded'] for r in outcomes),'rows':outcomes}
    results.append(v);print({k:v for k,v in v.items() if k!='rows'},flush=True)
(OUT/'validation-llm-fusion.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
