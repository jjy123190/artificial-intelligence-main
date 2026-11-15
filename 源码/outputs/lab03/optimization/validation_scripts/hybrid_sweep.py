from pathlib import Path
import sys,json,statistics,importlib
root=Path.cwd();sys.path.insert(0,str(root))
# Load function definitions only; never invoke the full sweep at import.
p=root/'outputs/qa/lab03-optimize-20261112/search_sweep.py'
ns={};exec(p.read_text(encoding='utf-8-sig').split('results=[]')[0],ns)
S=ns['S'];L=ns['L'];data=ns['data'];tasks=ns['tasks']
f=ns['feature_variant']('onehop');S.features=f;L.features=f
models={seed:L.fit_classifier(data,seed) for seed in (7,19,43)}
sweep=json.loads((p.parent/'validation-search-sweep.json').read_text(encoding='utf-8'))
selected=next(r for r in sweep if r['mode']=='onehop' and r['ga_budget']==8)
rows=[]
for lam in (0,.25,.5,1,1.5,2,3):
    runs=[]
    for seed,g,q in tasks:
        w=list(selected['weights'][str(seed)]);w[5]=lam
        for budget in (4,8,16):
            r=S.search(g,q,'hybrid',budget,w,models[seed],seed,trace=False)
            runs.append({'seed':seed,'budget':budget,'query':q.id,'valid':r['valid'],'expanded':r['expanded']})
    row={'ml_weight':lam,'budgets':{b:{'valid':sum(r['valid'] for r in runs if r['budget']==b),'expanded':statistics.mean(r['expanded'] for r in runs if r['budget']==b)} for b in (4,8,16)}}
    rows.append(row);print(json.dumps(row),flush=True)
(p.parent/'validation-hybrid-sweep.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
