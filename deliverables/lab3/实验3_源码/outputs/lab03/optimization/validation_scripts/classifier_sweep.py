from pathlib import Path
import sys,json,statistics,importlib,math
import numpy as np
root=Path.cwd();sys.path.insert(0,str(root))
p=root/'outputs/qa/lab03-optimize-20261112/search_sweep.py';ns={};exec(p.read_text(encoding='utf-8-sig').split('results=[]')[0],ns)
S=ns['S'];L=ns['L'];data=ns['data'];tasks=ns['tasks'];f=ns['feature_variant']('onehop');S.features=f;L.features=f
from experiments.lab03.domain import State,enumerate_paths
sweep=json.loads((p.parent/'validation-search-sweep.json').read_text(encoding='utf-8'));selected=next(r for r in sweep if r['mode']=='onehop' and r['ga_budget']==8)
samples={}
for seed in (7,19,43):
    vals=[]
    for r in data:
        if r['split']!='train' or r['seed']!=seed:continue
        g,qs=ns['unpack'](r)
        for q in qs:
            solutions=enumerate_paths(g,q);positive={s.edges[:i] for s in solutions for i in range(len(s.edges)+1)}
            todo=[State((q.start,))]
            while todo:
                s=todo.pop(); vals.append((f(g,s,q),int(s.edges in positive),q.hops-s.depth))
                todo.extend(g.apply(s,e) for e in g.successors(s,q))
    samples[seed]=vals
results=[]
for terminals in (True,False):
 for balance in (True,False):
  for reg in (.012,.002):
   models={}
   for seed,vals in samples.items():
    kept=[v for v in vals if terminals or v[2]>0];x=np.asarray([v[0] for v in kept]);y=np.asarray([v[1] for v in kept]);w=np.zeros(6);b=0.
    cw=np.where(y==1,len(y)/(2*sum(y)),len(y)/(2*(len(y)-sum(y)))) if balance else np.ones(len(y))
    for _ in range(700):
     pred=1/(1+np.exp(-np.clip(x@w+b,-30,30)));err=(pred-y)*cw;w-=.12*(x.T@err/len(y)+reg*w);b-=.12*err.mean()
    models[seed]={'seed':seed,'coef':w.tolist(),'intercept':float(b)}
   for lam in (.25,1.5,3):
    metrics={}
    for method in ('astar_ml','hybrid'):
     runs=[]
     for seed,g,q in tasks:
      w=list(selected['weights'][str(seed)]);w[5]=lam
      runs.append(S.search(g,q,method,8,w,models[seed],seed,trace=False))
     metrics[method]={'valid':sum(r['valid'] for r in runs),'expanded':statistics.mean(r['expanded'] for r in runs)}
    row={'include_terminals':terminals,'balanced':balance,'l2':reg,'lambda':lam,**metrics};results.append(row);print(json.dumps(row),flush=True)
(p.parent/'validation-classifier-sweep.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
