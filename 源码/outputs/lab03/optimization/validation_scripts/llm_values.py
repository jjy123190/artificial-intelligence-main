from pathlib import Path
import sys,json,time,statistics
root=Path.cwd();sys.path.insert(0,str(root));out=root/'outputs/qa/lab03-optimize-20261112'
ns={};exec((out/'search_sweep.py').read_text(encoding='utf-8-sig').split('results=[]')[0],ns)
S=ns['S'];L=ns['L'];data=ns['data'];f=ns['feature_variant']('onehop');S.features=f;L.features=f
from experiments.lab03.llm import local_runtime,MODEL_ID,REVISION
from experiments.lab03.domain import path_key
import torch
samples=next(x for x in json.loads((out/'validation-llm-sweep.json').read_text(encoding='utf-8')) if x['variant']=='truefalse')['records']
tok,model=local_runtime();ids=[tok.encode(x,add_special_tokens=False)[0] for x in ['A','B','C']]
values=[]
for r in samples:
    start=time.perf_counter();chat=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True);tokens=tok(chat,return_tensors='pt')
    with torch.inference_mode():p=torch.softmax(model(**tokens).logits[0,-1,ids],dim=0).tolist()
    values.append({**r,'value_probabilities':dict(zip(['A','B','C'],p)),'value_score':p[0]+.5*p[1],'inference_ms':1000*(time.perf_counter()-start)})
    if len(values)%8==0:print('Scored validation prefixes:',len(values),flush=True)
(out/'validation-llm-values.json').write_text(json.dumps({'model':MODEL_ID,'revision':REVISION,'records':values},ensure_ascii=False,indent=2),encoding='utf-8')
models={seed:L.fit_classifier(data,seed) for seed in (7,19,43)}
selected=next(r for r in json.loads((out/'validation-search-sweep.json').read_text(encoding='utf-8')) if r['mode']=='onehop' and r['ga_budget']==8)
tasks=[(seed,g,q) for seed,g,q in ns['tasks'] if q.id in {x['query'] for x in values}]
original_h=S.heuristic;summary=[]
for style in ('old','centered'):
 for alpha in (0,.25,.5,1,2):
    def h(g,s,q,w,model=None,llm=None):
        base=original_h(g,s,q,w,model,None)
        if g.goal(s,q):return 0.
        if llm and s.edges:
            score=llm.get(path_key(s.edges),.5)
            return max(q.hops-s.depth,base+alpha*((1 if style=='old' else .5)-score))
        return base
    S.heuristic=h;runs=[]
    for seed,g,q in tasks:
        scores={path_key(r['edges']):r['value_score']*(1 if r['prediction'] else .25) for r in values if r['query']==q.id}
        w=list(selected['weights'][str(seed)]);w[5]=.25
        r=S.search(g,q,'hybrid_llm',8,w,models[seed],seed,llm_scores=scores,trace=False)
        runs.append({'query':q.id,'valid':r['valid'],'expanded':r['expanded']})
    row={'style':style,'weight':alpha,'n':len(runs),'valid':sum(r['valid'] for r in runs),'expanded':sum(r['expanded'] for r in runs),'runs':runs}
    summary.append(row);print(json.dumps({k:v for k,v in row.items() if k!='runs'}),flush=True)
(out/'validation-llm-fusion.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
