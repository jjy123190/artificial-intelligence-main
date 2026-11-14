from pathlib import Path
import sys,json,time
root=Path.cwd();sys.path.insert(0,str(root))
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.llm import local_runtime,candidate_actions,prompt_for,legality_prompt_for,MODEL_ID,REVISION
import torch
out=root/'outputs/qa/lab03-optimize-20261112'
tok,model=local_runtime()
records=[]
# Fixed validation selection covers all three seeds, no test/OOD data used.
for seed in (7,19,43):
    r=next(r for r in load_dataset() if r['split']=='validation' and r['seed']==seed)
    g,qs=unpack(r)
    for q in qs[:3]:
        for s,e in candidate_actions(g,q):
            prefix=g.apply(s,e);es=[g.edges[k] for k in prefix.edges]
            records.append({'query':q.id,'edges':list(prefix.edges),'prompt':prompt_for(g,q,e,s),
                            'v':all(e.verified for e in es),'a':all(e.active for e in es),'c':min(e.confidence for e in es),
                            'target':all(e.verified and e.active and e.confidence>=.55 for e in es)})

def prompt(r,variant):
    if variant=='old':return legality_prompt_for(r['prompt'])
    fact=f'verified={r["v"]}; active={r["a"]}; minimum_confidence={r["c"]}'
    if variant=='simple':
        return 'Check evidence only. A path is legal if verified=True AND active=True AND minimum_confidence>=0.55. If ANY condition fails, it is illegal.\n'+fact+'\nAnswer Y if legal, N if illegal. Answer:'
    if variant=='fewshot':
        return ('Classify evidence. All three checks must pass: verified=True, active=True, minimum_confidence>=0.55. Any failure means N, otherwise Y.\n'
                'verified=True; active=True; minimum_confidence=0.80 => Y\n'
                'verified=False; active=True; minimum_confidence=0.80 => N\n'
                'verified=True; active=False; minimum_confidence=0.80 => N\n'
                'verified=True; active=True; minimum_confidence=0.30 => N\n'+fact+' =>')
    if variant=='truefalse':
        return ('Is every condition true? verified must be True; active must be True; minimum_confidence must be at least 0.55.\n'
                +fact+'\nReturn True if all three conditions hold, otherwise False. Result:')
    if variant=='chinese':
        return ('请判断证据链是否合格。三个条件缺一不可：verified为True、active为True、minimum_confidence不低于0.55。只要一项不满足就是不合格。\n'
                +fact+'\n合格只输出Y，不合格只输出N。结论：')
    raise ValueError(variant)

results=[]
for variant in ('old','simple','fewshot','truefalse','chinese'):
    letters=['True','False'] if variant=='truefalse' else ['Y','N']
    ids=[tok.encode(x,add_special_tokens=False) for x in letters]
    assert all(len(x)==1 for x in ids)
    values=[];start=time.perf_counter()
    for r in records:
        text=prompt(r,variant)
        chat=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
        tokens=tok(chat,return_tensors='pt')
        with torch.inference_mode():
            logits=model(**tokens).logits[0,-1]
            p=torch.softmax(logits[[x[0] for x in ids]],dim=0).tolist()
        values.append({**r,'variant_prompt':text,'p':p[0],'prediction':p[0]>=p[1]})
    pos=sum(r['target'] for r in values);neg=len(values)-pos
    tp=sum(r['target'] and r['prediction'] for r in values);tn=sum(not r['target'] and not r['prediction'] for r in values)
    result={'variant':variant,'model':MODEL_ID,'revision':REVISION,'n':len(values),'positive':pos,'negative':neg,'accuracy':(tp+tn)/len(values),
            'tpr':tp/max(1,pos),'tnr':tn/max(1,neg),'balanced_accuracy':.5*(tp/max(1,pos)+tn/max(1,neg)),
            'seconds':time.perf_counter()-start,'records':values}
    results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)
    (out/'validation-llm-sweep.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
