"""Frozen-LLM linear probe trained on existing train graphs, not a rule replacement."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03.llm import local_runtime,candidate_actions,prompt_for,legality_prompt_for
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.diagnostics import legality_metrics
import torch,numpy as np
OUT=ROOT/'outputs/lab03/reliability';T,M=local_runtime();rows=[];cache={};vectors=[]
for record in load_dataset():
    if record['split'] not in ('train','validation'):continue
    graph,queries=unpack(record)
    for q in queries:
        for s,e in candidate_actions(graph,q):
            text=legality_prompt_for(prompt_for(graph,q,e,s));prefix=graph.apply(s,e)
            target=all(graph.edges[k].verified and graph.edges[k].active and graph.edges[k].confidence>=.55 for k in prefix.edges)
            if text not in cache:
                chat=T.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True);tokens=T(chat,return_tensors='pt')
                with torch.inference_mode():h=M.model(**tokens,use_cache=False).last_hidden_state[0,-1].float().numpy().copy()
                cache[text]=len(vectors);vectors.append(h)
            rows.append({'split':record['split'],'graph_id':graph.id,'query_id':q.id,'edges':list(prefix.edges),'prompt':text,'index':cache[text],'target':target})
    print('encoded',record['graph']['id'],'unique',len(cache),flush=True)
# Explicit development boundary fixtures, not hidden held-out claims.
for c in [0.,.49,.50,.51,.54,.55,.56,.59,.6,.99,1.]:
    for v,a in [(True,True),(False,True),(True,False)]:
        text=('Is every condition true? verified must be True; active must be True; minimum_confidence must be at least 0.55.\n'+f'verified={v}; active={a}; minimum_confidence={c}'+'\nReturn True if all three conditions hold, otherwise False. Result:')
        if text not in cache:
            chat=T.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True);tokens=T(chat,return_tensors='pt')
            with torch.inference_mode():h=M.model(**tokens,use_cache=False).last_hidden_state[0,-1].float().numpy().copy()
            cache[text]=len(vectors);vectors.append(h)
        rows.append({'split':'development_boundary','prompt':text,'index':cache[text],'target':v and a and c>=.55})
np.savez_compressed(OUT/'validation-llm-hidden.npz',hidden=np.array(vectors))
(OUT/'validation-llm-hidden-records.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
# No validation labels in fit, deduplicate train prompts to avoid prefix repetition bias.
train=list({r['index']:r for r in rows if r['split']=='train'}.values());hidden=np.array(vectors,dtype=np.float32)
x=hidden[[r['index'] for r in train]];mean=x.mean(0);scale=np.maximum(x.std(0),.05);x=(x-mean)/scale;y=np.array([r['target'] for r in train],dtype=np.float32)
X=torch.from_numpy(x);Y=torch.from_numpy(y);results=[]
for l2 in [.001,.01,.1,1.]:
    w=torch.zeros(X.shape[1],requires_grad=True);b=torch.zeros((),requires_grad=True)
    opt=torch.optim.LBFGS([w,b],lr=.3,max_iter=120,line_search_fn='strong_wolfe')
    balance=torch.where(Y>0, len(Y)/(2*Y.sum()),len(Y)/(2*(len(Y)-Y.sum())))
    def closure():
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(X@w+b,Y,weight=balance)+l2*w.square().sum()/2;loss.backward();return loss
    opt.step(closure)
    metrics={};values=[]
    for r in rows:
        z=((hidden[r['index']]-mean)/scale)@w.detach().numpy()+float(b.detach());p=1/(1+np.exp(-np.clip(z,-30,30)))
        values.append({**r,'probability':float(p),'legal':bool(p>=.5),'expert_legal':r['target']})
    for split in ['train','validation','development_boundary']:metrics[split]=legality_metrics([r for r in values if r['split']==split])
    result={'l2':l2,'trained_prompts':len(train),'hidden_dim':X.shape[1],'metrics':metrics,'coef':w.detach().tolist(),'intercept':float(b.detach()),'mean':mean.tolist(),'scale':scale.tolist(),'predictions':[{k:r[k] for k in ('split','index','probability','legal','expert_legal')} for r in values]}
    results.append(result);print('probe',l2,metrics,flush=True)
    (OUT/'validation-llm-linear-probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
