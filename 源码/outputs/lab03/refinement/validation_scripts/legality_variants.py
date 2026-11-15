from pathlib import Path
import sys,json,time
root=Path.cwd();sys.path.insert(0,str(root));out=root/'outputs/qa/lab03-refine-20261112'
from experiments.lab03.llm import local_runtime,EVIDENCE_POLICY,legality_prompt_for
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.llm import candidate_actions,prompt_for
import torch
T,M=local_runtime();ids=[T.encode(x,add_special_tokens=False)[0] for x in ('True','False')]
records=[]
for r in load_dataset():
 if r['split']!='validation' or not r['graph']['id'].endswith('-0'):continue
 g,qs=unpack(r)
 for q in qs[:3]:
  for s,e in candidate_actions(g,q):
   full=g.apply(s,e);edges=[g.edges[k] for k in full.edges]
   records.append({'query':q.id,'edges':list(full.edges),'base_prompt':prompt_for(g,q,e,s),'v':all(e.verified for e in edges),'a':all(e.active for e in edges),'c':min(e.confidence for e in edges),'target':all(e.verified and e.active and e.confidence>=.55 for e in edges)})
# Supplementary deterministic development-only boundary fixtures; NOT held-out results.
for c in (0.,.49,.50,.51,.54,.55,.56,.59,.6,.99,1.):
 for v,a in [(True,True),(False,True),(True,False)]:
  records.append({'query':'development-boundary','edges':[],'v':v,'a':a,'c':c,'target':v and a and c>=.55})

def prompt(r,variant):
 if variant=='current':return (EVIDENCE_POLICY+f'\nverified={r["v"]}; active={r["a"]}; minimum_confidence={r["c"]}\nReturn True if all three conditions hold, otherwise False. Result:')
 if variant=='percent':return (f'Check a path. All three conditions must hold: verified=True AND active=True AND confidence at least 55 percent. Otherwise False.\nverified={r["v"]}; active={r["a"]}; confidence={r["c"]*100:g} percent.\nReturn True or False only. Result:')
 if variant=='integer':return (f'Check a path. All three conditions must hold: verified=True AND active=True AND confidence at least 550 out of 1000. Otherwise False.\nverified={r["v"]}; active={r["a"]}; confidence={r["c"]*1000:g} out of 1000.\nReturn True or False only. Result:')
 if variant=='boundary_examples':return ('Decide if evidence is eligible. True requires verified=True, active=True, and minimum_confidence>=0.55. Any failing condition means False.\nExamples:\nverified=True; active=True; minimum_confidence=0.50 => False\nverified=True; active=True; minimum_confidence=0.60 => True\nverified=False; active=True; minimum_confidence=0.90 => False\nverified=True; active=False; minimum_confidence=0.90 => False\n'+f'verified={r["v"]}; active={r["a"]}; minimum_confidence={r["c"]}\nReturn True or False only. Result:')
 if variant=='separate':return f'Is {r["c"]} greater than or equal to 0.55? Answer True if it is, otherwise False. Return True or False only. Result:'
 raise ValueError(variant)

results=[]
for variant in ('current','percent','integer','boundary_examples','separate'):
 values=[];start=time.perf_counter()
 for r in records:
  text=prompt(r,variant);chat=T.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True);tokens=T(chat,return_tensors='pt')
  with torch.inference_mode():
   # Only the final-token logits are needed; avoid an unnecessary full vocabulary tensor per prompt token.
   logits=M(**tokens,logits_to_keep=1).logits[0,-1];p=torch.softmax(logits[ids],dim=0).tolist()
  target=(r['c']>=.55) if variant=='separate' else r['target']
  values.append({**r,'prompt':text,'target':target,'probabilities':dict(zip(('True','False'),p)),'prediction':p[0]>=p[1]})
 metrics={}
 for subset in ('validation','development-boundary'):
  rows=[r for r in values if r['query'].startswith(subset)];pos=sum(r['target'] for r in rows);neg=len(rows)-pos;tp=sum(r['target'] and r['prediction'] for r in rows);tn=sum(not r['target'] and not r['prediction'] for r in rows)
  metrics[subset]={'n':len(rows),'positive':pos,'negative':neg,'accuracy':(tp+tn)/len(rows),'tpr':tp/max(1,pos),'tnr':tn/max(1,neg),'balanced_accuracy':.5*(tp/max(1,pos)+tn/max(1,neg))}
 row={'variant':variant,'metrics':metrics,'seconds':time.perf_counter()-start,'records':values};results.append(row)
 (out/'validation-legality-refinement.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in row.items() if k!='records'},ensure_ascii=False),flush=True)
