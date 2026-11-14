from pathlib import Path
import sys,json,time,statistics
root=Path.cwd();sys.path.insert(0,str(root));out=root/'outputs/qa/lab03-refine-20261112'
from experiments.lab03.llm import local_runtime,legality_prompt_for,prompt_for,candidate_actions
from experiments.lab03.dataset import load_dataset,unpack
import torch
T,M=local_runtime();torch.set_num_threads(6)
tasks=[]
for r in load_dataset():
 if r['split']=='validation' and r['seed']==7:
  g,qs=unpack(r)
  for q in qs[:1]:
   for s,e in candidate_actions(g,q,limit=3):tasks.append(prompt_for(g,q,e,s))
  break
rows=[]
for text in tasks:
 for name,content,letters in [('value',text,('A','B','C')),('legality',legality_prompt_for(text),('True','False'))]:
  chat=T.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True);tokens=T(chat,return_tensors='pt');ids=[T.encode(x,add_special_tokens=False)[0] for x in letters];modes={}
  for fast in (False,True):
   times=[]
   for _ in range(3):
    start=time.perf_counter()
    with torch.inference_mode():logits=M(**tokens,**({'logits_to_keep':1} if fast else {})).logits[0,-1];p=torch.softmax(logits[ids],dim=0)
    times.append(time.perf_counter()-start)
   modes['last_token' if fast else 'full_sequence']={'seconds_median':statistics.median(times),'p':p.tolist()}
  delta=max(abs(a-b) for a,b in zip(modes['last_token']['p'],modes['full_sequence']['p']))
  rows.append({'task':name,'tokens':tokens['input_ids'].shape[1],'max_probability_delta':delta,'same_argmax':max(range(len(ids)),key=lambda i:modes['last_token']['p'][i])==max(range(len(ids)),key=lambda i:modes['full_sequence']['p'][i]),**modes})
  print(name,rows[-1],flush=True)
(out/'validation-last-token-equivalence.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
