from pathlib import Path
import sys,json,time
root=Path.cwd();sys.path.insert(0,str(root));out=root/'outputs/qa/lab03-refine-20261112'
from experiments.lab03.llm import local_runtime,path_explanation
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.search import search
import torch
T,M=local_runtime();tasks=[]
for seed in (7,19,43):
 record=next(r for r in load_dataset() if r['split']=='validation' and r['seed']==seed)
 g,qs=unpack(record);q=qs[(seed//7)%3];result=search(g,q,'astar_fixed',16)
 assert result['valid'];tasks.append((g,q,result))
rows=[]
for variant in ('free','evidence_restatement','quote_verified'):
 for g,q,result in tasks:
  chain='; '.join(f'{g.nodes[g.edges[e].source].name} --{g.edges[e].relation}--> {g.nodes[g.edges[e].target].name}' for e in result['edge_ids'])
  verified=path_explanation(g,q,result)
  if variant=='free':prompt='请只用中文解释下面已经由程序找到并核验的图谱路径，不新增实体或关系，不作法律判断。两句话以内。已核验路径：'+chain+'。说明它满足指定的关系顺序，且只是教学证据线索。'
  elif variant=='evidence_restatement':prompt='你只负责说明程序已找到的证据链，不解答新问题。请逐跳复述下面的中文证据，不要新增结论、推断因果或判断风险。保留每个公司名称和关系类型。最后说明这是教学图谱中的关系线索。证据：'+verified
  else:prompt='只返回下方已核验的中文证据说明，不要改写，不添加解释，不删减，不输出引号。该文本由搜索程序产生，你不能生成任何新事实。\n'+verified
  chat=T.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True);tokens=T(chat,return_tensors='pt');start=time.perf_counter()
  with torch.inference_mode():tokens_out=M.generate(**tokens,max_new_tokens=240,do_sample=False,pad_token_id=T.eos_token_id)
  text=T.decode(tokens_out[0,tokens['input_ids'].shape[1]:],skip_special_tokens=True)
  r={'query':q.id,'variant':variant,'prompt':prompt,'reply':text,'verified':verified,'exact':text.strip()==verified,'seconds':time.perf_counter()-start};rows.append(r);print(json.dumps(r,ensure_ascii=False),flush=True)
  (out/'validation-explanations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
