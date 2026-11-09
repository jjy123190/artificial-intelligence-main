"""LLM scores candidate actions, never supplies answer IDs or creates graph edges.

Default local backend is Qwen2.5-0.5B-Instruct, pinned to a published revision.
Remote Responses API is optional and uses LAB03_LLM_* environment variables.
No Codex credentials are read by this application.
"""
from __future__ import annotations
import json,math,os,time,urllib.request
from pathlib import Path
from .domain import expert_legal,digest
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
_runtime=None


def prompt_for(graph,query,edge):
    a,b=graph.nodes[edge.source],graph.nodes[edge.target]
    return ('You evaluate one candidate edge in a synthetic corporate knowledge graph. Do NOT answer the question or name any answer entity. '
       'Only rate if this edge is a promising, legally usable search action. A = promising and evidenced; B = uncertain; C = unsafe or poor.\n'
       f'Required relation sequence: {" then ".join(query.relations)}. Desired endpoint: listed company in sector {query.sector}.\n'
       f'Candidate: {a.kind} ({a.sector}) --{edge.relation}--> {b.kind} ({b.sector}); '
       f'target listed={b.listed}; confidence={edge.confidence}; verified={edge.verified}; active={edge.active}.\n'
       'Choose only A, B or C. Rating:')


def local_runtime():
    global _runtime
    if _runtime is None:
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        torch.set_num_threads(min(6,os.cpu_count() or 2))
        local=Path(os.environ.get('LAB03_LOCAL_MODEL',str(Path.home()/'.cache/lab03-models/Qwen2.5-0.5B-Instruct')))
        source=str(local) if (local/'model.safetensors').is_file() else MODEL_ID
        tok=AutoTokenizer.from_pretrained(source,revision=REVISION,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(source,revision=REVISION,local_files_only=True,dtype=torch.float32)
        model.eval();_runtime=(tok,model)
    return _runtime


def local_score(prompt):
    import torch
    tok,model=local_runtime()
    started=time.perf_counter()
    def predict(text,letters):
        chat=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
        tokens=tok(chat,return_tensors='pt')
        with torch.inference_mode():
            logits=model(**tokens).logits[0,-1]
            ids=[tok.encode(s,add_special_tokens=False)[0] for s in letters]
            ps=torch.softmax(logits[ids],dim=0).tolist()
        return ps,tokens['input_ids'].shape[1]
    ps,n=predict(prompt,['A','B','C'])
    legality_prompt=('Assess only evidence eligibility, not the target answer. An edge is eligible iff verified=True AND active=True AND confidence>=0.55. '
                     'Return Y for eligible, N for ineligible.\n'+prompt.split('Candidate:')[1].split('Choose only')[0]+'\nEligibility Y/N:')
    lp,ln=predict(legality_prompt,['Y','N'])
    return {'score':ps[0]+.5*ps[1],'legal_confidence':lp[0],'legal':lp[0]>=lp[1],
            'probabilities':dict(zip(['A','B','C'],ps)),'legality_probabilities':dict(zip(['Y','N'],lp)),
            'legality_prompt':legality_prompt,'label':max(zip(ps,['A','B','C']))[1],
            'backend':'local','model':MODEL_ID,'revision':REVISION,'prompt_tokens':n+ln,
            'inference_ms':(time.perf_counter()-started)*1000,'protocol':'restricted next-token A/B/C value + Y/N legality; no free-form answer generation'}


def remote_score(prompt):
    key=os.environ.get('LAB03_LLM_API_KEY','');model=os.environ.get('LAB03_LLM_MODEL','')
    base=os.environ.get('LAB03_LLM_BASE_URL','https://api.openai.com/v1').rstrip('/')
    if not key or not model:raise ValueError('请配置LAB03_LLM_API_KEY和LAB03_LLM_MODEL')
    schema={'type':'object','properties':{'score':{'type':'number'},'legal':{'type':'boolean'},'reason':{'type':'string'}},'required':['score','legal','reason'],'additionalProperties':False}
    payload={'model':model,'instructions':'You only evaluate a candidate search action. Never provide an answer entity or fabricate an edge. Output the requested numeric evaluation.','input':prompt,
             'text':{'format':{'type':'json_schema','name':'node_evaluation','strict':True,'schema':schema}},'store':False,'max_output_tokens':250}
    req=urllib.request.Request(base+'/responses',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    start=time.perf_counter()
    with urllib.request.urlopen(req,timeout=25) as r:data=json.load(r)
    text=''.join(c.get('text','') for o in data.get('output',[]) for c in o.get('content',[]) if c.get('type')=='output_text')
    value=json.loads(text)
    if set(value)!=set(schema['required']) or not isinstance(value['legal'],bool):raise ValueError('LLM输出结构不合法')
    if type(value['score']) not in (int,float) or not math.isfinite(value['score']) or not 0<=value['score']<=1:raise ValueError('LLM评分越界')
    return {**value,'backend':'remote','model':data.get('model',model),'usage':data.get('usage'),
            'inference_ms':(time.perf_counter()-start)*1000,'protocol':'Responses structured node evaluation'}


def score_actions(graph,query,edges,backend='local'):
    out=[]
    for edge in edges:
        prompt=prompt_for(graph,query,edge)
        result=remote_score(prompt) if backend=='remote' else local_score(prompt)
        out.append({'edge_id':edge.id,'prompt':prompt,'prompt_sha256':digest(prompt),'expert_legal':expert_legal(edge),**result})
    return out


def explain_verified_path(graph,query,result):
    """Natural-language explanation generation happens after search; never yields its answer."""
    if not result.get('valid'):return {'text':'未找到通过证据约束的有效路径，不生成结论解释。','generated':False}
    import torch
    tok,model=local_runtime()
    chain='; '.join(f'{graph.nodes[graph.edges[e].source].name} --{graph.edges[e].relation}--> {graph.nodes[graph.edges[e].target].name}' for e in result['edge_ids'])
    prompt='请只用中文解释下面已经由程序找到并核验的图谱路径，不新增实体或关系，不作法律判断。两句话以内。已核验路径：'+chain+'。说明它满足指定的关系顺序，且只是教学证据线索。'
    text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
    inp=tok(text,return_tensors='pt');start=time.perf_counter()
    with torch.inference_mode():out=model.generate(**inp,max_new_tokens=110,do_sample=False,pad_token_id=tok.eos_token_id)
    reply=tok.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
    return {'text':reply,'generated':True,'model':MODEL_ID,'revision':REVISION,'prompt':prompt,
            'inference_ms':(time.perf_counter()-start)*1000,
            'grounded':all(graph.nodes[n].name in reply for n in result['path']),
            'missing_path_entities':[graph.nodes[n].name for n in result['path'] if graph.nodes[n].name not in reply],
            'note':'模型解释可能不准确；有效答案及路径由搜索程序确定，以原始关系为准。'}