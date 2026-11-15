"""A train-only supervised readout on frozen Qwen evidence representations.

This is NOT raw zero-shot model legality. Original token probabilities remain
reported alongside the calibrated readout and the independent expert labels.
"""
from __future__ import annotations
import hashlib,json,math,time,os
from .dataset import ROOT,load_dataset,unpack
from .artifacts import source_hashes,provenance
READOUT_PROTOCOL='qwen-evidence-linear-readout-v1'
READOUT_L2=.1
OUT=ROOT/'outputs/lab03'

def readout_provenance():
    from .llm import MODEL_ID,REVISION,WEIGHTS_SHA256
    return {'dataset_sha256':provenance()['dataset_sha256'],'source_sha256':source_hashes(),
            'base_model':MODEL_ID,'revision':REVISION,'weights_sha256':WEIGHTS_SHA256}

def fit_from_hidden(hidden,targets):
    import numpy as np
    import torch
    torch.set_num_threads(min(6,os.cpu_count() or 2))
    x=np.asarray(hidden,dtype=np.float32);y=np.asarray(targets,dtype=np.float32)
    if x.ndim!=2 or not len(y) or len(x)!=len(y) or set(y.tolist())!={0.,1.} or not np.isfinite(x).all():
        raise ValueError('LLM校准训练需要有限隐向量和正负训练样本')
    mean=x.mean(0);scale=np.maximum(x.std(0),.05);x=(x-mean)/scale
    X=torch.from_numpy(x);Y=torch.from_numpy(y)
    w=torch.zeros(X.shape[1],requires_grad=True);b=torch.zeros((),requires_grad=True)
    opt=torch.optim.LBFGS([w,b],lr=.3,max_iter=120,line_search_fn='strong_wolfe')
    balance=torch.where(Y>0,len(Y)/(2*Y.sum()),len(Y)/(2*(len(Y)-Y.sum())))
    def closure():
        opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(X@w+b,Y,weight=balance)+READOUT_L2*w.square().sum()/2
        loss.backward();return loss
    opt.step(closure)
    return {'coef':w.detach().tolist(),'intercept':float(b.detach()),'mean':mean.tolist(),'scale':scale.tolist(),
            'l2':READOUT_L2,'hidden_dim':X.shape[1],'trained_unique_prompts':len(Y),
            'positive':int(Y.sum()),'negative':len(Y)-int(Y.sum())}

def probability(model,hidden):
    import numpy as np
    x=np.asarray(hidden,dtype=np.float32)
    if x.shape!=(model['hidden_dim'],) or not np.isfinite(x).all():raise ValueError('LLM证据隐向量不合法')
    z=(x-np.asarray(model['mean'],dtype=np.float32))/np.asarray(model['scale'],dtype=np.float32)
    z=float(z@np.asarray(model['coef'],dtype=np.float32)+model['intercept'])
    return 1./(1.+math.exp(-max(-30.,min(30.,z))))

def validate_readout(model):
    if (not isinstance(model,dict) or model.get('protocol')!=READOUT_PROTOCOL
            or model.get('provenance')!=readout_provenance() or type(model.get('hidden_dim')) is not int
            or model.get('hidden_dim')!=896 or model.get('l2')!=READOUT_L2):
        raise ValueError('LLM证据分类头缺失、过期或与固定Qwen表示不匹配；请重跑lab03 llm')
    size=model['hidden_dim']
    for name in ('coef','mean','scale'):
        xs=model.get(name)
        if not isinstance(xs,list) or len(xs)!=size or any(type(x) not in (float,int) or not math.isfinite(x) for x in xs):
            raise ValueError('LLM证据分类头参数不合法')
    if any(x<=0 for x in model['scale']) or type(model.get('intercept')) not in (int,float) or not math.isfinite(model['intercept']):
        raise ValueError('LLM证据分类头尺度或截距不合法')
    expected=sorted(r['graph']['id'] for r in load_dataset() if r['split']=='train')
    if type(model.get('trained_unique_prompts')) is not int or model['trained_unique_prompts']<2:
        raise ValueError('LLM证据分类头训练记录缺失')
    if model.get('train_graph_ids')!=expected:raise ValueError('LLM证据分类头训练图来源不一致')
    return model

def load_readout():
    path=OUT/'llm_evidence_readout.json'
    try:value=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:raise ValueError('缺少有效的LLM证据分类头；请重跑lab03 llm') from exc
    return validate_readout(value)

def train_readout(records=None):
    from .llm import candidate_actions,prompt_for,legality_prompt_for,evidence_representation
    from .domain import expert_legal,digest
    records=load_dataset() if records is None else records
    started=time.perf_counter();unique={};samples=[];train_ids=[]
    for record in records:
        if record['split']!='train':continue
        graph,queries=unpack(record);train_ids.append(graph.id)
        for query in queries:
            for state,edge in candidate_actions(graph,query):
                prompt=legality_prompt_for(prompt_for(graph,query,edge,state))
                prefix=state.edges+(edge.id,)
                target=all(expert_legal(graph.edges[e]) for e in prefix)
                if prompt not in unique:
                    hidden,raw,tokens=evidence_representation(prompt)
                    unique[prompt]={'prompt':prompt,'hidden':hidden,'target':target,'prompt_tokens':tokens}
                elif unique[prompt]['target']!=target:raise ValueError('相同证据提示出现冲突训练标签')
                samples.append({'graph_id':graph.id,'query_id':query.id,'path_edges':list(prefix),
                                'prompt_sha256':digest(prompt),'target':target})
        print('Readout train graph',graph.id,'unique',len(unique),flush=True)
    if not unique:raise ValueError('LLM证据分类头没有训练图样本')
    rows=list(unique.values());model=fit_from_hidden([r['hidden'] for r in rows],[r['target'] for r in rows])
    model.update({'kind':'frozen_llm_logistic_readout','protocol':READOUT_PROTOCOL,'provenance':readout_provenance(),
                  'train_graph_ids':sorted(train_ids),'training_prefixes':len(samples),
                  'training_elapsed_ms':(time.perf_counter()-started)*1000,
                  'training_model_passes':len(rows),'training_prompt_tokens':sum(r['prompt_tokens'] for r in rows),
                  'training_prompt_sha256':digest([r['prompt'] for r in rows]),
                  'note':'Train图原始证据文本的固定LLM隐藏表示→有监督分类头；不是原始下一token预测，也不是运行时专家标签替代。'})
    OUT.mkdir(exist_ok=True)
    (OUT/'llm_evidence_readout.json').write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'llm_readout_training.json').write_text(json.dumps({'protocol':READOUT_PROTOCOL,'samples':samples,'unique_prompts':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    return model


def verify_readout():
    from .llm import candidate_actions,prompt_for,legality_prompt_for
    from .domain import expert_legal,digest
    from .diagnostics import legality_metrics
    import numpy as np
    model=load_readout();training=json.loads((OUT/'llm_readout_training.json').read_text(encoding='utf-8'))
    expected=[];ordered={}
    for record in load_dataset():
        if record['split']!='train':continue
        graph,queries=unpack(record)
        for query in queries:
            for state,edge in candidate_actions(graph,query):
                prompt=legality_prompt_for(prompt_for(graph,query,edge,state));prefix=state.edges+(edge.id,)
                target=all(expert_legal(graph.edges[e]) for e in prefix)
                ordered.setdefault(prompt,target)
                expected.append({'graph_id':graph.id,'query_id':query.id,'path_edges':list(prefix),'prompt_sha256':digest(prompt),'target':target})
    assert training['samples']==expected
    rows=training['unique_prompts'];assert [(r['prompt'],r['target']) for r in rows]==list(ordered.items())
    assert model['training_prompt_sha256']==digest(list(ordered))
    assert model['training_prefixes']==len(expected) and model['training_model_passes']==len(rows)
    assert model['training_prompt_tokens']==sum(r['prompt_tokens'] for r in rows)
    fit=fit_from_hidden([r['hidden'] for r in rows],[r['target'] for r in rows])
    for key in ('coef','intercept','mean','scale','positive','negative','hidden_dim','l2','trained_unique_prompts'):
        if key in ('coef','intercept','mean','scale'):assert np.allclose(fit[key],model[key],rtol=1e-6,atol=1e-6),key
        else:assert fit[key]==model[key],key
    return {'status':'passed','train_prefixes':len(expected),'train_unique_prompts':len(rows),'hidden_dim':model['hidden_dim']}
