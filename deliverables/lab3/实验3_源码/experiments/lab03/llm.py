"""LLM scores candidate actions, never supplies answer IDs or creates graph edges.

Default local backend is Qwen2.5-0.5B-Instruct, pinned to a published revision.
Remote Responses API is optional and uses LAB03_LLM_* environment variables.
No Codex credentials are read by this application.
"""
from __future__ import annotations
import hashlib,json,math,os,re,time,urllib.request
from collections import OrderedDict
from threading import RLock
from pathlib import Path
from .domain import expert_legal,digest,State,Edge,RELATIONS,path_key
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
_runtime=None
_runtime_lock=RLock()
SCORE_PROTOCOL = 'path-prefix-value-and-evidence-v7'
EXPLANATION_PROTOCOL = 'model-selected-evidence-wording-v1'
_prediction_cache = OrderedDict()
_evidence_cache = OrderedDict()
_prediction_lock = RLock()
WEIGHTS_SHA256 = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
EVIDENCE_POLICY = ('Is every condition true? verified must be True; active must be True; '
                   'minimum_confidence must be at least 0.55.')
LEGALITY_TOKENS = ('True', 'False')
_MODEL_MANIFEST = json.loads((Path(__file__).resolve().parents[2]/'data/lab03/llm_model_manifest.json').read_text(encoding='utf-8'))
MODEL_FILE_HASHES = {name:entry['sha256'] for name,entry in _MODEL_MANIFEST['files'].items() if name != 'model.safetensors'}



def legality_prompt_for(prompt):
    """Small-model boolean classification, selected on validation prefixes only.

    Copy the three observed prefix facts. Do not calculate an eligibility label or
    substitute deterministic expert judgments for the model's output.
    """
    match = re.search(r'Prefix verified=(True|False); active=(True|False); minimum confidence=([0-9.eE+-]+)\.\n', prompt)
    if match is None:
        raise ValueError('合法性评估缺少候选路径事实')
    verified, active, confidence = match.groups()
    if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        raise ValueError('合法性评估的置信度不合法')
    return (EVIDENCE_POLICY + '\nverified=' + verified + '; active=' + active +
            '; minimum_confidence=' + confidence +
            '\nReturn True if all three conditions hold, otherwise False. Result:')


def verified_local_source():
    """Never label arbitrary LAB03_LOCAL_MODEL weights as the pinned Qwen model."""
    configured = os.environ.get('LAB03_LOCAL_MODEL')
    local = Path(configured or str(Path.home()/'.cache/lab03-models/Qwen2.5-0.5B-Instruct'))
    if not (local/'model.safetensors').is_file():
        if configured:
            raise ValueError('LAB03_LOCAL_MODEL未指向完整的固定版本Qwen权重目录')
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    with (local/'model.safetensors').open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != WEIGHTS_SHA256:
        raise ValueError('本地模型权重与固定Qwen版本不一致，拒绝错误标注模型身份；请运行scripts/prepare_lab03_model.py')
    for name,expected in MODEL_FILE_HASHES.items():
        path=local/name
        if not path.is_file():
            raise ValueError('固定Qwen版本缺少必要配置/分词器文件：'+name)
        with path.open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:
            raise ValueError('固定Qwen版本配置/分词器校验失败：'+name)
    return str(local.resolve())


def candidate_actions(graph, query, limit=8, max_depth=2, *, account=None):
    """Bounded breadth-first look-ahead; never consults goal labels or an oracle."""
    if type(limit) is not int or not 1 <= limit <= 64 or type(max_depth) is not int or not 1 <= max_depth <= query.hops:
        raise ValueError('LLM前瞻范围不合法')
    todo = [State((query.start,))]
    candidates = []
    while todo and len(candidates) < limit:
        state = todo.pop(0)
        if account is not None:
            account(len(graph.out[state.node]) if state.depth < query.hops else 0)
        for edge in graph.successors(state, query, False):
            candidates.append((state, edge))
            if state.depth + 1 < max_depth:
                todo.append(graph.apply(state, edge))
            if len(candidates) == limit:
                break
    return candidates


def prompt_for(graph, query, edge, state=None):
    state = State((query.start,)) if state is None else state
    if (not graph.valid_prefix(state, query, expert=False) or state.depth >= query.hops
            or not isinstance(edge, Edge) or edge.relation != query.relations[state.depth]
            or edge.target in state.nodes):
        raise ValueError('候选动作必须绑定当前问题的合法结构前缀')
    prefix = graph.apply(state, edge)
    a, b = graph.nodes[edge.source], graph.nodes[edge.target]
    evidence = [graph.edges[e] for e in prefix.edges]
    history = [dict(step=i + 1, source_kind=graph.nodes[e.source].kind,
                    source_sector=graph.nodes[e.source].sector, relation=e.relation,
                    target_kind=graph.nodes[e.target].kind, target_sector=graph.nodes[e.target].sector,
                    target_listed=graph.nodes[e.target].listed, confidence=e.confidence,
                    verified=e.verified, active=e.active) for i, e in enumerate(evidence)]
    return ('You evaluate one candidate PATH PREFIX in a synthetic corporate knowledge graph. Do NOT answer the question or name any answer entity. '
            'Only rate the potential reward of continuing this path. A = promising and evidenced; B = uncertain; C = unsafe or poor.\n'
            f'Required relation sequence: {" then ".join(query.relations)}. Desired endpoint: Company in sector {query.sector}, listed={query.listed}.\n'
            f'Completed steps after action: {prefix.depth}/{query.hops}; remaining relations: {list(query.relations[prefix.depth:])}.\n'
            f'Path prefix facts (not a proposed answer): {json.dumps(history, ensure_ascii=False)}.\n'
            f'Candidate: {a.kind} ({a.sector}) --{edge.relation}--> {b.kind} ({b.sector}); '
            f'target listed={b.listed}; confidence={edge.confidence}; verified={edge.verified}; active={edge.active}. '
            f'Prefix verified={all(e.verified for e in evidence)}; active={all(e.active for e in evidence)}; '
            f'minimum confidence={min(e.confidence for e in evidence)}.\n'
            'Choose only A, B or C. Rating:')


def local_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            source = verified_local_source()
            import torch
            from transformers import AutoTokenizer,AutoModelForCausalLM
            torch.set_num_threads(min(6,os.cpu_count() or 2))
            tok=AutoTokenizer.from_pretrained(source,revision=REVISION,local_files_only=True)
            model=AutoModelForCausalLM.from_pretrained(source,revision=REVISION,local_files_only=True,dtype=torch.float32)
            model.eval();_runtime=(tok,model)
            clear_prediction_caches()
        return _runtime

def local_predict(text, letters):
    """Only final-token logits; bounded in-memory memo of identical requests.

    Key includes runtime identity, token options and the complete prompt. Reuse
    is not a model call; cache hits and actual processed token counts are explicit.
    The expert's eligibility decision is never an input to this function.
    """
    import torch
    tok, model = local_runtime()
    key = (id(tok), id(model), SCORE_PROTOCOL, text, tuple(letters))
    with _prediction_lock:
        if key in _prediction_cache:
            value = _prediction_cache.pop(key)
            _prediction_cache[key] = value
            return value[0][:], value[1], True
    chat = tok.apply_chat_template([{'role':'user','content':text}], tokenize=False, add_generation_prompt=True)
    tokens = tok(chat, return_tensors='pt')
    encoded = [tok.encode(letter, add_special_tokens=False) for letter in letters]
    if any(len(ids)!=1 for ids in encoded) or len({ids[0] for ids in encoded})!=len(letters):
        raise ValueError('受限评分选项必须对应互不相同的单个token')
    with torch.inference_mode():
        logits = model(**tokens, logits_to_keep=1).logits[0,-1]
        ps = torch.softmax(logits[[ids[0] for ids in encoded]], dim=0).tolist()
    count = tokens['input_ids'].shape[1]
    with _prediction_lock:
        _prediction_cache[key] = (ps[:], count)
        while len(_prediction_cache)>256:
            _prediction_cache.popitem(last=False)
    return ps, count, False


def clear_prediction_caches():
    """Separate fitting/development from measured inference, including identical facts."""
    with _prediction_lock:
        _prediction_cache.clear()
        _evidence_cache.clear()


def evidence_representation(text):
    """Frozen Qwen last hidden vector and raw token probabilities in ONE pass.

    Raw facts in, no expert labels. The supervised readout is applied separately.
    The same LM head is used for the original True/False comparison, retained in
    all records so a trained readout is never mislabelled as zero-shot capability.
    """
    import torch
    tok,model=local_runtime()
    chat=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    tokens=tok(chat,return_tensors='pt')
    encoded=[tok.encode(letter,add_special_tokens=False) for letter in LEGALITY_TOKENS]
    if any(len(ids)!=1 for ids in encoded) or len({ids[0] for ids in encoded})!=len(encoded):
        raise ValueError('合法性选项必须对应不同的单个token')
    with torch.inference_mode():
        hidden=model.model(**tokens,use_cache=False).last_hidden_state[0,-1].float()
        logits=model.lm_head(hidden)
        probabilities=torch.softmax(logits[[ids[0] for ids in encoded]],dim=0).tolist()
    return hidden.tolist(),probabilities,tokens['input_ids'].shape[1]


def evidence_prediction(text):
    tok,model=local_runtime();key=(id(tok),id(model),SCORE_PROTOCOL,text)
    with _prediction_lock:
        if key in _evidence_cache:
            h,p,n=_evidence_cache.pop(key);_evidence_cache[key]=(h,p,n)
            return h[:],p[:],n,True
    hidden,probabilities,count=evidence_representation(text)
    with _prediction_lock:
        _evidence_cache[key]=(hidden[:],probabilities[:],count)
        while len(_evidence_cache)>256:_evidence_cache.popitem(last=False)
    return hidden,probabilities,count,False


def local_score(prompt,readout=None):
    local_runtime()  # Loading cost is separate from prediction, as before.
    started=time.perf_counter()
    ps,n,value_hit=local_predict(prompt,('A','B','C'))
    legality_prompt=legality_prompt_for(prompt)
    calibration={}
    if readout is None:
        lp,ln,legal_hit=local_predict(legality_prompt,LEGALITY_TOKENS)
    else:
        from .calibration import probability,READOUT_PROTOCOL
        hidden,lp,ln,legal_hit=evidence_prediction(legality_prompt)
        score=probability(readout,hidden)
        calibration={'calibrated_legal':score>=.5,'calibrated_confidence':score,
                     'readout_protocol':READOUT_PROTOCOL,'readout_sha256':digest(readout),
                     'evidence_hidden':hidden,
                     'calibration_note':'有监督LLM表示分类头，训练图拟合；原始模型预测另列，不替代为专家标签。'}
    return {'score':ps[0]+.5*ps[1],'legal_confidence':lp[0],'legal':lp[0]>=lp[1],
            'probabilities':dict(zip(['A','B','C'],ps)),'legality_probabilities':dict(zip(LEGALITY_TOKENS,lp)),
            'legality_prompt':legality_prompt,'legality_tokens':list(LEGALITY_TOKENS),'label':max(zip(ps,['A','B','C']))[1],
            'backend':'local','model':MODEL_ID,'revision':REVISION,'weights_sha256':WEIGHTS_SHA256,'prompt_tokens':n+ln,
            'prediction_cache_hits':int(value_hit)+int(legal_hit),
            'processed_prompt_tokens':(0 if value_hit else n)+(0 if legal_hit else ln),
            'inference_ms':(time.perf_counter()-started)*1000,'protocol':SCORE_PROTOCOL,**calibration}


def remote_score(prompt):
    key=os.environ.get('LAB03_LLM_API_KEY','');model=os.environ.get('LAB03_LLM_MODEL','')
    base=os.environ.get('LAB03_LLM_BASE_URL','https://api.openai.com/v1').rstrip('/')
    if not key or not model:raise ValueError('请配置LAB03_LLM_API_KEY和LAB03_LLM_MODEL')
    schema={'type':'object','properties':{'score':{'type':'number','minimum':0,'maximum':1},'legal':{'type':'boolean'},'reason':{'type':'string','enum':['promising','uncertain','unsafe']}},'required':['score','legal','reason'],'additionalProperties':False}
    instructions=('You only evaluate a candidate search PATH PREFIX. Never provide an answer entity or fabricate an edge. '
                  + EVIDENCE_POLICY + ' Return legal=true only when the entire prefix meets this policy. '
                  'Return score in [0,1] for continuation reward (A/promising=1, B/uncertain=0.5, C/unsafe=0). '
                  'Return reason as promising, uncertain, or unsafe; no free text or answer field. '
                  'Use the JSON schema, not a single-letter response.')
    remote_prompt=prompt.replace('Choose only A, B or C. Rating:',
                                 'Return only score, legal and reason using the required JSON schema.')
    payload={'model':model,'instructions':instructions,'input':remote_prompt,
             'text':{'format':{'type':'json_schema','name':'node_evaluation','strict':True,'schema':schema}},'store':False,'max_output_tokens':250}
    req=urllib.request.Request(base+'/responses',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    start=time.perf_counter()
    with urllib.request.urlopen(req,timeout=18) as r:data=json.load(r)
    if not isinstance(data, dict) or not isinstance(data.get('output'), list):
        raise ValueError('LLM响应格式不合法')
    if data.get('status') not in (None, 'completed') or data.get('error') or data.get('incomplete_details'):
        raise ValueError('LLM响应未完成或包含错误，不能用于搜索评分')
    texts=[]
    try:
        for item in data['output']:
            if item.get('status') not in (None, 'completed'):
                raise ValueError('LLM消息未完成，不能用于搜索评分')
            for content in item.get('content', []):
                if content.get('type') == 'refusal':
                    raise ValueError('LLM拒绝评估，不能把拒绝当作有效分数')
                if content.get('type') == 'output_text':
                    if not isinstance(content.get('text'), str):
                        raise ValueError('LLM评分文本不合法')
                    texts.append(content['text'])
    except (AttributeError, TypeError) as exc:
        raise ValueError('LLM响应内容不合法') from exc
    if len(texts) != 1:
        raise ValueError('LLM必须返回唯一的结构化评估，不接受缺失或多个评估')
    value=json.loads(texts[0])
    if (not isinstance(value, dict) or set(value) != set(schema['required'])
            or type(value['legal']) is not bool or value['reason'] not in ['promising', 'uncertain', 'unsafe']):
        raise ValueError('LLM输出结构不合法；不得返回答案实体或自由文本')
    if type(value['score']) not in (int,float) or not math.isfinite(value['score']) or not 0<=value['score']<=1:raise ValueError('LLM评分越界')
    return {**value,'backend':'remote','model':data.get('model',model),'usage':data.get('usage'),
            'inference_ms':(time.perf_counter()-start)*1000,'protocol':SCORE_PROTOCOL,
            'instructions':instructions,'request_prompt':remote_prompt}


def score_actions(graph, query, candidates, backend='local'):
    if backend not in {'local', 'remote'}:
        raise ValueError('未知LLM后端')
    candidates=list(candidates)
    from .calibration import load_readout
    readout=load_readout() if backend=='local' and candidates else None
    out = []
    for state, edge in candidates:
        prompt = prompt_for(graph, query, edge, state)
        prefix = graph.apply(state, edge)
        result = remote_score(prompt) if backend == 'remote' else local_score(prompt,readout=readout)
        if (type(result.get('score')) not in (int, float) or not math.isfinite(result['score'])
                or not 0 <= result['score'] <= 1 or type(result.get('legal')) is not bool):
            raise ValueError('LLM候选评分不合法')
        if readout is not None and (type(result.get('calibrated_legal')) is not bool
                or type(result.get('calibrated_confidence')) not in (int,float)
                or not math.isfinite(result['calibrated_confidence']) or not 0<=result['calibrated_confidence']<=1
                or result.get('readout_sha256')!=digest(readout)):
            raise ValueError('LLM有监督证据评估缺失或不合法；不得回退为原始模型冒充校准')
        out.append({'edge_id': edge.id, 'path_edges': list(prefix.edges), 'parent_edges': list(state.edges),
                    'path_key': path_key(prefix.edges), 'prompt': prompt, 'prompt_sha256': digest(prompt),
                    'expert_legal': all(expert_legal(graph.edges[e]) for e in prefix.edges), **result})
    return out


def bound_scores(records):
    # Raw remote prediction or explicit supervised local readout; never expert_legal.
    return {r['path_key']: r['score'] if r.get('calibrated_legal',r['legal']) else .25*r['score'] for r in records}


def path_explanation(graph, query, result):
    """Only verified triples determine the authoritative explanation."""
    try:
        state = State(tuple(result['path']), tuple(result['edge_ids']))
    except (KeyError, TypeError):
        return ''
    if not result.get('valid') or not graph.valid_path(state, query):
        return ''
    sentences = [f'第{i + 1}跳：{graph.nodes[e.source].name}通过{RELATIONS[e.relation]}关系连接{graph.nodes[e.target].name}。'
                 for i, e in enumerate(graph.edges[eid] for eid in state.edges)]
    sentences.append(f'该证据链满足{query.hops}跳关系顺序，终点为{query.sector}行业的'
                     f'{"上市" if query.listed else "非上市"}公司；全部边均已核验、有效且置信度不低于0.55。')
    return ''.join(sentences)


def audit_explanation(graph, query, result, text):
    verified = path_explanation(graph, query, result)
    text = text if isinstance(text, str) else ''
    # Merely mentioning every entity does not establish the claimed relations.
    # Only the fully evidence-rendered text is automatically certified; other prose
    # remains visible as unverified model output, never as new graph evidence.
    return {'grounded': bool(verified and isinstance(text, str) and text.strip() == verified),
            'verified_text': verified, 'verification': 'exact evidence-rendered text only; free-form prose is not certified',
            'missing_path_entities': [graph.nodes[n].name for n in result.get('path', [])
                                      if n in graph.nodes and graph.nodes[n].name not in text]}


def controlled_explanation_options(graph, query, result):
    """Finite factual paraphrases, all derived after a fully validated search.

    Choosing wording is a model task, NOT evidence generation or answer selection.
    No free-form sentence is automatically certified by mentioning known entities.
    """
    verified = path_explanation(graph, query, result)
    if not verified:
        return []
    tail = verified[verified.index('该证据链满足'):]
    triples = [(graph.nodes[graph.edges[e].source].name, RELATIONS[graph.edges[e].relation],
                graph.nodes[graph.edges[e].target].name) for e in result['edge_ids']]
    explicit = ''.join(f'第{i+1}条证据：从{a}沿{relation}关系到达{b}。' for i,(a,relation,b) in enumerate(triples)) + tail
    numbered = ''.join(f'第{i+1}跳为{a}—{relation}→{b}。' for i,(a,relation,b) in enumerate(triples)) + tail
    return [verified, explicit, numbered]


def controlled_explanation_prompt(options):
    if len(options)!=3 or any(not isinstance(text,str) or not text for text in options):
        raise ValueError('受控解释必须提供三种已核验的完整证据措辞')
    return ('下面三段中文只是在复述搜索程序已经找到的同一条证据链，不允许产生新答案或新事实。'
            '请选择表述最清晰的一段，只输出A、B或C。\n' +
            '\n'.join(f'{label}: {text}' for label,text in zip('ABC',options)) + '\n选择：')


def controlled_explanation(graph, query, result):
    options = controlled_explanation_options(graph,query,result)
    if not options:
        return {'controlled_text':'','controlled_grounded':False,'controlled_generated':False}
    prompt = controlled_explanation_prompt(options)
    started=time.perf_counter();ps,tokens,hit=local_predict(prompt,('A','B','C'))
    selected=max(range(3),key=lambda i:ps[i])
    return {'controlled_text':options[selected],'controlled_grounded':True,'controlled_generated':True,
            'controlled_choice':'ABC'[selected],'controlled_probabilities':dict(zip('ABC',ps)),
            'controlled_prompt':prompt,'controlled_prompt_sha256':digest(prompt),
            'controlled_inference_ms':(time.perf_counter()-started)*1000,
            'controlled_prediction_cache_hit':hit,'controlled_prompt_tokens':tokens,
            'controlled_protocol':EXPLANATION_PROTOCOL,
            'controlled_note':'真实LLM只选择已核验事实的表达方式；文字来自有限证据模板，不是自由生成事实，也不参与搜索答案。'}


def explain_verified_path(graph,query,result):
    """Natural-language explanation generation happens after search; never yields its answer."""
    verified = path_explanation(graph, query, result)
    if not verified:return {'text':'未找到通过证据约束的有效路径，不生成结论解释。','generated':False,'grounded':False,'verified_text':''}
    import torch
    tok,model=local_runtime()
    chain='; '.join(f'{graph.nodes[graph.edges[e].source].name} --{graph.edges[e].relation}--> {graph.nodes[graph.edges[e].target].name}' for e in result['edge_ids'])
    prompt='请只用中文解释下面已经由程序找到并核验的图谱路径，不新增实体或关系，不作法律判断。两句话以内。已核验路径：'+chain+'。说明它满足指定的关系顺序，且只是教学证据线索。'
    text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
    inp=tok(text,return_tensors='pt');start=time.perf_counter()
    with torch.inference_mode():out=model.generate(**inp,max_new_tokens=110,do_sample=False,pad_token_id=tok.eos_token_id)
    reply=tok.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
    return {'text':reply,'generated':True,'model':MODEL_ID,'revision':REVISION,'weights_sha256':WEIGHTS_SHA256,'prompt':prompt,
            'inference_ms':(time.perf_counter()-start)*1000,
            **audit_explanation(graph, query, result, reply),
            'graph_sha256':digest(graph.to_dict()),'query_sha256':digest(query.__dict__),
            'edge_ids':result['edge_ids'],
            **controlled_explanation(graph,query,result),
            'note':'模型解释可能不准确；有效答案及路径由搜索程序确定，以原始关系为准。'}