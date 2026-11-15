"""Inspect whether restrictive next-token classification matches actual bounded model outputs."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03.llm import local_runtime
import torch
T,M=local_runtime();records=[]
# Development values, no test examples.
for c in [.49,.54,.55,.56,.75]:
    for style in ('json','expression','simple'):
        if style=='json':text=f'Return only JSON {{"legal": true}} or {{"legal": false}}. A record is legal if and only if verified=True and active=True and confidence >= 0.55. Record: verified=True, active=True, confidence={c}.'
        elif style=='expression':text=f'What is the result of ({c} >= 0.55)? Output only True or False.'
        else:text=f'Is {c} at least 0.55? Answer only yes or no.'
        chat=T.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True);tokens=T(chat,return_tensors='pt');start=time.perf_counter()
        with torch.inference_mode():
            logits=M(**tokens,logits_to_keep=1).logits[0,-1];best=torch.topk(torch.softmax(logits,dim=0),10)
            reply=M.generate(**tokens,max_new_tokens=32,do_sample=False,pad_token_id=T.eos_token_id)
        r={'confidence':c,'style':style,'prompt':text,'top_tokens':[(T.decode([int(i)]),float(p)) for p,i in zip(best.values,best.indices)],'text':T.decode(reply[0,tokens['input_ids'].shape[1]:],skip_special_tokens=True),'seconds':time.perf_counter()-start}
        records.append(r);print(c,style,r['text'],flush=True)
(Path(__file__).resolve().parents[1]/'validation-generation-diagnostic.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
