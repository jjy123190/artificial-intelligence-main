"""Compare one-pass hidden/head path with original last-token logits on validation only."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03.llm import evidence_representation,local_predict,LEGALITY_TOKENS
from experiments.lab03.calibration import fit_from_hidden,probability
import numpy as np
OUT=ROOT/'outputs/lab03/reliability'
records=json.loads((OUT/'validation-llm-hidden-records.json').read_text(encoding='utf-8'))
vectors=np.load(OUT/'validation-llm-hidden.npz')['hidden']
train=list({r['index']:r for r in records if r['split']=='train'}.values())
fit=fit_from_hidden(vectors[[r['index'] for r in train]],[r['target'] for r in train])
previous=next(p for p in json.loads((OUT/'validation-llm-linear-probe.json').read_text(encoding='utf-8')) if p['l2']==.1)
assert fit['coef']==previous['coef'] and fit['intercept']==previous['intercept']
validation=list({r['index']:r for r in records if r['split']=='validation'}.values())
# Fixed first 12 distinct validation texts, no held-out prompt selection.
rows=[]
for r in validation[:12]:
    hidden,p,n=evidence_representation(r['prompt']);q,m,_=local_predict(r['prompt'],LEGALITY_TOKENS)
    difference=max(abs(a-b) for a,b in zip(p,q))
    assert n==m and (p[0]>=p[1])==(q[0]>=q[1]) and difference<1e-5
    assert max(abs(a-float(b)) for a,b in zip(hidden,vectors[r['index']]))<1e-5
    rows.append({'prompt':r['prompt'],'raw_max_probability_difference':difference,'calibrated_probability':probability(fit,hidden)})
value={'status':'passed','readout_fit_exact':True,'raw_argmax_unchanged':True,'rows':rows,'max_difference':max(r['raw_max_probability_difference'] for r in rows)}
(OUT/'validation-onepass-equivalence.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in value.items() if k!='rows'})
