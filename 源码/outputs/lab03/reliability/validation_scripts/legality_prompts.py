"""Validation-only numeric/boolean evidence prompts. Never rule-substitute predictions."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03.llm import local_predict,MODEL_ID,REVISION,legality_prompt_for
from experiments.lab03.diagnostics import legality_metrics
OUT=ROOT/'outputs/lab03/reliability'
old=json.loads((ROOT/'outputs/lab03/refinement/validation-legality-refinement.json').read_text(encoding='utf-8-sig'))[0]
records=[r for r in old['records'] if r['query'].startswith('validation')]
assert len(records)==72 and all(r['query'].startswith('validation') for r in records)
boundary=[]
for c in [0.,.49,.50,.51,.54,.55,.56,.59,.6,.99,1.]:
    for v,a in [(True,True),(False,True),(True,False)]:
        boundary.append({'query':'development-boundary','edges':[],'v':v,'a':a,'c':c,'target':v and a and c>=.55})

def make(r,variant):
    v,a,c=r['v'],r['a'],r['c']
    if variant=='current':
        return ('Is every condition true? verified must be True; active must be True; minimum_confidence must be at least 0.55.\n'+f'verified={v}; active={a}; minimum_confidence={c}'+'\nReturn True if all three conditions hold, otherwise False. Result:',('True','False'))
    if variant=='json_choice':
        return ('Classify a record using the rule: verified is true AND active is true AND confidence >= 0.55. '+
          'A: record passes ALL requirements. B: at least one requirement fails.\n'+
          json.dumps({'verified':v,'active':a,'confidence':c})+'\nSelect A or B only:',('A','B'))
    if variant=='expression':
        return (f'Evaluate the Python Boolean expression:\n({v} and {a} and ({c} >= 0.55))\n'+
                'A = True\nB = False\nThe expression evaluates to (A or B):',('A','B'))
    if variant=='comparison_reverse':
        return ('Evidence passes only if verified=True, active=True, and confidence >= 0.55.\n'+
                f'verified={v}; active={a}; confidence={c}\n'+
                'A = FAILS the evidence rule.\nB = PASSES the evidence rule.\nReturn only A or B:',('B','A'))
    if variant=='yes_no':
        return (f'A record has verified={v}, active={a}, and confidence {c}. The minimum allowed confidence is 0.55. '+
                'Does this record satisfy verified=True AND active=True AND confidence>=0.55? Answer Yes or No only:',('Yes','No'))
    if variant=='decimal_digits':
        return ('Decide all conditions: verified=True AND active=True AND confidence at least 0.550. '+
                'Compare the decimal digits from left to right.\n'+
                f'verified={v}; active={a}; confidence={c:.3f}\n'+
                'A = all conditions hold\nB = any condition fails\nAnswer A or B only:',('A','B'))
    if variant=='fewshot_bool':
        return ('Evaluate each Boolean expression.\n'+
                'True and True and 0.49 >= 0.55: False\nTrue and True and 0.55 >= 0.55: True\n'+
                'False and True and 0.99 >= 0.55: False\nTrue and False and 0.99 >= 0.55: False\n'+
                f'{v} and {a} and {c} >= 0.55:',('True','False'))
    raise ValueError(variant)

variants=['current','json_choice','expression','comparison_reverse','yes_no','decimal_digits','fewshot_bool']
results=[]
for variant in variants:
    start=time.perf_counter();rows=[]
    # A fixed small boundary screen prevents wasting full validation on an
    # obviously degenerate all-positive/all-negative next-token classifier.
    for r in boundary:
        prompt,letters=make(r,variant);p,n,hit=local_predict(prompt,letters)
        rows.append({**r,'prompt':prompt,'letters':list(letters),'p_eligible':p[0],
                     'prediction':p[0]>=p[1],'probabilities':dict(zip(letters,p)),
                     'prompt_tokens':n,'cache_hit':hit})
    screen=legality_metrics([{'legal':r['prediction'],'expert_legal':r['target']} for r in rows])
    run_full=variant=='current' or (screen['eligible_recall']==1. and screen['ineligible_recall']>=.80)
    if run_full:
        for r in records:
            prompt,letters=make(r,variant);p,n,hit=local_predict(prompt,letters)
            rows.append({**r,'prompt':prompt,'letters':list(letters),'p_eligible':p[0],
                         'prediction':p[0]>=p[1],'probabilities':dict(zip(letters,p)),
                         'prompt_tokens':n,'cache_hit':hit})
    validation=[{'legal':r['prediction'],'expert_legal':r['target']} for r in rows if r['query'].startswith('validation')]
    value={'variant':variant,'screen':screen,'validation':legality_metrics(validation) if validation else None,'seconds':time.perf_counter()-start,'rows':rows}
    results.append(value);(OUT/'validation-legality-prompts.json').write_text(json.dumps({'model':MODEL_ID,'revision':REVISION,'results':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k!='rows'},ensure_ascii=False),flush=True)
