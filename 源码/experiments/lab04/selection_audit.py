"""Recompute validation-only selection from the original score journal.

The selection flag is not evidence by itself. Verify the full registered grid,
search shortlist, confirmation rule and positive paired interval before release.
This module has no model/data loading dependency.
"""
from __future__ import annotations
import json
import math
import statistics


def require(condition, message):
    if not condition: raise ValueError("Validation selection: " + message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected, name):
    require(finite(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10), "summary mismatch / " + name)


def half_width(values):
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.


def validate_selection(plan, selection, lines):
    require(plan.get('test_accessed') is False and selection.get('test_accessed') is False and selection.get('status')=='complete','unlocked/test-accessed selection')
    grid=plan['candidates']; search_n=plan['search_episodes']; confirm_n=plan['confirmation_episodes']; shortlist_n=plan['shortlist']
    require(isinstance(grid,list) and grid and all(isinstance(c,dict) for c in grid),'invalid candidate grid')
    require(len({json.dumps(c,sort_keys=True) for c in grid})==len(grid),'duplicate candidates')
    require(all(type(v) is int and v>=2 for v in (search_n,confirm_n)) and type(shortlist_n) is int and 1<=shortlist_n<=len(grid),'invalid sampling/shortlist size')
    require(set(selection['shots'])=={'1','5'},'missing/extra shot')
    journal={}
    for line in lines:
        row=json.loads(line) if isinstance(line,str) else line
        require(isinstance(row,dict) and set(row)=={'key','values'} and isinstance(row['key'],str),'invalid journal row')
        require(row['key'] not in journal,'duplicate journal key')
        journal[row['key']]=row['values']
    used=set(); result={}
    def values(key,count):
        require(key in journal,'missing scores / '+key);v=journal[key]
        require(isinstance(v,list) and len(v)==count,'wrong sample count / '+key)
        require(all(finite(n) and 0<=n<=1 and abs(n*75-round(n*75))<1e-8 for n in v),'invalid task accuracy / '+key)
        used.add(key);return v
    def summary(raw,reported,name):
        require(reported['episodes']==len(raw) and type(reported['episodes']) is int,'wrong reported episodes / '+name)
        mean=statistics.mean(raw);ci=half_width(raw)
        close(reported['accuracy'],mean,name+'/mean');close(reported['ci95'],ci,name+'/ci95')
        if 'ci95_low' in reported:close(reported['ci95_low'],max(0.,mean-ci),name+'/low')
        if 'ci95_high' in reported:close(reported['ci95_high'],min(1.,mean+ci),name+'/high')
    def paired(raw,reference,reported,name):
        difference=[a-b for a,b in zip(raw,reference)]
        mean=statistics.mean(difference);ci=half_width(difference)
        close(reported['mean'],mean,name+'/mean');close(reported['ci95'],ci,name+'/ci95')
        for field,count in (('wins',sum(v>0 for v in difference)),('ties',sum(v==0 for v in difference)),('losses',sum(v<0 for v in difference))):
            require(type(reported[field]) is int and reported[field]==count,'paired '+field+' / '+name)
        return mean,ci
    for shot in ('1','5'):
        branch=selection['shots'][shot];candidates=branch['candidates']
        require(isinstance(candidates,list) and len(candidates)==len(grid),'incomplete search grid')
        require([r['index'] for r in candidates]==list(range(len(grid))) and all(type(r['index']) is int for r in candidates),'candidate index/order mismatch')
        references={}
        for phase,n in (('search',search_n),('confirmation',confirm_n)):
            references[phase]=values(f'{shot}/reference/{phase}',n)
            summary(references[phase],branch['reference'][phase],f'{shot}/reference/{phase}')
        for row,candidate in zip(candidates,grid):
            require(row['candidate']==candidate,'candidate differs from registered grid')
            raw=values(f"{shot}/{row['index']}/search",search_n)
            summary(raw,row['search'],f"{shot}/{row['index']}/search")
            paired(raw,references['search'],row['search_paired'],f"{shot}/{row['index']}/search")
        shortlist=sorted(candidates,key=lambda r:(-r['search']['accuracy'],r['index']))[:shortlist_n]
        indices={r['index'] for r in shortlist}
        for row in candidates:
            require(('confirmation' in row)==(row['index'] in indices) and ('confirmation_paired' in row)==(row['index'] in indices),'confirmation not exactly search shortlist')
        for row in shortlist:
            raw=values(f"{shot}/{row['index']}/confirmation",confirm_n)
            summary(raw,row['confirmation'],f"{shot}/{row['index']}/confirmation")
            paired(raw,references['confirmation'],row['confirmation_paired'],f"{shot}/{row['index']}/confirmation")
        winner=max(shortlist,key=lambda r:(r['confirmation_paired']['mean']-r['confirmation_paired']['ci95'],-r['index']))
        require(branch['selected']==winner,'winner does not follow confirmation lower-bound rule')
        promising=winner['confirmation_paired']['mean']-winner['confirmation_paired']['ci95']>0
        require(type(branch['promising']) is bool and branch['promising']==promising,'incorrect improvement gate')
        result[shot]={'selected_index':winner['index'],'promising':promising}
    require(set(journal)==used,'unregistered extra journal entries')
    return {'verified':True,'candidate_evaluations':len(grid)*2,'score_series':len(used),'shots':result}
