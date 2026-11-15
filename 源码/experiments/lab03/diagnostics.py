"""Transparent LLM error analysis; controlled cases are never model observations."""
from __future__ import annotations


def legality_metrics(records):
    """Legal is the positive class; report unsafe detection, not just accuracy."""
    if any(type(r.get('legal')) is not bool or type(r.get('expert_legal')) is not bool for r in records):
        raise ValueError('合法性统计需要真实布尔预测和独立专家标签')
    tp=sum(r['legal'] and r['expert_legal'] for r in records)
    tn=sum(not r['legal'] and not r['expert_legal'] for r in records)
    fp=sum(r['legal'] and not r['expert_legal'] for r in records)
    fn=sum(not r['legal'] and r['expert_legal'] for r in records)
    positive, negative = tp+fn, tn+fp
    def ratio(a,b): return a/b if b else None
    sensitivity, specificity = ratio(tp,positive), ratio(tn,negative)
    return {'total':len(records), 'eligible':positive, 'ineligible':negative,
            'true_legal':tp, 'true_illegal':tn, 'false_accept':fp, 'false_reject':fn,
            'accuracy':ratio(tp+tn,len(records)), 'eligible_recall':sensitivity,
            'ineligible_recall':specificity, 'false_accept_rate':ratio(fp,negative),
            'balanced_accuracy':(sensitivity+specificity)/2 if positive and negative else None,
            'always_legal_accuracy':ratio(positive,len(records)),
            'note':'合法为正类；总体一致率受类比例影响，必须同时查看不合法证据检出率和误放行数。'}


def llm_diagnostics(audit, rows):
    records=[r for task in audit for r in task['evaluations']]
    indexed={(r['query_id'],r['method']):r for r in rows}
    if len(indexed)!=len(rows): raise ValueError('LLM配对记录重复')
    paired=[]
    for task in audit:
        qid=task['query_id'];before=indexed[(qid,'hybrid')];after=indexed[(qid,'hybrid_llm')]
        paired.append({'query_id':qid, 'baseline_valid':before['valid'], 'llm_valid':after['valid'],
                       'baseline_expanded':before['expanded'], 'llm_expanded':after['expanded'],
                       'expanded_delta':after['expanded']-before['expanded'],
                       'success_delta':int(after['valid'])-int(before['valid']),
                       'baseline_status':before['status'], 'llm_status':after['status']})
    errors=[{'query_id':task['query_id'], 'path_key':r['path_key'], 'path_edges':r['path_edges'],
             'expected_legal':r['expert_legal'], 'predicted_legal':r['legal'], 'reward':r['score']}
            for task in audit for r in task['evaluations'] if r['legal']!=r['expert_legal']]
    return {'source':'real saved local model evaluations; not controlled injections',
            'legality':legality_metrics(records),
            'calibrated_legality':legality_metrics([{**r,'legal':r['calibrated_legal']} for r in records]) if records and all('calibrated_legal' in r for r in records) else None,
            'calibration_label':'Separate train-only supervised LLM readout; raw token model metrics above retained.', 'legality_errors':errors, 'paired_cases':paired,
            'improved_answers':sum(p['success_delta']>0 for p in paired),
            'lost_answers':sum(p['success_delta']<0 for p in paired),
            'extra_expansion_cases':sum(p['expanded_delta']>0 for p in paired),
            'fewer_expansion_cases':sum(p['expanded_delta']<0 for p in paired),
            'extra_expansions':sum(p['expanded_delta'] for p in paired),
            'interpretation':'有限小样本；合法性误判不等于已经产生错误答案，排序成本增加也不等于答案准确率下降。'}


def controlled_counterexamples():
    """Two minimal, fixed fixtures isolate unsafe evidence and finite-budget harm."""
    from .domain import Graph,Node,Edge,Query,path_key
    from .search import search
    model={'seed':7,'coef':[0.]*6,'intercept':0.}
    weights=[0.]*6  # Synthetic neutral fixture, NOT trained-model performance.
    def run(g,q,scores,expert=True,budget=3,w=weights):
        result=search(g,q,'hybrid_llm',budget,w,model,7,expert=expert,llm_scores=scores,trace=False)
        return {k:result[k] for k in ('valid','found','status','expanded','total_work','edge_ids','expert_rejections')}
    q=Query('controlled','p',('HOLDS','HOLDS'),'制造')
    nodes=[Node('p','起点',kind='Person'),Node('a','安全中间'),Node('b','干扰中间'),
           Node('z','目标公司',sector='制造',listed=True)]
    graph=Graph(nodes,[Edge('good','p','a','HOLDS'),Edge('bad','p','b','HOLDS',verified=False),
                       Edge('az','a','z','HOLDS'),Edge('bz','b','z','HOLDS')])
    scores={path_key(['good']):0.,path_key(['bad']):1.}
    ablation={'expert_on':run(graph,q,scores), 'expert_off':run(graph,q,scores,False)}
    # Outside the two-hop observation horizon, fabricated optimism can still
    # consume a finite budget despite perfectly valid evidence and expert rules.
    nodes += [Node('c','第二层',sector='制造'),Node('d','第三层',sector='制造')]
    graph=Graph(nodes,[Edge('a-live','p','a','HOLDS'),Edge('z-dead','p','b','HOLDS'),
                       Edge('middle','a','c','HOLDS'),Edge('next','c','d','HOLDS'),Edge('end','d','z','HOLDS'),
                       Edge('bad-next','b','z','HOLDS')])
    q4=Query('controlled-budget','p',('HOLDS',)*4,'制造')
    neutral={path_key(['a-live']):.5,path_key(['z-dead']):.5}
    wrong={path_key(['a-live']):0.,path_key(['z-dead']):1.}
    fixture_weights=[.1,0.,0.,0.,0.,0.]
    budget={'neutral_scores':run(graph,q4,neutral,budget=5,w=fixture_weights),
            'wrong_scores':run(graph,q4,wrong,budget=5,w=fixture_weights),
            'all_evidence_valid':True, 'budget':5}
    return {'source':'controlled injected scores on fixed tiny graphs; NOT real model predictions',
            'parameters':'neutral synthetic classifier; zero weights for unsafe evidence, weak 0.1 topic weight and 4 hops beyond suffix horizon for budget ordering',
            'unsafe_branch':ablation, 'budget_trap':budget,
            'conclusion':'专家规则拦截不合格证据，但不能保证在合法分支中排序正确；错误评分仍可能浪费预算并丢失可达答案。'}
