"""A small, serializable logistic path-value classifier; labels use training graphs only."""
from __future__ import annotations
import math,random
from .domain import State,features,enumerate_paths,FEATURE_NAMES
from .dataset import unpack

def sigmoid(x):return 1/(1+math.exp(-max(-30,min(30,x))))

def probability(model,x):
    if not model:return .5
    return sigmoid(sum(w*v for w,v in zip(model['coef'],x))+model['intercept'])

def fit_classifier(records,seed):
    import numpy as np
    samples=[];ids=[]
    for record in records:
        if record['split']!='train' or record['seed']!=seed:continue
        graph,queries=unpack(record);ids.append(graph.id)
        for q in queries:
            solutions=enumerate_paths(graph,q)
            positive={s.edges[:i] for s in solutions for i in range(len(s.edges)+1)}
            todo=[State((q.start,))]
            while todo:
                state=todo.pop()
                samples.append((features(graph,state,q),int(state.edges in positive)))
                if state.depth<q.hops:
                    todo.extend(graph.apply(state,e) for e in graph.successors(state,q,True))
    x=np.asarray([s[0] for s in samples],dtype=float);y=np.asarray([s[1] for s in samples],dtype=float)
    w=np.zeros(x.shape[1]);b=0.
    positives=int(y.sum());negatives=len(y)-positives
    weight=np.where(y==1,len(y)/(2*max(1,positives)),len(y)/(2*max(1,negatives)))
    for _ in range(700):
        pred=1/(1+np.exp(-np.clip(x@w+b,-30,30)))
        err=(pred-y)*weight
        w-=.12*(x.T@err/len(y)+.012*w);b-=.12*err.mean()
    preds=1/(1+np.exp(-np.clip(x@w+b,-30,30)))
    return {'kind':'logistic_regression','coef':w.tolist(),'intercept':float(b),'features':FEATURE_NAMES,
            'seed':seed,'train_graph_ids':ids,'samples':len(y),'positive':positives,'negative':negatives,
            'train_brier':float(np.mean((preds-y)**2)), 'label':'state has at least one valid goal continuation in training graph',
            'note':'训练图穷举产生正负路径前缀；不使用节点编号、目标答案或测试标签作为特征。'}


def classifier_evaluation(records,model,split='test'):
    ys=[];ps=[]
    for record in records:
        if record['seed']!=model['seed'] or record['split']!=split:continue
        graph,queries=unpack(record)
        for q in queries:
            sol=enumerate_paths(graph,q);positive={s.edges[:i] for s in sol for i in range(len(s.edges)+1)}
            todo=[State((q.start,))]
            while todo:
                s=todo.pop();ys.append(int(s.edges in positive));ps.append(probability(model,features(graph,s,q)))
                todo.extend(graph.apply(s,e) for e in graph.successors(s,q,True))
    tp=sum(y and p>=.5 for y,p in zip(ys,ps));fp=sum(not y and p>=.5 for y,p in zip(ys,ps));fn=sum(y and p<.5 for y,p in zip(ys,ps))
    return {'split':split,'samples':len(ys),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),
            'brier':sum((p-y)**2 for y,p in zip(ys,ps))/max(1,len(ys))}