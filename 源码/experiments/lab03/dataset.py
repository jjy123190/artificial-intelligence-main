"""Disjoint graph splits with 2–4-hop questions, decoys, and evidence traps."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import csv,json,random
from .domain import Graph,Node,Edge,Query,RELATIONS,enumerate_paths,digest
ROOT=Path(__file__).resolve().parents[2]
SECTORS=['新能源','制造','物流','金融']


def make_graph(seed,index,split,n=48,queries=9,ood=False):
    rng=random.Random(seed*100003+index*31+sum(map(ord,split)))
    prefix=f'{split}-{seed}-{index}'
    nodes=[Node(f'{prefix}-N{i}',f'{SECTORS[i%4]}企业{i:02d}',sector=SECTORS[i%4],listed=rng.random()<.52) for i in range(n)]
    for i in range(4):nodes[i]=Node(nodes[i].id,f'教学投资人{i+1}',kind='Person',sector=SECTORS[i%4])
    edges=[];used=set()
    def add(a,b,rel,valid=True):
        if a==b:return
        if (a,b,rel) in used:
            if valid:
                for i,e in enumerate(edges):
                    if (e.source,e.target,e.relation)==(nodes[a].id,nodes[b].id,rel):
                        edges[i]=Edge(e.id,e.source,e.target,e.relation,max(.75,e.confidence),True,True,e.value)
            return
        used.add((a,b,rel));bad=not valid
        edges.append(Edge(f'{prefix}-E{len(edges):04}',nodes[a].id,nodes[b].id,rel,
          confidence=round(rng.uniform(.68,1.0) if valid else rng.uniform(.25,.95),3),
          verified=not bad or rng.random()<.35,active=not bad or rng.random()<.5,value=round(rng.uniform(.15,.85),2)))
    for i in range(n):
        for _ in range(5 if ood else 4):
            # Homophily is weaker out of distribution; no answer labels are features.
            group=[j for j in range(4,n) if nodes[j].sector==nodes[i].sector]
            b=rng.choice(group) if rng.random() < (.15 if ood else .68) else rng.randrange(4,n)
            add(i,b,'HOLDS' if i<4 else rng.choice(list(RELATIONS)),rng.random()>.18)
    qs=[]
    for j in range(queries):
        hops=2+j%3
        # Reserve one endpoint per question; never overwrite previous target attributes.
        endpoint=n-queries+j
        chain=[j%4]+rng.sample(range(4,n-queries),hops-1)+[endpoint]
        sector=SECTORS[j%4]
        nodes[chain[-1]]=Node(nodes[chain[-1]].id,nodes[chain[-1]].name,sector=sector,listed=True)
        if not ood:
            for a in chain[1:-1]:nodes[a]=Node(nodes[a].id,nodes[a].name,sector=sector,listed=nodes[a].listed)
        rels=('HOLDS',)+tuple(rng.choice(list(RELATIONS)) for _ in range(hops-1))
        for a,b,rel in zip(chain,chain[1:],rels):add(a,b,rel,True)
        q=Query(f'{prefix}-Q{j}',nodes[chain[0]].id,rels,sector,True,
            f'从{nodes[chain[0]].name}出发，依次经过'+ ' → '.join(RELATIONS[r] for r in rels)+f'，可到达哪些{sector}上市企业？请给出一条有效证据路径。')
        qs.append(q)
    # Names reflect final sector attributes; no misleading sector/name mismatch.
    nodes=[Node(n.id,n.name if n.kind=='Person' else f'{n.sector}企业{i:02d}',n.kind,n.sector,n.listed) for i,n in enumerate(nodes)]
    rng.shuffle(edges)
    # Random IDs remove insertion-order clues about planted answer chains.
    edges=[Edge(f'{prefix}-R{i:04}',e.source,e.target,e.relation,e.confidence,e.verified,e.active,e.value) for i,e in enumerate(edges)]
    return Graph(nodes,edges,prefix,split),qs


def build_dataset():
    records=[]
    for seed in [7,19,43]:
        for split,count in [('train',6),('validation',3),('test',4),('ood',2)]:
            for i in range(count):
                graph,queries=make_graph(seed,i,split,n=48 if split!='ood' else 64,ood=split=='ood')
                records.append({'seed':seed,'split':split,'graph':graph.to_dict(),'queries':[asdict(q) for q in queries]})
    return records


def unpack(record):return Graph.from_dict(record['graph']),[Query(**{**q,'relations':tuple(q['relations'])}) for q in record['queries']]


def save_dataset(path):
    data=build_dataset();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'version':1,'description':'教学合成图；不同split的实体完全不重叠；问题为固定关系序列的2–4跳证据检索。','records':data},ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    return data


def load_dataset(path=None):
    path=path or ROOT/'data/lab03/benchmark.json'
    return json.loads(path.read_text(encoding='utf-8'))['records'] if path.exists() else save_dataset(path)


def shared_graph():
    raw=ROOT/'data/raw'
    def rows(n):
        with (raw/(n+'.csv')).open(encoding='utf-8-sig') as f:return list(csv.reader(f))[1:]
    nodes=[Node(r[0],r[1],kind='Person') for r in rows('persons')]
    nodes += [Node(r[0],r[1],sector='综合',listed=r[0] in ['C3','C12']) for r in rows('companies')]
    edges=[]
    for file,rel in [('holdings','HOLDS'),('guarantees','GUARANTEES'),('trades','TRADES')]:
        for r in rows(file):edges.append(Edge(f'S{len(edges)}',r[0],r[1],rel,value=float(r[2])))
    return Graph(nodes,edges,'shared-lab01','shared'),[
       Query('shared-2','P1',('HOLDS','HOLDS'),'综合',True,'从王建国出发，经过两层持股关系，可以到达哪家上市企业？'),
       Query('shared-3','P1',('HOLDS','HOLDS','HOLDS'),'综合',True,'从王建国出发，经过三层持股关系，可以到达哪家上市企业？'),
       Query('shared-trade','P1',('HOLDS','TRADES'),'综合',True,'王建国的直接持股企业，与哪家上市企业有交易关系？'),
    ]