"""同一无环数据、同一多重路径语义，比较查询耗时；建库另记。"""
import csv, json, platform, sqlite3, statistics, time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import networkx as nx
from experiments.lab01.dataset import synthetic_dataset
from experiments.lab01.kr.knowledge_graph import EquityKG
from experiments.lab01.kr.relational import RelationalStore
from experiments.lab01.reasoning.graph_reasoning import upstream_paths

OUT=Path(__file__).resolve().parents[2]/'outputs'/'lab01'/'reproduced'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for depth in (2,4,6,8):
        data=synthetic_dataset(n_layers=8,width=50,n_persons=30,seed=42)
        t=time.perf_counter(); h=EquityKG(data).relation_graph('HOLDS'); graph_build=(time.perf_counter()-t)*1000
        t=time.perf_counter(); db=RelationalStore(data['holdings']); sql_build=(time.perf_counter()-t)*1000
        targets=[f'L8C{i}' for i in range(20)]
        methods={'图 DFS':lambda c:upstream_paths(h,c,depth),'SQL 递归 CTE':lambda c:db.upstream_recursive(c,depth),'SQL 逐层 JOIN':lambda c:db.upstream_by_joins(c,depth)}
        canon=lambda rows: sorted((p,round(r,12),n) for p,r,n in rows)
        for c in targets:
            expected=canon(methods['图 DFS'](c))
            for fn in methods.values(): assert canon(fn(c))==expected
        for name,fn in methods.items():
            samples=[]
            for _ in range(9):
                t=time.perf_counter()
                for c in targets: fn(c)
                samples.append((time.perf_counter()-t)*1000/len(targets))
            results.append(dict(depth=depth,method=name,median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples),samples_ms=samples,graph_build_ms=graph_build,sql_build_ms=sql_build))
        db.conn.close()
    info=dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),networkx=nx.__version__,sqlite=sqlite3.sqlite_version,seed=42,entities=430,holdings=len(data['holdings']),queries_per_repeat=20,repeats=9,results=results)
    (OUT/'benchmark.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'benchmark.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['depth','method','median_ms','min_ms','max_ms']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in results)
    print(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
