"""实验三命令行：dataset / benchmark / llm / report / all / demo。"""
from __future__ import annotations
import argparse,json
from .dataset import ROOT,save_dataset

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['dataset','benchmark','llm','report','all','demo'],default='demo',nargs='?')
    args=p.parse_args(argv)
    if args.command=='dataset':save_dataset(ROOT/'data/lab03/benchmark.json')
    if args.command in ['benchmark','all']:
        from .evaluate import run_benchmark
        run_benchmark()
    if args.command in ['llm','all']:
        from .evaluate import run_llm
        run_llm()
    if args.command in ['report','all']:
        from .report import build_report
        build_report()
    if args.command=='demo':
        from .service import run_task
        r=run_task({'query_id':'shared-3','method':'hybrid','budget':16})
        print(json.dumps({k:v for k,v in r.items() if k!='graph'},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())