"""实验三命令行：dataset / benchmark / llm / report / all / demo。"""
from __future__ import annotations
import argparse,json
from .dataset import ROOT,save_dataset

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['dataset','benchmark','llm','report','all','demo'],default='demo',nargs='?')
    p.add_argument('--method', default='astar_fixed', choices=['bfs','astar_admissible','astar_fixed','astar_ga','astar_ml','hybrid','mcts','hybrid_no_expert','hybrid_llm'], help='demo使用的方法；学习/LLM方法需要先初始化产物')
    args=p.parse_args(argv)
    if args.command=='dataset':save_dataset(ROOT/'data/lab03/benchmark.json')
    if args.command in ['benchmark','all']:
        from .evaluate import run_benchmark
        run_benchmark()
    if args.command in ['llm','all']:
        from .evaluate import run_llm
        run_llm()
        # Archived comparisons need their original inputs. A source-only checkout
        # can run real LLM inference without requiring historical ZIP snapshots.
        # When archives have been restored, refresh them after real inference.
        archive=ROOT/'outputs/lab03/reliability/stage_three_baseline.zip'
        if archive.is_file():
            from .reliability import rebuild
            rebuild()
    if args.command in ['report','all']:
        from .report import build_report
        build_report()
    if args.command=='demo':
        from .service import run_task
        r=run_task({'query_id':'shared-3','method':args.method,'budget':16})
        print(json.dumps({k:v for k,v in r.items() if k!='graph'},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())