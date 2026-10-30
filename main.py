"""课程项目入口；所有源码和样例数据均从本文件所在目录加载。

环境：Python 3.11及以上，pip install -r requirements.txt。
核心演示：python main.py lab02 demo（只需标准库）。
完整实验：python scripts/prepare_lab02.py；自动生成结果、图表并核验。
命令参数：python main.py lab02 --help；回归测试：python main.py check。
网页：双击“打开网站.cmd”，首次从前端源码构建需要Node.js 18及以上。
生成的outputs、.state、.venv、web/dist和node_modules均是运行产物。
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    command=args[0] if args else "web"
    if command=="web":
        import argparse
        from backend.serve import serve
        p=argparse.ArgumentParser();p.add_argument("--port",type=int,default=8765);p.add_argument("--open",action="store_true")
        value=p.parse_args(args[1:]);serve(value.port,value.open);return 0
    if command=="check":
        import io, json
        from datetime import datetime, timezone
        suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),top_level_dir=str(ROOT))
        stream=io.StringIO()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        text=stream.getvalue();print(text)
        out=ROOT/"outputs"/"qa";out.mkdir(parents=True,exist_ok=True)
        (out/"test-results.txt").write_text(text,encoding="utf-8")
        (out/"test-summary.json").write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"successful":result.wasSuccessful()},ensure_ascii=False,indent=2),encoding="utf-8")
        return 0 if result.wasSuccessful() else 1
    if command=="lab01":
        if args[1:] and args[1]!="--step":
            if args[1]=="benchmark":
                from experiments.lab01.benchmark import main as run
                run();return 0
            raise ValueError("lab01 supports --step or benchmark")
        from experiments.lab01.main import main as run
        run();return 0
    from experiments.lab02.main import main as lab02
    return lab02(args[1:] if command=="lab02" else args)

if __name__=="__main__":
    raise SystemExit(main())
