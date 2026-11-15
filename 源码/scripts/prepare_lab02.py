"""从本目录源码生成实验二运行数据，不复制或引用别的项目的历史结果。

python scripts/prepare_lab02.py                 # 缺少结果时完整生成
python scripts/prepare_lab02.py --force         # 明确要求重新计算
python scripts/prepare_lab02.py --output-dir PATH

正式参数为3个随机种子；所有模型先学习再评估，图表只读取本次生成的CSV。
运行时间取决于机器；默认不执行缩小规模的冒烟实验。
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lab02.main import main as run_lab02

REQUIRED_FILES = (
    "demo/summary.json", "demo/traces.json", "demo/manifest.json",
    "experiments/summary.csv", "experiments/models.json", "experiments/manifest.json",
    "experiments/sensitivity.csv", "experiments/ablations.csv", "experiments/conflict_summary.csv",
    "end_to_end/summary.csv", "end_to_end/models.json", "end_to_end/paired_summary.csv",
    "end_to_end/manifest.json", "benchmarks/forward_summary.csv", "benchmarks/goal_slicing.csv",
    "benchmarks/resolution.csv", "benchmarks/manifest.json", "governance/equity_summary.csv",
    "governance/manifest.json",
)


def results_present(output: Path) -> bool:
    """这里只判断是否需要首次初始化；结果真伪由独立verify命令核验。"""
    return all((output / relative).is_file() for relative in REQUIRED_FILES)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lab02")
    parser.add_argument("--force", action="store_true", help="即使已有完整结果也重新计算")
    parser.add_argument("--check", action="store_true", help="只检查运行数据是否齐全，不生成文件")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.check:
        return 0 if results_present(output) else 1
    if results_present(output) and not args.force:
        print(f"已有实验结果：{output}；需要重算时添加 --force。", flush=True)
        return 0
    if importlib.util.find_spec("matplotlib") is None:
        print("缺少绘图依赖，请先运行 python -m pip install -r requirements.txt。", file=sys.stderr)
        return 2

    # 独立阶段便于定位失败；不在每个阶段读取其他目录的模型或正式输出。
    stages = ("demo", "experiments", "e2e", "governance", "benchmark", "plots", "verify")
    for index, stage in enumerate(stages, 1):
        print(f"\n[{index}/{len(stages)}] {stage}：正在本机计算，请稍候……", flush=True)
        command = [stage, "--output-dir", str(output)]
        if stage in ("experiments", "e2e", "governance", "benchmark"):
            command.append("--no-plots")
        if stage == "verify":
            command.append("--full-replay")
        status = run_lab02(command)
        if status:
            return status
    print(f"实验数据、模型和图表生成完毕，完整回放通过：{output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
