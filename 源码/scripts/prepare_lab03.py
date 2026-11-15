"""Generate lab-three runtime results from source, without reports, slides or recordings.

The default initializes classifiers, validation-only GA and the 3,888-run baseline.
--llm explicitly performs real local-model inference; no saved scores are fabricated.
Historical before/after audits need their archived inputs and are not part of setup.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.lab03.artifacts import current_manifest, source_hashes, provenance
from experiments.lab03.service import read_json, valid_model, valid_ga
from experiments.lab03.runtime_integrity import request_scoped

REQUIRED_FILES = (
    "models.json", "ga_history.json", "learning_metrics.json", "runs.csv", "summary.csv",
    "summary.json", "paired_effects.json", "failure_cases.json", "search_examples.json",
    "boundary_cases.json", "manifest.json",
)
LLM_FILES = ("llm_evidence_readout.json", "llm_readout_training.json", "llm_audit.json",
             "llm_cache.json", "llm_runs.json", "llm_summary.json", "llm_diagnostics.json",
             "llm_explanations.json", "hallucination_stress.json")


@request_scoped
def results_present(output: Path = ROOT / "outputs/lab03") -> bool:
    """Read/replay saved evidence; no training, new model inference or file writes."""
    if not all((output / name).is_file() for name in REQUIRED_FILES):
        return False
    try:
        manifest = read_json(output / "manifest.json")
        models = read_json(output / "models.json")
        ga = read_json(output / "ga_history.json")
        if not (current_manifest(manifest) and isinstance(models, dict) and isinstance(ga, dict)
                and all(valid_model(models.get(str(seed)), seed) and valid_ga(ga.get(str(seed)), seed)
                        for seed in (7, 19, 43))):
            return False
        from experiments.lab03.dataset import load_dataset
        from experiments.lab03.learned_integrity import learning_metrics
        from experiments.lab03.runtime_integrity import benchmark_tables
        learning_metrics(read_json(output / "learning_metrics.json"), models, load_dataset())
        benchmark_tables((output / "runs.csv").read_text(encoding="utf-8-sig"),
                         read_json(output / "summary.json"), read_json(output / "paired_effects.json"),
                         read_json(output / "failure_cases.json"), manifest)
        return True
    except (OSError, ValueError, TypeError):
        return False


@request_scoped
def llm_results_present(output: Path = ROOT / "outputs/lab03") -> bool:
    """A cache is ready only when raw predictions, readout and paired runs agree."""
    if not all((output / name).is_file() for name in LLM_FILES):
        return False
    try:
        from experiments.lab03.calibration import validate_readout
        from experiments.lab03.domain import digest
        from experiments.lab03.service import tasks
        from experiments.lab03.runtime_integrity import (audit_index, original_audit, same,
                                                         original_llm_summary)
        readout = validate_readout(read_json(output / "llm_evidence_readout.json"))
        readout_sha = digest(readout)
        summary = read_json(output / "llm_summary.json")
        cache = read_json(output / "llm_cache.json")
        audit = read_json(output / "llm_audit.json")
        models, ga = read_json(output / "models.json"), read_json(output / "ga_history.json")
        if not (isinstance(cache, dict) and isinstance(models, dict) and isinstance(ga, dict)
                and all(valid_model(models.get(str(seed)), seed) and valid_ga(ga.get(str(seed)), seed)
                        for seed in (7, 19, 43))):
            return False
        task_map = tasks()
        selected = {qid: value for qid, value in task_map.items() if value[0].split == 'test'}
        indexed = audit_index(audit)
        if set(cache) != set(selected) or set(indexed) != set(selected):
            return False
        for qid, (graph, query, _) in selected.items():
            record = cache[qid]
            expected = original_audit(indexed[qid], graph, query, readout, readout_sha)
            expected.update({'protocol': summary['protocol'], 'source_sha256': source_hashes(),
                             'graph_sha256': digest(graph.to_dict()), 'query_sha256': digest(query.__dict__),
                             'model': summary['model'], 'revision': summary['revision'],
                             'weights_sha256': summary['weights_sha256'], 'readout_sha256': readout_sha})
            if not isinstance(record, dict) or any(not same(record.get(k), v) for k, v in expected.items()):
                return False
        original_llm_summary(summary, audit, read_json(output / "llm_runs.json"),
                             read_json(output / "llm_explanations.json"), read_json(output / "llm_diagnostics.json"),
                             read_json(output / "hallucination_stress.json"), task_map, cache, models, ga,
                             readout, readout_sha, source_hashes(), provenance())
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查初始化状态，不生成文件")
    parser.add_argument("--force", action="store_true", help="重新计算，不复用已有结果")
    parser.add_argument("--llm", action="store_true", help="明确启用真实本地LLM推断，需要额外依赖和固定模型权重")
    args = parser.parse_args(argv)
    ready = results_present()
    llm_ready = llm_results_present() if args.llm else True
    if args.check:
        return 0 if ready and llm_ready else 1
    regenerated = not ready or args.force
    if regenerated:
        from experiments.lab03.evaluate import run_benchmark
        print("实验三：从训练图/验证图重新拟合模型与GA，再计算基准；不生成报告、PPT或视频。", flush=True)
        run_benchmark()
    else:
        print("实验三基础结果已初始化；需要重算时添加 --force。", flush=True)
    if args.llm and (not llm_ready or regenerated):
        from experiments.lab03.evaluate import run_llm
        run_llm()
    if not results_present() or (args.llm and not llm_results_present()):
        print("实验三初始化未生成完整有效结果。", file=sys.stderr)
        return 1
    print("实验三初始化完成。未执行历史归档比较或生成交付材料。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
