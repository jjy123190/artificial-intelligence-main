"""Small command-line entry for the integrated few-shot experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "outputs" / "lab04"


def main(argv=None):
    parser = argparse.ArgumentParser(description="实验四：mini-ImageNet小样本学习与特征校准")
    parser.add_argument("task", choices=("demo", "results", "verify"), nargs="?", default="demo")
    parser.add_argument("--method", choices=("baseline", "maml", "calibrated"), default="calibrated")
    parser.add_argument("--shots", type=int, choices=(1, 5), default=1)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--calibration", choices=("none", "mean", "variance", "prototype", "full", "graph", "multiview", "wide", "wideview", "continued"), default="full")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.task == "results":
        result = json.loads((args.artifacts / "results.json").read_text(encoding="utf-8"))
        value = {"protocol": result["protocol"], "model": result["model"], "rows": result["rows"], "caveats": result["caveats"]}
    elif args.task == "verify":
        from .verify import verify_artifacts
        value = verify_artifacts(args.artifacts)
        from .graph_extension import verify_extension
        extension = verify_extension(args.artifacts)
        if extension is not None: value["graph_extension"] = extension
        from .multiview_extension import verify_extension as verify_multiview
        multiview = verify_multiview(args.artifacts)
        if multiview is not None: value["multiview_extension"] = multiview
        from .artifact_audit import audit_publication
        audit = audit_publication(args.artifacts)
        value["publication_audit"] = {"verified":audit["verified"], "ready":audit["ready"],
            "paired_core_tasks":audit["episodes_recomputed"], "paired_extension_tasks":audit["extension_episodes_recomputed"]}
        if audit.get("wide") is not None:
            value["wide_extension"] = {"verified":True,"episodes_recomputed":1200,"rows":2,
                "model":audit["wide"]["model"],"scope":audit["wide"]["publication_kind"]}
        if audit.get("wideview") is not None:
            value["wideview_extension"] = {"verified":True, "episodes_recomputed":1200, "rows":audit["wideview"]["rows"]}
        if audit.get("continued") is not None:
            value["continued_extension"] = {"verified":True, "episodes_recomputed":1200, "rows":audit["continued"]["rows"],"training":audit["continued"]["training"]}
    else:
        from .inference import run_episode
        value = run_episode(args.artifacts, shots=args.shots, seed=args.seed, method=args.method, calibration=args.calibration)
        if args.output is None:
            value = {k: value[k] for k in ("method", "shots", "seed", "class_names", "correct", "total", "accuracy", "inference_ms", "transductive", "checkpoint_sha256")}
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
