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
    parser.add_argument("--calibration", choices=("none", "mean", "variance", "prototype", "full"), default="full")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.task == "results":
        result = json.loads((args.artifacts / "results.json").read_text(encoding="utf-8"))
        value = {"protocol": result["protocol"], "model": result["model"], "rows": result["rows"], "caveats": result["caveats"]}
    elif args.task == "verify":
        from .data import digest_file
        result = json.loads((args.artifacts / "results.json").read_text(encoding="utf-8"))
        entries = list(result["weights"].values()) + list(result.get("priors", {}).values()) + [result["demo_pool"]]
        checked = []
        for entry in entries:
            file = args.artifacts / entry["path"]
            actual = digest_file(file)
            if actual != entry["sha256"]:
                raise ValueError(f"Hash mismatch: {file.name}")
            checked.append({"file": file.name, "sha256": actual})
        value = {"verified": True, "files": checked}
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
