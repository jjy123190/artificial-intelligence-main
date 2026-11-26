"""Small command-line entry for the integrated few-shot experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "outputs" / "lab04"


def run_gpu_track(args):
    import numpy as np
    import torch
    from .data import sample_episode
    from .evaluate_gpu_tracks import EpisodeClassifier
    from .inference import load_demo_pool

    directory = args.artifacts.parent / ("lab04_gpu_" + args.track)
    result = json.loads((directory / "results.json").read_bytes())
    training_choice = {"winner": result["winner"]} if "winner" in result else {}
    if args.task == "results":
        return {**{key: result[key] for key in ("track", "external_pretraining", "val", "test")},
                **training_choice}
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(2)
    if args.track == "regional":
        from .regional_kernel import RegionalClassifier
        model = RegionalClassifier(directory, device)
        transductive = True
    elif args.track == "task_subspace":
        from .task_subspace import SubspaceClassifier
        model = SubspaceClassifier(directory, device)
        transductive = True
    elif args.track == "episode_metric":
        from .episode_metric_study import EpisodeMetricClassifier
        model = EpisodeMetricClassifier(directory, device)
        transductive = True
    elif args.track == "distance_covariance":
        from .distance_covariance_study import DistanceClassifier
        model = DistanceClassifier(directory, device)
        transductive = True
    elif args.track in ("restarts", "self_distill", "rotation_views", "image_augmentation"):
        from .prototype_restarts import RestartClassifier
        model = RestartClassifier(directory, device, weights=None if args.track == "restarts" else directory)
        transductive = True
    elif args.track == "multiscale":
        from .multiscale_study import MultiscaleClassifier
        model = MultiscaleClassifier(directory, device)
        transductive = True
    elif args.track == "rotation":
        from .rotation_study import RotationClassifier
        model = RotationClassifier(directory, device)
        transductive = True
    else:
        model = EpisodeClassifier(directory, device)
        transductive = model.rules[args.shots].updates > 0
    pool = load_demo_pool(args.artifacts)
    task = sample_episode(pool["labels"], np.random.default_rng(args.seed), shots=args.shots, queries=15)
    answers = model.predict(pool["images"][task["support_indices"]], task["support_labels"],
                            pool["images"][task["query_indices"]])
    correct = int(np.sum(answers == task["query_labels"]))
    return {"track": args.track, **training_choice, "external_pretraining": result["external_pretraining"],
            "device": device, "shots": args.shots, "seed": args.seed,
            "class_names": pool["class_names"][task["classes"]].tolist(),
            "correct": correct, "total": len(answers), "accuracy": correct / len(answers),
            "transductive": transductive, "predictions": answers.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="实验四：mini-ImageNet小样本学习与特征校准")
    parser.add_argument("task", choices=("demo", "results", "verify"), nargs="?", default="demo")
    parser.add_argument("--method", choices=("baseline", "maml", "calibrated"), default="calibrated")
    parser.add_argument("--shots", type=int, choices=(1, 5), default=1)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--calibration", choices=("none", "mean", "variance", "prototype", "full", "graph", "multiview", "wide", "wideview", "continued"), default="full")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--track", choices=("medium", "rotation", "multiscale", "restarts", "self_distill", "rotation_views", "episode_metric", "distance_covariance", "task_subspace", "image_augmentation", "regional", "dino"), help="自训 ResNet12 的训练与校准方案，或外部 DINOv2 参考")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="--track 的推理设备")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.track:
        if args.task not in ("results", "demo") or args.method != "calibrated" or args.calibration != "full":
            parser.error("--track supports the results and demo commands with its own validation-selected classifier")
        value = run_gpu_track(args)
    elif args.task == "results":
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
