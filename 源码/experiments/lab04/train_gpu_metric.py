"""Fit train-only transforms on the fresh ResNet12, select on validation, then test once."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .data import digest_file, load_split, selection_digest
from .experiment import load_checkpoint, summarize
from .gpu_metric import embed, encode, fit_metric, fuse_features
from .graph_calibration import GraphConfig, predict
from .study_cached_refinement import paired_stats
from .supervised_whitening import fit as fit_transform


def read(path):
    return json.loads(path.read_bytes())


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def reference_rows(artifacts, split, release):
    name = "continued_validation_tasks.jsonl" if split == "val" else "continued_episodes.json"
    path = artifacts / name
    if digest_file(path) != release["files"][name]["sha256"]:
        raise ValueError(f"Reference tasks changed: {name}")
    if split == "val":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]
    return read(path)["episodes"]


def checked_training_run(directory, expected_ids):
    checkpoint = directory / "fresh.pt"
    model, payload = load_checkpoint(checkpoint)
    plan = read(directory / "backbone_plan.json")
    history = read(directory / "backbone_training.json")
    if (payload["metadata"]["plan"] != plan or plan["train_selection_sha256"] != expected_ids
            or [row["epoch"] for row in history] != list(range(1, plan["epochs"] + plan["meta_epochs"] + 1))):
        raise ValueError("The checkpoint, completed epoch log and base split must agree")
    return model


def pair_scores(split, features, rows, configs):
    lookup = {name: i for i, name in enumerate(split.ids)}
    by_shot, evidence = {}, []
    for shot in (1, 5):
        config = GraphConfig(**configs[str(shot)])
        tasks = [row for row in rows if row["shots"] == shot]
        if sorted(row["episode"] for row in tasks) != list(range(600)):
            raise ValueError("Expected the same 600 paired episodes per shot")
        candidate, previous = [], []
        for row in tasks:
            support = [lookup[name] for name in row["support_ids"]]
            query = [lookup[name] for name in row["query_ids"]]
            support_classes = split.labels[support]
            class_order = list(dict.fromkeys(support_classes.tolist()))
            if (len(class_order) != 5 or len(support) != 5 * shot
                    or set(Counter(support_classes).values()) != {shot}
                    or len(set(support + query)) != 5 * (shot + 15) or len(query) != 75):
                raise ValueError("Invalid support/query task")
            labels = np.array([class_order.index(c) for c in support_classes])
            answer = predict(features[support], labels, features[query], config)["predictions"]
            target = np.array([class_order.index(c) for c in split.labels[query]])
            if not np.array_equal(target, row["query_labels"]) or Counter(target) != dict.fromkeys(range(5), 15):
                raise ValueError("Reference query labels do not match the restored images")
            score = float(np.mean(target == answer))
            reference = float(np.mean(target == row["predictions"]))
            if not math.isclose(reference, row["accuracy"]):
                raise ValueError("Reference accuracy does not match its predictions")
            candidate.append(score)
            previous.append(reference)
            evidence.append({"shots": shot, "episode": row["episode"], "predictions": answer.tolist(),
                             "correct": int(np.sum(target == answer)), "reference_correct": int(np.sum(target == row["predictions"]))})
        by_shot[str(shot)] = {"candidate": summarize(candidate), "continued": summarize(previous),
                              "paired": paired_stats(candidate, previous)}
        comparison = by_shot[str(shot)]
        print(split.split, shot, "shot", json.dumps(comparison), flush=True)
    return by_shot, evidence


def evaluate_split(args, name, manifest, artifacts, backbone, head, device, rows, configs, transform, weights):
    split = load_split(args.data_root, name)
    expected = (read(artifacts / "continued_validation.json")["validation_ids_sha256"]
                if name == "val" else read(artifacts / "protocol.json")["test_selection_sha256"])
    if selection_digest(split) != expected or set(split.class_names) != set(manifest[name]["class_counts"]):
        raise ValueError(f"Unexpected {name} split")
    raw, descriptors = encode(backbone, split.images, args.batch_size, device, spatial=any(weights))
    spatial = embed(head, descriptors, device) if descriptors is not None else None
    results = {}
    for weight in weights:
        features = fuse_features(raw.numpy(), transform, spatial, weight)
        results[weight] = pair_scores(split, features, rows, configs)
    return results


def run(args):
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Training steps and batch size must be positive")
    cuda = torch.cuda.is_available()
    device = torch.device(("cuda" if cuda else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not cuda:
        raise RuntimeError("This Python environment has no working CUDA device")
    torch.set_num_threads(2)
    artifacts, out = args.artifacts.resolve(), args.output.resolve()
    if out in (artifacts, args.trained.resolve()) or artifacts in out.parents:
        raise ValueError("Use a separate evaluation output directory")
    out.mkdir(parents=True, exist_ok=True)
    release = read(artifacts / "continued_results.json")
    checkpoint = args.trained / "fresh.pt"
    for name in ("continued.pt", "continued_validation.json", "continued_configuration.json"):
        if digest_file(artifacts / name) != release["files"][name]["sha256"]:
            raise ValueError("Unexpected reference artifact: " + name)
    protocol = read(artifacts / "protocol.json")
    pretrained_config = read(artifacts / "continued_configuration.json")["selected"]
    configs = {s: row["config"] for s, row in pretrained_config.items()}
    if any((row["views"], row["exponent"], row["residual_mix"]) != ("mean", .5, .5)
           for row in pretrained_config.values()):
        raise ValueError("Unexpected reference transform configuration")
    manifest = read(artifacts / "dataset_manifest.json")["splits"]
    source = Path(__file__)
    weights = [0.0, 0.5]
    plan = {"seed": args.seed, "steps": args.steps, "spatial_weight_candidates": weights,
            "batch_size": args.batch_size, "device": str(device), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "backbone_sha256": digest_file(checkpoint), "previous_sha256": release["checkpoint_sha256"],
            "train_selection_sha256": protocol["train_selection_sha256"],
            "source_sha256": {p.name: digest_file(p) for p in (source, source.with_name("gpu_metric.py"))},
            "test_rule": "Select by mean 1/5-shot validation accuracy among candidates gaining >=1pp and positive paired 95% lower bounds at both shots. Test only the locked candidate."}
    if (out / "plan.json").exists() and read(out / "plan.json") != plan:
        raise ValueError("Output contains a different run; choose a new output directory")
    save_json(out / "plan.json", plan)
    if (out / "results.json").is_file():
        results = read(out / "results.json")
        for name, checksum in results["files"].items():
            if digest_file(out / name) != checksum:
                raise ValueError("Saved file changed: " + name)
        print("Completed result:", out / "results.json", flush=True)
        return results
    backbone = checked_training_run(args.trained, protocol["train_selection_sha256"])
    for filename in ("fresh.pt", "backbone_plan.json", "backbone_training.json"):
        shutil.copyfile(args.trained / filename, out / filename)
    training = load_split(args.data_root, "train")
    if (selection_digest(training) != protocol["train_selection_sha256"]
            or dict(zip(training.class_names, np.bincount(training.labels).tolist())) != manifest["train"]["class_counts"]):
        raise ValueError("Unexpected base training split")
    start = time.perf_counter()
    print("Encoding 38,400 training images with", device, flush=True)
    clean = []
    raw, descriptors = encode(backbone, training.images, args.batch_size, device, clean_features=clean)
    clean = torch.cat(clean)
    with torch.inference_mode():
        predictions = backbone.classifier(clean.to(device)).argmax(1).cpu().numpy()
    correlations = np.corrcoef(clean.numpy(), rowvar=False).diagonal(offset=1)[::2]
    diagnostic = {"clean_train_classification_accuracy": float(np.mean(predictions == training.labels)),
                  "mean_adjacent_channel_correlation": float(correlations.mean())}
    del clean
    transform = fit_transform(raw.numpy(), training.labels, strength=1., exponent=.5, residual_mix=.5)
    np.savez_compressed(out / "base_transform.npz", **transform)
    print("Fitting the spatial metric on base classes", flush=True)
    head, losses = fit_metric(descriptors, training.labels, device=device, steps=args.steps, seed=args.seed)
    del descriptors, raw, training
    head_path = out / "spatial_metric.pt"
    temp_head = out / "spatial_metric.pt.tmp"
    torch.save({k: v.cpu() for k, v in head.state_dict().items()}, temp_head)
    temp_head.replace(head_path)
    results = {"plan": plan, "head_sha256": digest_file(head_path), "first_last_train_loss": losses,
               "clean_base_train_diagnostic": diagnostic,
               "fit_seconds": time.perf_counter() - start, "test_evaluated": False}
    def evaluate(name, candidates):
        return evaluate_split(args, name, manifest, artifacts, backbone, head, device,
                              reference_rows(artifacts, name, release), configs, transform, candidates)

    validation = evaluate("val", weights)
    results["val_candidates"] = {str(w): pair[0] for w, pair in validation.items()}
    eligible = [w for w, (scores, _) in validation.items() if all(
        v["paired"]["mean"] >= .01 and v["paired"]["mean"] > v["paired"]["ci95"] for v in scores.values())]
    best = max(eligible or weights, key=lambda w: sum(v["candidate"]["accuracy"] for v in validation[w][0].values()))
    scores, evidence = validation[best]
    results["val"], results["spatial_weight"] = scores, best
    save_json(out / "val_predictions.json", evidence)
    inference_names = ["fresh.pt", "base_transform.npz"]
    if best:
        inference_names.append("spatial_metric.pt")
    lock = {"test_eligible": bool(eligible), "validation": scores, "spatial_weight": best, "configs": configs,
            "plan_sha256": digest_file(out / "plan.json"),
            "inference_files": {n: digest_file(out / n) for n in inference_names}}
    save_json(out / "validation_lock.json", lock)
    if eligible:
        scores, evidence = evaluate("test", [best])[best]
        save_json(out / "test_predictions.json", evidence)
        results["test"], results["test_evaluated"] = scores, True
    results["total_seconds"] = time.perf_counter() - start
    names = ["plan.json", "backbone_plan.json", "backbone_training.json", "fresh.pt", "spatial_metric.pt",
             "base_transform.npz", "validation_lock.json", "val_predictions.json"]
    if results["test_evaluated"]:
        names.append("test_predictions.json")
    results["files"] = {name: digest_file(out / name) for name in names}
    save_json(out / "results.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--trained", type=Path, default=Path(".local/lab04-gpu-backbone"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_fresh"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261126)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
