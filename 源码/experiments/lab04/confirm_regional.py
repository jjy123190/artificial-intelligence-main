"""Replay the locked regional choice and its matched control on freshly sampled validation tasks."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from .accuracy90_data import json_from, load_archive
from .data import digest_file, sample_episode
from .evaluate_gpu_tracks import verified_files
from .regional_study import infer, score
from .train_gpu_metric import save_json


def confirm(args):
    torch.set_num_threads(2)
    out = args.output
    plan = {"split": "val", "episodes_per_shot": 600, "seeds": {"1": 18110901, "5": 18110905},
            "study_plan_sha256": digest_file(out / "plan.json"), "control": "augmented",
            "selection": "Descriptive replication on the same 16 validation classes. Keep the existing winner.",
            "source_sha256": digest_file(Path(__file__))}
    file = out / "confirmation_plan.json"
    if file.exists():
        initial = json_from(file)
        if {k: v for k, v in initial.items() if k != "source_sha256"} != {k: v for k, v in plan.items() if k != "source_sha256"}:
            raise ValueError("A different confirmation plan is present")
    else:
        save_json(file, plan)
    if args.plan_only:
        return
    fixed = json_from(out / "plan.json")
    result = json_from(out / "results.json")
    verified_files(out, {"plan.json": plan["study_plan_sha256"], "validation_lock.json": result["files"]["validation_lock.json"]})
    lock = json_from(out / "validation_lock.json")
    verified_files(out, lock["inference_files"])
    original = json_from(args.control / "validation_lock.json")
    verified_files(args.control, original["inference_files"])
    if original["views"] != lock["views"] or original["selected"] != lock["selected"]:
        raise ValueError("Expected identical global views and prototype rules")
    stamps = {"plan_sha256": digest_file(file), "validation_lock_sha256": result["files"]["validation_lock.json"],
              "winner": lock["winner"], "control_sha256": digest_file(args.control / "validation_lock.json"),
              "execution_source_sha256": plan["source_sha256"]}
    summary = out / "confirmation.json"
    if summary.exists():
        saved = json_from(summary)
        if any(saved[k] != v for k, v in stamps.items()):
            raise ValueError("Confirmation and locked models differ")
        verified_files(out, saved["files"])
        return
    val = load_archive(args.data_root, args.artifacts, "val")
    tasks, arrays = {}, {}
    for shot, seed in plan["seeds"].items():
        rng = np.random.default_rng(seed)
        episodes = [sample_episode(val.labels, rng, shots=int(shot), queries=15)
                    for _ in range(plan["episodes_per_shot"])]
        keys = ("support_indices", "support_labels", "query_indices", "query_labels")
        tasks[shot] = tuple(np.stack([ep[k] for ep in episodes]).astype(np.int64, copy=False) for k in keys)
        arrays.update({f"{k}_{shot}": v for k, v in zip(keys, tasks[shot])})
    kernel = lock["kernel"] is not None
    print("Fresh validation tasks, locked winner:", lock["winner"], flush=True)
    if lock["inference_files"] == original["inference_files"]:
        baseline, other = infer(args.control, val, tasks, fixed, standard=True, kernel=kernel)
        winner = other if kernel else baseline
    else:
        baseline, _ = infer(args.control, val, tasks, fixed, standard=True, kernel=False)
        standard, other = infer(out, val, tasks, fixed, standard=not kernel, kernel=kernel)
        winner = other if kernel else standard
    scores = {}
    for s, t in tasks.items():
        arrays[f"augmented_{s}"], arrays[f"winner_{s}"] = baseline[s], winner[s]
        scores[s] = {"winner": score(winner[s], t[3], baseline[s]), "augmented_correct": int((baseline[s] == t[3]).sum())}
    predictions = out / "confirmation_predictions.npz"
    np.savez_compressed(predictions, **arrays)
    save_json(summary, {**stamps, "scores": scores, "files": {predictions.name: digest_file(predictions)},
                        "split_order_sha256": hashlib.sha256("\n".join(val.ids).encode()).hexdigest()})
    print({s: (v["augmented_correct"], v["winner"]["correct"]) for s, v in scores.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--control", type=Path, default=Path("outputs/lab04_gpu_image_augmentation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_regional"))
    parser.add_argument("--plan-only", action="store_true")
    confirm(parser.parse_args())
