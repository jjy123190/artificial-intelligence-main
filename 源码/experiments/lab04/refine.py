"""Validation-only refinement, locked evaluation, and portable v2 artifacts.

Run cache, maml, tune, evaluate in that order. Test images are never loaded by
cache/maml/tune. Candidate caches are fingerprinted; interrupted jobs resume.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .calibration import CalibrationConfig, base_statistics, calibrated_predict, unit_features
from .data import digest_file, load_split, normalized_tensor, sample_episode, selection_digest
from .experiment import extract_features, load_checkpoint, log, save_checkpoint, summarize, write_json
from .feature_space import fit_feature_space, transform_features
from .models import ResNet12, maml_adapt, maml_logits


def cache_features(args):
    directory = Path(args.output)
    manifest_path = directory / "feature_cache.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for split in ("train", "val"):
        images = None
        for name in ("baseline", "aligned"):
            checkpoint = directory / f"{name}.pt"
            key = f"{name}_{split}"
            fingerprint = digest_file(checkpoint)
            cache = directory / f"{key}_features.npz"
            if (cache.exists() and manifest.get(key, {}).get("checkpoint_sha256") == fingerprint
                    and manifest[key].get("sha256") == digest_file(cache)):
                log("cache_exists", key=key)
                continue
            if images is None:
                images = load_split(args.data_root, split, max_per_class=600, seed=args.seed)
            network, _ = load_checkpoint(checkpoint)
            log("extract", model=name, split=split, images=len(images.images))
            features = extract_features(network, images.images, batch_size=64)
            np.savez_compressed(cache, features=features, labels=images.labels,
                                ids=np.array(images.ids), class_names=np.array(images.class_names))
            manifest[key] = {"checkpoint_sha256": fingerprint, "sha256": digest_file(cache),
                             "images": len(images.images), "selection_sha256": selection_digest(images)}
            write_json(manifest_path, manifest)
            del network, features
        del images
        gc.collect()
    log("cache_complete")


def read_cache(directory, name, split):
    key = f"{name}_{split}"
    path = directory / f"{key}_features.npz"
    manifest = json.loads((directory / "feature_cache.json").read_text())
    if manifest[key]["checkpoint_sha256"] != digest_file(directory / f"{name}.pt"):
        raise ValueError("Stale feature cache checkpoint")
    if manifest[key]["sha256"] != digest_file(path):
        raise ValueError("Feature cache checksum mismatch")
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def validation_partition(labels, seed):
    """Two disjoint image pools: search on 200/class, confirmation on 400/class."""
    rng = np.random.default_rng(seed)
    first, second = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        first.extend(indices[:200]); second.extend(indices[200:])
    return np.array(first), np.array(second)


def tasks_for(labels, seed, shots, episodes):
    rng = np.random.default_rng(seed)
    return [sample_episode(labels, rng, shots=shots, queries=15) for _ in range(episodes)]


def score_configuration(features, labels, prior, tasks, config, flags=None):
    flags = flags or {"mean": True, "variance": True, "prototype": True}
    scores = []
    for task in tasks:
        result = calibrated_predict(features[task["support_indices"]], task["support_labels"],
                                     features[task["query_indices"]], prior, config=config, **flags)
        scores.append(float(np.mean(result["predictions"] == task["query_labels"])))
    return summarize(scores)


def validation_score_maml(network, split, seed, episodes=25, steps=3, lr=.1):
    scores = []
    network.eval()
    for shots in (1, 5):
        rng = np.random.default_rng(seed + shots)
        per_shot = []
        for _ in range(episodes):
            task = sample_episode(split.labels, rng, shots=shots, queries=10)
            support = normalized_tensor(split.images[task["support_indices"]])
            query = normalized_tensor(split.images[task["query_indices"]])
            logits = maml_logits(network, support, torch.from_numpy(task["support_labels"]),
                                  query, steps=steps, learning_rate=lr)
            per_shot.append(float((logits.argmax(1).numpy() == task["query_labels"]).mean()))
        scores.append(float(np.mean(per_shot)))
    return {"1shot": scores[0], "5shot": scores[1], "mean": float(np.mean(scores))}


def train_maml_v2(args):
    directory = Path(args.output)
    if (directory / "maml_training.json").exists():
        done = json.loads((directory / "maml_training.json").read_text())
        if done.get("status") == "complete" and done.get("tasks") == args.maml_tasks:
            log("maml_already_complete"); return
    base, _ = load_checkpoint(directory / "baseline.pt")
    base_hash = digest_file(directory / "baseline.pt")
    torch.manual_seed(args.seed + 3)
    network = ResNet12(classes=5, widths=base.widths)
    network.encoder.load_state_dict(base.encoder.state_dict())
    del base
    training = load_split(args.data_root, "train", max_per_class=600, seed=args.seed)
    validation = load_split(args.data_root, "val", max_per_class=200, seed=args.seed + 1)
    optimizer = torch.optim.Adam(network.parameters(), lr=.0005)
    meta_batch = 4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.maml_tasks // meta_batch,
                                                         eta_min=.00005)
    rng = np.random.default_rng(args.seed + 3)
    records, best, start_task = [], -1.0, 0
    resume_file = directory / "maml_resume.pt"
    if resume_file.exists():
        saved = torch.load(resume_file, weights_only=True, map_location="cpu")
        if saved["base_sha256"] != base_hash or saved["tasks"] != args.maml_tasks:
            raise ValueError("MAML resume does not match the training plan")
        network.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"]); rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        records, best, start_task = saved["records"], saved["best"], saved["completed"]
    aggregate = []
    started = time.perf_counter()
    for task_index in range(start_task, args.maml_tasks):
        network.train()
        if task_index % meta_batch == 0:
            optimizer.zero_grad(set_to_none=True)
        shots = 1 if task_index % 2 == 0 else 5
        task = sample_episode(training.labels, rng, shots=shots, queries=5)
        support = normalized_tensor(training.images[task["support_indices"]])
        query = normalized_tensor(training.images[task["query_indices"]])
        support_labels = torch.from_numpy(task["support_labels"])
        query_labels = torch.from_numpy(task["query_labels"])
        adapted = maml_adapt(network, support, support_labels, steps=1,
                             learning_rate=.1, second_order=True)
        logits = torch.func.functional_call(network, adapted, (query,))
        loss = F.cross_entropy(logits, query_labels)
        (loss / meta_batch).backward()
        aggregate.append(float(loss.detach()))
        if (task_index + 1) % meta_batch == 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
            optimizer.step(); scheduler.step()
        if (task_index + 1) % 500 == 0 or task_index + 1 == args.maml_tasks:
            score = validation_score_maml(network, validation, args.seed + 800, episodes=25)
            row = {"tasks": task_index + 1, "loss": float(np.mean(aggregate)),
                   "validation": score, "seconds_this_session": time.perf_counter() - started}
            records.append(row); aggregate.clear()
            if score["mean"] > best:
                best = score["mean"]
                save_checkpoint(directory / "maml.pt", network, purpose="validation-selected exact second-order MAML",
                                metadata={"seed": args.seed + 3, "tasks": task_index + 1,
                                          "second_order": True, "base_sha256": base_hash})
            temporary_resume = resume_file.with_suffix(".tmp")
            torch.save({"model": network.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "records": records, "best": best,
                        "completed": task_index + 1, "tasks": args.maml_tasks, "base_sha256": base_hash}, temporary_resume)
            temporary_resume.replace(resume_file)
            write_json(directory / "maml_progress.json", {"status": "training", "records": records})
            log("maml_v2", **row)
    del training, network
    gc.collect()
    network, checkpoint = load_checkpoint(directory / "maml.pt")
    # Lock per-shot adaptation using validation only; no test predictions here.
    progress_path = directory / "maml_adaptation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    checkpoint_hash = digest_file(directory / "maml.pt")
    if progress.get("checkpoint_sha256") not in (None, checkpoint_hash):
        progress = {}
    tuning, chosen = {}, {}
    for shots in (1, 5):
        task_list = tasks_for(validation.labels, args.seed + 901 + shots, shots, 60)
        scores = []
        for steps in (1, 3, 5):
            for lr in (.025, .05, .1):
                existing = next((row for row in progress.get(str(shots), [])
                                 if row["steps"] == steps and row["learning_rate"] == lr), None)
                if existing is not None:
                    scores.append(existing)
                    continue
                accuracies = []
                for task in task_list:
                    logits = maml_logits(network, normalized_tensor(validation.images[task["support_indices"]]),
                        torch.from_numpy(task["support_labels"]), normalized_tensor(validation.images[task["query_indices"]]),
                        steps=steps, learning_rate=lr)
                    accuracies.append(float((logits.argmax(1).numpy() == task["query_labels"]).mean()))
                scores.append({"steps": steps, "learning_rate": lr, **summarize(accuracies)})
                progress["checkpoint_sha256"] = checkpoint_hash
                progress[str(shots)] = scores
                write_json(progress_path, progress)
        tuning[str(shots)] = scores
        winner = max(scores, key=lambda row: row["accuracy"])
        chosen[str(shots)] = {"steps": winner["steps"], "learning_rate": winner["learning_rate"]}
        log("maml_adaptation_selected", shots=shots, **winner)
    write_json(directory / "maml_training.json", {"status": "complete", "tasks": args.maml_tasks,
        "selected_tasks": checkpoint["metadata"]["tasks"], "meta_batch_size": meta_batch,
        "second_order": True, "training_shots": [1, 5], "queries_per_class": 5,
        "inner_steps": 1, "inner_lr": .1, "outer_lr": .0005,
        "history": records, "adaptation": chosen, "adaptation_tuning": tuning,
        "selection": "validation classes only; no query labels in test adaptation"})
    log("maml_complete")


def tune(args):
    directory = Path(args.output)
    training, validation = read_cache(directory, "aligned", "train"), read_cache(directory, "aligned", "val")
    search_ids, confirm_ids = validation_partition(validation["labels"], args.seed + 410)
    search_labels, confirm_labels = validation["labels"][search_ids], validation["labels"][confirm_ids]
    selected, records = {}, {}
    for shots in (1, 5):
        tasks = tasks_for(search_labels, args.seed + 420 + shots, shots, 120)
        confirm_tasks = tasks_for(confirm_labels, args.seed + 430 + shots, shots, 300)
        spaces = []
        for strength in (0., .5, 1.):
            for power in (1., .5):
                prior = fit_feature_space(training["features"], training["labels"], center_strength=strength, power=power)
                features = transform_features(validation["features"][search_ids], prior)
                config = CalibrationConfig(mean_weight=.05, variance_blend=.1, prototype_weight=.4,
                                           prototype_steps=4, temperature=.3, confidence_threshold=0.)
                score = score_configuration(features, search_labels, prior, tasks, config)
                spaces.append({"center_strength": strength, "power": power, **score})
        best_spaces = sorted(spaces, key=lambda row: -row["accuracy"])[:2]
        candidates = []
        for space in best_spaces:
            prior = fit_feature_space(training["features"], training["labels"],
                                      center_strength=space["center_strength"], power=space["power"])
            features = transform_features(validation["features"][search_ids], prior)
            # Staged search is specified in advance, keeps the search finite,
            # and retains every result including weaker configurations.
            anchor = CalibrationConfig(mean_weight=.05, variance_blend=.1, prototype_weight=.4,
                                       prototype_steps=4, temperature=.3, confidence_threshold=0.)
            stage_records = []
            for balanced in (False, True):
                for temperature in (.15, .3, .6):
                    for weight in (.15, .4, .8):
                        cfg = replace(anchor, balanced_queries=balanced, temperature=temperature, prototype_weight=weight)
                        score = score_configuration(features, search_labels, prior, tasks, cfg)
                        stage_records.append({"config": cfg.to_dict(), "space": {k: space[k] for k in ("center_strength", "power")}, **score})
            winner = max(stage_records, key=lambda row: row["accuracy"])
            anchor = CalibrationConfig(**winner["config"])
            for blend in (0., .05, .15, .4, 1.):
                for mean_weight in (0., .03, .1):
                    cfg = replace(anchor, variance_blend=blend, mean_weight=mean_weight)
                    stage_records.append({"config": cfg.to_dict(), "space": {k: space[k] for k in ("center_strength", "power")},
                        **score_configuration(features, search_labels, prior, tasks, cfg)})
            candidates.extend(stage_records)
        # Independent validation images confirm a shortlist. Choosing zero
        # shrinkage is legitimate, and must be reported instead of hidden.
        unique = {json.dumps({"space": row["space"], "config": row["config"]}, sort_keys=True):row for row in candidates}
        shortlist = sorted(unique.values(), key=lambda row:-row["accuracy"])[:5]
        for row in shortlist:
            prior = fit_feature_space(training["features"], training["labels"], **row["space"])
            features = transform_features(validation["features"][confirm_ids], prior)
            row["confirmation"] = score_configuration(features, confirm_labels, prior, confirm_tasks, CalibrationConfig(**row["config"]))
        winner = max(shortlist, key=lambda row:row["confirmation"]["accuracy"])
        selected[str(shots)] = {"space": winner["space"], "config": winner["config"], "validation_accuracy": winner["confirmation"]["accuracy"]}
        records[str(shots)] = {"spaces": spaces, "search_episodes":len(tasks), "confirmation_episodes":len(confirm_tasks),
                              "candidates":candidates, "shortlist":shortlist, "selected":selected[str(shots)]}
        write_json(directory / "v2_validation_tuning.json", records)
        log("calibration_selected", shots=shots, **selected[str(shots)])
    write_json(directory / "v2_calibration_locked.json", {
        "status":"locked", "locked_utc":datetime.now(timezone.utc).isoformat(), "seed":args.seed,
        "selection":"validation-only search on 200 images/class, confirmation on disjoint 400 images/class",
        "test_accessed":False, "configurations":selected,
        "weights":{name:digest_file(directory/f"{name}.pt") for name in ("baseline","aligned")},
    })
    log("calibration_locked")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cache","maml","tune"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maml-tasks", type=int, default=4000)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    {"cache":cache_features,"maml":train_maml_v2,"tune":tune}[args.stage](args)


if __name__ == "__main__":
    main()
