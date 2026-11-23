"""Run the measured mini-ImageNet experiment; all randomness is recorded.

Usage: python -m lab04.experiment --data-root PATH --output PATH
No test examples or labels enter optimization or hyperparameter selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .calibration import CalibrationConfig, METHODS, base_statistics, calibrated_predict
from .data import (digest_file, load_split, normalized_tensor, sample_episode,
                   selection_digest, verify_splits)
from .models import ResNet12, distribution_alignment, maml_adapt, maml_logits


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=json_default), encoding="utf-8")
    temporary.replace(path)


def log(stage, **values):
    print(json.dumps({"stage": stage, **values}, default=json_default), flush=True)


def extract_features(model, images, *, batch_size=96):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            output.append(model.features(normalized_tensor(images[start:start + batch_size])).numpy())
    return np.concatenate(output)


def validation_accuracy(features, labels, prior, *, seed, episodes=30, shots=5):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(episodes):
        episode = sample_episode(labels, rng, shots=shots)
        result = calibrated_predict(features[episode["support_indices"]], episode["support_labels"],
                                    features[episode["query_indices"]], prior)
        values.append(float(np.mean(result["predictions"] == episode["query_labels"])))
    return float(np.mean(values))


def save_checkpoint(path, model, *, purpose, metadata):
    torch.save({"state_dict": model.state_dict(), "widths": list(model.widths),
                "classes": model.classifier.out_features, "purpose": purpose,
                "metadata": metadata}, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ResNet12(classes=checkpoint["classes"], widths=checkpoint["widths"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def restore_evaluation_configuration(args, provided_options):
    """Inherit and verify training facts instead of relabelling old weights.

    Only runtime thread/batch size and the number of evaluation episodes may
    differ. Data selection, training seed and training schedule remain the
    facts stored in all three checkpoints.
    """
    inherited = ("seed", "train_per_class", "eval_per_class", "epochs", "branch_epochs", "maml_tasks")
    metadata = []
    for name in ("baseline", "aligned", "maml"):
        checkpoint = torch.load(Path(args.output) / f"{name}.pt", map_location="cpu", weights_only=True)
        metadata.append(checkpoint["metadata"])
    for key in inherited:
        values = [record[key] for record in metadata]
        if any(value != values[0] for value in values):
            raise ValueError(f"Checkpoint training metadata disagree for {key}: {values}")
        option = "--" + key.replace("_", "-")
        if option in provided_options and getattr(args, key) != values[0]:
            raise ValueError(f"{option} conflicts with the trained checkpoints: expected {values[0]}")
        setattr(args, key, values[0])


def train_supervised(model, split, validation, *, epochs, batch_size, learning_rate,
                     seed, alignment_weight=0.0, paired_views=False, stage="pretrain",
                     select_best=False):
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    best_score, best_state = -1, None
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(split.images))
        totals = {"ce": 0.0, "alignment": 0.0, "correct": 0}
        epoch_started = time.perf_counter()
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            images = split.images[indices]
            labels = torch.from_numpy(split.labels[indices])
            first = normalized_tensor(images, augment=True, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(first, return_features=True)
            ce_loss = F.cross_entropy(logits, labels)
            alignment_loss = features.new_zeros(())
            if paired_views:
                second = normalized_tensor(images, augment=True, generator=generator)
                second_logits, second_features = model(second, return_features=True)
                ce_loss = 0.5 * (ce_loss + F.cross_entropy(second_logits, labels))
                alignment_loss = distribution_alignment(features, second_features)
            loss = ce_loss + alignment_weight * alignment_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            totals["ce"] += float(ce_loss.detach()) * len(indices)
            totals["alignment"] += float(alignment_loss.detach()) * len(indices)
            totals["correct"] += int((logits.detach().argmax(1) == labels).sum())
        scheduler.step()
        # This metric is only a checkpoint diagnostic. Calibration parameters
        # are selected later from the same held-out validation classes.
        val_features = extract_features(model, validation.images)
        # A dummy prior is sufficient for the plain Euclidean val classifier;
        # the common distance scale never changes its predictions.
        dummy_prior = {"means": np.zeros((1, val_features.shape[1])),
                       "variances": np.ones((1, val_features.shape[1]))}
        score = validation_accuracy(val_features, validation.labels, dummy_prior,
                                    seed=seed + 100, episodes=30)
        row = {"epoch": epoch + 1, "ce_loss": totals["ce"] / len(order),
               "alignment_loss": totals["alignment"] / len(order),
               "training_accuracy": totals["correct"] / len(order),
               "validation_5shot_accuracy": score,
               "seconds": time.perf_counter() - epoch_started}
        history.append(row)
        if score > best_score:
            best_score, best_state = score, deepcopy(model.state_dict())
        log(stage, **row)
    if select_best:
        model.load_state_dict(best_state)
    return {"epochs": history, "seconds": time.perf_counter() - started,
            "alignment_weight": alignment_weight, "paired_views": paired_views,
            "selection": "best_validation_5shot" if select_best else "fixed_final_epoch"}


def train_maml(initial, split, *, tasks, seed, batch_size=2, inner_lr=0.1,
               outer_lr=0.001, second_order=True):
    model = ResNet12(classes=5, widths=initial.widths)
    model.encoder.load_state_dict(initial.encoder.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    rng = np.random.default_rng(seed)
    history, aggregate = [], []
    started = time.perf_counter()
    for task in range(tasks):
        model.train()
        if task % batch_size == 0:
            optimizer.zero_grad(set_to_none=True)
        # Balance 1-shot and 5-shot training. Every parameter adapts using
        # support labels, and outer gradients use disjoint training queries.
        shots = 1 if task % 2 == 0 else 5
        episode = sample_episode(split.labels, rng, shots=shots, queries=5)
        support = normalized_tensor(split.images[episode["support_indices"]])
        query = normalized_tensor(split.images[episode["query_indices"]])
        support_labels = torch.from_numpy(episode["support_labels"])
        query_labels = torch.from_numpy(episode["query_labels"])
        parameters = maml_adapt(model, support, support_labels, steps=1,
                                learning_rate=inner_lr, second_order=second_order)
        logits = torch.func.functional_call(model, parameters, (query,))
        loss = F.cross_entropy(logits, query_labels)
        # Correct normalization also handles a final incomplete meta-batch.
        current_batch = min(batch_size, tasks - (task // batch_size) * batch_size)
        (loss / current_batch).backward()
        if (task + 1) % batch_size == 0 or task + 1 == tasks:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        aggregate.append((float(loss.detach()),
                          float((logits.detach().argmax(1) == query_labels).float().mean())))
        if (task + 1) % 50 == 0 or task + 1 == tasks:
            row = {"tasks": task + 1, "query_loss": float(np.mean([v[0] for v in aggregate])),
                   "query_accuracy": float(np.mean([v[1] for v in aggregate])),
                   "seconds": time.perf_counter() - started}
            history.append(row)
            aggregate.clear()
            log("maml", **row)
    return model, {"tasks": tasks, "meta_batch_size": batch_size, "inner_steps": 1,
                   "training_shots": [1, 5], "queries_per_class": 5,
                   "inner_lr": inner_lr, "outer_lr": outer_lr,
                   "second_order": second_order, "history": history,
                   "seconds": time.perf_counter() - started}


def summarize(values):
    values = np.asarray(values, dtype=float)
    half_width = (1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"accuracy": float(values.mean()), "ci95": float(half_width),
            "episodes": int(len(values)),
            "ci95_low": float(max(0, values.mean() - half_width)),
            "ci95_high": float(min(1, values.mean() + half_width))}


def tune_calibration(features, labels, prior, *, seed, shots, episodes=40):
    rng = np.random.default_rng(seed)
    tasks = [sample_episode(labels, rng, shots=shots) for _ in range(episodes)]
    candidates = []
    for mean_weight in (0.05, 0.15, 0.30):
        for prototype_weight in (0.10, 0.30, 0.60):
            for temperature in (0.5, 1.0):
                candidates.append(CalibrationConfig(mean_weight=mean_weight,
                                                   prototype_weight=prototype_weight,
                                                   temperature=temperature))
    records = []
    for config in candidates:
        scores = []
        for task in tasks:
            result = calibrated_predict(features[task["support_indices"]], task["support_labels"],
                                        features[task["query_indices"]], prior,
                                        mean=True, variance=True, prototype=True, config=config)
            scores.append(np.mean(result["predictions"] == task["query_labels"]))
        records.append({"config": asdict(config), "validation_accuracy": float(np.mean(scores))})
    best = max(range(len(records)), key=lambda index: records[index]["validation_accuracy"])
    return candidates[best], {"seed": seed, "episodes": episodes, "shots": shots,
                              "selected_index": best, "candidates": records}


def evaluate_test(baseline, aligned, maml, test, features, priors, configs, *,
                  seed, episodes, inner_lr):
    rows, episode_records = [], []
    for shots in (1, 5):
        rng = np.random.default_rng(seed + shots)
        scores = {method: [] for method in (*METHODS, "maml")}
        durations = {method: [] for method in scores}
        for index in range(episodes):
            task = sample_episode(test.labels, rng, shots=shots)
            record = {"episode": index, "shots": shots, "classes": task["classes"],
                      "support_ids": [test.ids[i] for i in task["support_indices"]],
                      "query_ids": [test.ids[i] for i in task["query_indices"]], "methods": {}}
            for method, flags in METHODS.items():
                source = "baseline" if method in {"baseline", "no_alignment_full"} else "aligned"
                embedded = features[source]
                started = time.perf_counter()
                prediction = calibrated_predict(embedded[task["support_indices"]], task["support_labels"],
                                                embedded[task["query_indices"]], priors[source],
                                                config=configs[shots], **flags)
                elapsed = time.perf_counter() - started
                score = float(np.mean(prediction["predictions"] == task["query_labels"]))
                scores[method].append(score)
                durations[method].append(elapsed)
                record["methods"][method] = {"accuracy": score, "seconds": elapsed}
            started = time.perf_counter()
            logits = maml_logits(maml, normalized_tensor(test.images[task["support_indices"]]),
                                 torch.from_numpy(task["support_labels"]),
                                 normalized_tensor(test.images[task["query_indices"]]),
                                 steps=3, learning_rate=inner_lr)
            elapsed = time.perf_counter() - started
            score = float(np.mean(logits.argmax(1).numpy() == task["query_labels"]))
            scores["maml"].append(score)
            durations["maml"].append(elapsed)
            record["methods"]["maml"] = {"accuracy": score, "seconds": elapsed}
            episode_records.append(record)
            if (index + 1) % 25 == 0:
                log("evaluate", shots=shots, episodes=index + 1,
                    accuracies={key: float(np.mean(value)) for key, value in scores.items()})
        for method, values in scores.items():
            delta = np.asarray(values) - np.asarray(scores["baseline"])
            half_width = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
            rows.append({"method": method, "shots": shots, **summarize(values),
                         "paired_delta": float(delta.mean()), "paired_ci95": float(half_width),
                         "mean_episode_seconds": float(np.mean(durations[method])),
                         "timing_scope": ("image_encoding_and_support_adaptation" if method == "maml"
                                          else "inference_from_cached_features"),
                         "transductive": bool(METHODS.get(method, {}).get("prototype", False))})
    return rows, episode_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--train-per-class", type=int, default=600)
    parser.add_argument("--eval-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--branch-epochs", type=int, default=2)
    parser.add_argument("--maml-tasks", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "evaluate":
        provided = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        restore_evaluation_configuration(args, provided)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    args_dict = vars(args).copy()
    args_dict.pop("data_root")
    args_dict.pop("output")
    split_integrity = verify_splits(args.data_root)
    log("load_data", configuration=args_dict)
    training = load_split(args.data_root, "train", max_per_class=args.train_per_class, seed=args.seed)
    validation = load_split(args.data_root, "val", max_per_class=args.eval_per_class, seed=args.seed + 1)
    protocol = {"dataset": "mini-ImageNet", "split": "Ravi & Larochelle 64/16/20",
                "image_size": [84, 84], "training_classes": 64, "validation_classes": 16,
                "test_classes": 20, "train_per_class": args.train_per_class,
                "evaluation_pool_per_class": args.eval_per_class,
                "training_images": len(training.images), "validation_images": len(validation.images),
                "ways": 5, "shots": [1, 5], "queries_per_class": 15,
                "test_episodes_per_shot": args.episodes, "seed": args.seed,
                "train_selection_sha256": selection_digest(training),
                "validation_selection_sha256": selection_digest(validation),
                "full_training_split": len(training.images) == 38400,
                "full_evaluation_split": args.eval_per_class >= 600,
                "maml_train_queries_per_class": 5, "maml_train_inner_steps": 1,
                "maml_test_inner_steps": 3,
                "scope": "CPU narrow-backbone experiment; scores are not published standard-width benchmarks",
                "no_test_tuning": True}
    if args.stage != "evaluate":
        warm = ResNet12(classes=64)
        warm_history = train_supervised(warm, training, validation, epochs=args.epochs,
                                       batch_size=args.batch_size, learning_rate=0.003,
                                       seed=args.seed, stage="warm_ce", select_best=False)
        save_checkpoint(output / "warm.pt", warm, purpose="common_initialization", metadata=args_dict)
        baseline, aligned = deepcopy(warm), deepcopy(warm)
        branch_seed = args.seed + 2
        baseline_history = train_supervised(baseline, training, validation, epochs=args.branch_epochs,
                                           batch_size=args.batch_size, learning_rate=0.0005,
                                           seed=branch_seed, alignment_weight=0.0, paired_views=True,
                                           stage="baseline_ce_control")
        save_checkpoint(output / "baseline.pt", baseline, purpose="CE-only matched control", metadata=args_dict)
        aligned_history = train_supervised(aligned, training, validation, epochs=args.branch_epochs,
                                          batch_size=args.batch_size, learning_rate=0.0005,
                                          seed=branch_seed, alignment_weight=0.5, paired_views=True,
                                          stage="aligned_ce")
        save_checkpoint(output / "aligned.pt", aligned, purpose="CE plus mean-variance consistency", metadata=args_dict)
        # The MAML classifier initialization is seeded independently of the
        # random numbers consumed by the two supervised branches.
        torch.manual_seed(args.seed + 3)
        maml, maml_history = train_maml(baseline, training, tasks=args.maml_tasks,
                                        seed=args.seed + 3, second_order=True)
        save_checkpoint(output / "maml.pt", maml, purpose="full second-order MAML", metadata=args_dict)
        history = {"warm_ce": warm_history, "baseline_control": baseline_history,
                   "aligned": aligned_history, "maml": maml_history,
                   "matched_control": "same warm checkpoint, examples, order, augmentations, CE steps and epochs; only alignment coefficient differs"}
        write_json(output / "training.json", history)
    else:
        baseline, _ = load_checkpoint(output / "baseline.pt")
        aligned, _ = load_checkpoint(output / "aligned.pt")
        maml, _ = load_checkpoint(output / "maml.pt")
        history = json.loads((output / "training.json").read_text(encoding="utf-8"))
    if args.stage == "train":
        write_json(output / "protocol.json", protocol)
        return
    # Test data are loaded only after all training is complete.
    models = {"baseline": baseline, "aligned": aligned}
    priors, configurations, tuning = {}, {}, {}
    for name, model in models.items():
        log("extract_training_prior", model=name, images=len(training.images))
        base_features = extract_features(model, training.images)
        priors[name] = base_statistics(base_features, training.labels)
        np.savez_compressed(output / f"{name}_prior.npz", **priors[name])
    aligned_validation = extract_features(aligned, validation.images)
    for shots in (1, 5):
        configurations[shots], tuning[shots] = tune_calibration(
            aligned_validation, validation.labels, priors["aligned"],
            seed=args.seed + 100 + shots, shots=shots)
    write_json(output / "calibration_config.json", {str(k): asdict(v) for k, v in configurations.items()})
    write_json(output / "validation_tuning.json", tuning)
    log("validation_locked", configurations={key: asdict(value) for key, value in configurations.items()})
    test = load_split(args.data_root, "test", max_per_class=args.eval_per_class, seed=args.seed + 4)
    protocol["test_selection_sha256"] = selection_digest(test)
    protocol["test_images"] = len(test.images)
    write_json(output / "protocol.json", protocol)
    features = {name: extract_features(model, test.images) for name, model in models.items()}
    rows, episode_records = evaluate_test(baseline, aligned, maml, test, features, priors,
                                          configurations, seed=args.seed + 200,
                                          episodes=args.episodes, inner_lr=0.1)
    # A fixed, small test pool makes the local demonstration self-contained.
    # It is separate from training and retains 40 examples per held-out class.
    demo_indices = np.concatenate([np.flatnonzero(test.labels == label)[:40]
                                   for label in np.unique(test.labels)])
    names_path = Path(args.data_root) / "class_names.json"
    names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else {}
    readable = [names.get(wnid, wnid).replace("_", " ") for wnid in test.class_names]
    np.savez_compressed(output / "demo_pool.npz", images=test.images[demo_indices],
                        labels=test.labels[demo_indices], ids=np.array(test.ids)[demo_indices],
                        class_wnids=np.array(test.class_names), class_names=np.array(readable),
                        baseline_features=features["baseline"][demo_indices],
                        aligned_features=features["aligned"][demo_indices])
    weights = {name: {"path": f"{name}.pt", "sha256": digest_file(output / f"{name}.pt")}
               for name in ("baseline", "aligned", "maml")}
    result = {"schema_version": 1, "status": "complete",
              "completed_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "dataset_integrity": split_integrity,
              "model": {"name": "ResNet12-Narrow", "widths": [8, 16, 24, 32],
                        "normalization": "GroupNorm(4)", "embedding_dim": 32,
                        "convolution_layers": 12, "parameters_baseline": sum(p.numel() for p in baseline.parameters()),
                        "parameters_maml": sum(p.numel() for p in maml.parameters()),
                        "standard_width_benchmark": False},
              "runtime": {"torch": str(torch.__version__), "device": "cpu", "threads": args.threads},
              "training": history, "weights": weights, "rows": rows,
              "priors": {name: {"path": f"{name}_prior.npz",
                                  "sha256": digest_file(output / f"{name}_prior.npz")}
                         for name in ("baseline", "aligned")},
              "calibration": {str(k): asdict(v) for k, v in configurations.items()},
              "demo_pool": {"path": "demo_pool.npz", "sha256": digest_file(output / "demo_pool.npz"),
                            "classes": 20, "images_per_class": 40, "images": len(demo_indices),
                            "source": "held-out mini-ImageNet test classes; educational demonstration"},
              "seconds": time.perf_counter() - started,
              "caveats": ["Narrow 50k-parameter CPU backbone; do not compare its scores with standard 640-channel published ResNet12 results.",
                          "100 held-out images per class are sampled for evaluation, while all 600 per class are used for training.",
                          "Prototype refinement is transductive and uses the joint unlabelled query batch.",
                          "Normal-approximation 95% CIs describe episode sampling conditional on one trained checkpoint, not variability across training seeds.",
                          "Calibration softmax scores are relative model scores; no claim of calibrated probability reliability is made.",
                          "Cached-feature timing excludes image encoding and cannot be directly compared with MAML full inference timing."]}
    write_json(output / "results.json", result)
    write_json(output / "episodes.json", {"schema_version": 1, "seed": args.seed + 200,
                                          "episodes": episode_records})
    log("complete", rows=rows, seconds=result["seconds"])


if __name__ == "__main__":
    main()
