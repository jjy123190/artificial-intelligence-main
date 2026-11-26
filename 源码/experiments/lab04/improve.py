"""Resumable validation-selected CPU training for the second lab04 study.

No test image is opened by this training entry point. Existing published
artifacts are never overwritten: train into a separate candidate directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_split, normalized_tensor, selection_digest
from .experiment import (extract_features, load_checkpoint, log, save_checkpoint,
                         train_supervised, validation_accuracy, write_json)
from .models import ResNet12


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    configuration = vars(args).copy()
    configuration["selection"] = "validation classes only; never test accuracy"
    write_json(output / "training_plan.json", configuration)
    training = load_split(args.data_root, "train", max_per_class=600, seed=args.seed)
    validation = load_split(args.data_root, "val", max_per_class=200, seed=args.seed + 1)
    write_json(output / "training_data.json", {
        "train_images": len(training.images), "validation_images": len(validation.images),
        "train_sha256": selection_digest(training),
        "validation_sha256": selection_digest(validation),
        "train_classes": training.class_names, "validation_classes": validation.class_names,
        "test_accessed": False,
    })
    assert not set(training.class_names) & set(validation.class_names)
    model = ResNet12(classes=64, widths=args.widths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                         eta_min=1e-5)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    history, best_score, start_epoch = [], -1.0, 0
    resume_path = output / "resume.pt"
    if resume_path.exists():
        resume = torch.load(resume_path, weights_only=True, map_location="cpu")
        # Only safe tensor/primitive deserialization is accepted, including local resume state.
        for key in ("widths", "epochs", "seed", "learning_rate", "batch_size"):
            if resume["configuration"][key] != configuration[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        rng.bit_generator.state = resume["rng"]
        generator.set_state(resume["generator"])
        torch.set_rng_state(resume["torch_rng"])
        history, best_score = resume["history"], resume["best_score"]
        start_epoch = len(history)
        log("resume", epoch=start_epoch)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        started, total_loss, total_correct = time.perf_counter(), 0.0, 0
        order = rng.permutation(len(training.images))
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            images = normalized_tensor(training.images[ids], augment=True, generator=generator)
            labels = torch.from_numpy(training.labels[ids])
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
            total_correct += int((logits.detach().argmax(1) == labels).sum())
        scheduler.step()
        features = extract_features(model, validation.images, batch_size=args.batch_size)
        prior = {"means": np.zeros((1, features.shape[1])), "variances": np.ones((1, features.shape[1]))}
        scores = {shots: validation_accuracy(features, validation.labels, prior,
                  seed=args.seed + 100 + shots, episodes=100, shots=shots) for shots in (1, 5)}
        score = float(np.mean(list(scores.values())))
        row = {"epoch": epoch + 1, "ce_loss": total_loss / len(order),
               "training_accuracy": total_correct / len(order),
               "validation_1shot_accuracy": scores[1], "validation_5shot_accuracy": scores[5],
               "validation_mean": score, "seconds": time.perf_counter() - started}
        history.append(row)
        if score > best_score:
            best_score = score
            save_checkpoint(output / "warm.pt", model, purpose="validation-selected common initialization",
                            metadata={**configuration, "selected_epoch": epoch + 1, "validation_mean": score})
        temporary = resume_path.with_suffix(".tmp")
        torch.save({"configuration": configuration, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng": rng.bit_generator.state, "generator": generator.get_state(),
                    "torch_rng": torch.get_rng_state(), "history": history,
                    "best_score": best_score}, temporary)
        temporary.replace(resume_path)
        write_json(output / "training_progress.json", {"status": "pretraining", "epochs": history,
                   "best_validation_mean": best_score, "total_epochs": args.epochs})
        log("pretrain_v2", **row)
    warm, checkpoint = load_checkpoint(output / "warm.pt")
    histories = {}
    for name, alignment in (("baseline", 0.0), ("aligned", 0.5)):
        branch_history = output / f"{name}_training.json"
        if branch_history.exists() and (output / f"{name}.pt").exists():
            histories[name] = json.loads(branch_history.read_text(encoding="utf-8"))
            continue
        branch = deepcopy(warm)
        result = train_supervised(branch, training, validation, epochs=args.branch_epochs,
                  batch_size=args.batch_size, learning_rate=0.0002, seed=args.seed + 2,
                  alignment_weight=alignment, paired_views=True, stage=f"v2_{name}")
        save_checkpoint(output / f"{name}.pt", branch, purpose=f"matched {name} branch",
                        metadata={**configuration, "warm_epoch": checkpoint["metadata"]["selected_epoch"]})
        histories[name] = result
        write_json(branch_history, result)
        del branch
    write_json(output / "training.json", {
        "warm_ce": {"epochs": history, "selection": "best mean validation 1shot/5shot", "selected_epoch": checkpoint["metadata"]["selected_epoch"],
                    "label_smoothing": 0.05, "weight_decay": 0.0005},
        "baseline_control": histories["baseline"], "aligned": histories["aligned"],
        "matched_control": "same initialization, images, order, augmentations and CE steps; alignment weight 0 vs 0.5",
    })
    write_json(output / "training_progress.json", {"status": "complete", "epochs": history,
               "best_validation_mean": best_score, "total_epochs": args.epochs})
    log("training_complete", output=str(output), selected_epoch=checkpoint["metadata"]["selected_epoch"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--widths", type=int, nargs=4, default=[8, 16, 32, 64])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--branch-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261124)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
