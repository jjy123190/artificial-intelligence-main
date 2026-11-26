"""Compare two matched continuations of the course ResNet12 on base images only."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .accuracy90_data import json_from, load_archive
from .data import digest_file, normalized_tensor
from .models import distribution_alignment
from .train_gpu_backbone import atomic_save, batches_by_class, episode_loss, training_view
from .train_gpu_metric import save_json
from .train_medium_gpu import medium_model
from .train_resnet_rotation import cutmix


BRANCHES = ("control", "distill")
TEMPERATURE = 3.0


def restore_backbone(weights):
    model = medium_model().to("cuda", memory_format=torch.channels_last)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model


@torch.inference_mode()
def soft_targets(weights, images, dtype):
    teacher = restore_backbone(weights).eval()
    targets = []
    for offset in range(0, len(images), 64):
        x = normalized_tensor(images[offset:offset + 64]).cuda()
        views = torch.cat((x, x.flip(-1))).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=dtype):
            logits = teacher(views)
        first, second = (logits.float() / TEMPERATURE).softmax(1).chunk(2)
        targets.append((first + second) * .5)
    return torch.cat(targets)


def train_branch(directory, plan, name, weights, images, labels, targets, teacher, dtype):
    directory.mkdir(exist_ok=True)
    model = restore_backbone(weights).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0004, weight_decay=.03, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    checkpoint = directory / "resume.pt"
    history = []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["plan"] != plan or state["branch"] != name:
            raise ValueError("Checkpoint belongs to a different training run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        del state

    for epoch in range(len(history), plan["epochs"]):
        started = time.perf_counter()
        rng = np.random.default_rng(plan["seed"] + epoch)
        torch.manual_seed(plan["seed"] + epoch)
        cosine = (1 + math.cos(math.pi * epoch / (plan["epochs"] - 1))) / 2
        lr = (.000025 + .000375 * cosine) * min(1., (epoch + 1) / 2)
        optimizer.param_groups[0]["lr"] = lr
        totals = torch.zeros(5, device="cuda")
        for step, indices in enumerate(batches_by_class(labels, rng)):
            idx = torch.as_tensor(indices, device="cuda")
            x, y = training_view(images[idx], 84), targets[idx]
            # Two clean examples from each class receive a second, independent crop.
            pair = torch.arange(step % 4, len(y), 4, device="cuda")
            second = training_view(images[idx[pair]], 84)
            mixed = step % 4 == 0
            if mixed:
                x, perm, area = cutmix(x, rng)
            views = torch.cat((x, second)).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=dtype):
                logits, features = model(views, return_features=True)
            logits, features = logits.float(), features.float()
            main = logits[:len(y)]
            ce = F.cross_entropy(main, y, label_smoothing=.04)
            if mixed:
                ce = (1 - area) * ce + area * F.cross_entropy(main, y[perm], label_smoothing=.04)
            ce = .8 * ce + .2 * F.cross_entropy(logits[len(y):], y[pair], label_smoothing=.04)
            # Compare clean crops only. Moment alignment uses the original feature scale.
            align = features.new_zeros(()) if mixed else distribution_alignment(features[pair], features[len(y):])
            episodic = features.new_zeros(()) if mixed else episode_loss(features[:len(y)], 1 if step % 2 else 5)
            kd = main.new_zeros(())
            if teacher is not None:
                soft = teacher[idx]
                if mixed:
                    soft = (1 - area) * soft + area * soft[perm]
                kd = F.kl_div(F.log_softmax(main / TEMPERATURE, dim=1), soft, reduction="batchmean") * TEMPERATURE ** 2
            loss = ce + .5 * align + .35 * episodic + .7 * kd
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals += torch.stack((loss, ce, align, episodic, kd)).detach()
        values = (totals / (step + 1)).tolist()
        if step + 1 != 300 or not all(map(math.isfinite, values)):
            raise RuntimeError("Invalid epoch length or loss")
        row = dict(zip(("loss", "ce", "alignment", "episodic", "kl_t_squared"), values))
        row.update(epoch=epoch + 1, updates=step + 1, lr=lr, seconds=round(time.perf_counter() - started, 2))
        history.append(row)
        atomic_save(checkpoint, {"branch": name, "plan": plan, "model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "history": history})
        print(name, row, flush=True)
    atomic_save(directory / "medium.pt", {k: v.cpu() for k, v in model.state_dict().items()})
    save_json(directory / "training.json", {"branch": name, "history": history,
              "medium_sha256": digest_file(directory / "medium.pt")})


def train(args):
    if args.epochs < 2 or not torch.cuda.is_available():
        raise ValueError("At least two epochs and a CUDA device are required")
    torch.set_num_threads(2)
    weights = args.parent / "medium.pt"
    parent = json_from(args.parent / "plan.json")
    result = json_from(args.parent / "results.json")
    ids = json_from(args.artifacts / "protocol.json")["train_selection_sha256"]
    if (digest_file(args.parent / "plan.json") != result["files"]["plan.json"]
            or digest_file(weights) != result["files"]["medium.pt"] or digest_file(weights) != parent["continued_sha256"]
            or parent["external_pretraining"] or parent["training"]["external_weights"]
            or parent["training"]["parent_training"]["external_weights"] or parent["training"]["train_ids"] != ids):
        raise ValueError("Expected the locked, base-only parent ResNet12")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    files = ("train_self_distill.py", "train_gpu_backbone.py", "train_resnet_rotation.py",
             "train_medium_gpu.py", "models.py", "accuracy90_data.py", "data.py")
    plan = {"parent_sha256": digest_file(weights), "external_weights": False, "train_ids": ids,
            "seed": args.seed, "epochs": args.epochs, "updates_per_epoch": 300, "batch_size": 128,
            "branches": list(BRANCHES), "distill_temperature": TEMPERATURE, "distill_weight": .7,
            "optimizer": "AdamW", "dtype": str(dtype), "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(), "source": {n: digest_file(Path(__file__).with_name(n)) for n in files},
            "selection": "Last epoch of each branch. Maximize total correct validation queries across both shots using fixed restarts inference. Ties favor parent, control, then distill."}
    args.output.mkdir(exist_ok=True, parents=True)
    plan_file = args.output / "training_plan.json"
    if plan_file.exists() and json_from(plan_file) != plan:
        raise ValueError("Different plan in the training directory")
    save_json(plan_file, plan)
    todo = []
    for name in BRANCHES:
        report = args.output / name / "training.json"
        if report.exists():
            verify_training(args.output / name, name, plan)
        else:
            todo.append(name)
    if not todo:
        return
    print("Loading 64 base classes for", todo, "on", plan["gpu"], flush=True)
    base = load_archive(args.data_root, args.artifacts, "train")
    labels = base.labels
    if np.bincount(labels).tolist() != [600] * 64:
        raise ValueError("Expected 600 images in each base class")
    images = torch.from_numpy(base.images).permute(0, 3, 1, 2).cuda()
    targets = torch.from_numpy(labels).cuda()
    for name in todo:
        teacher = None
        if name == "distill":
            started = time.perf_counter()
            teacher = soft_targets(weights, base.images, dtype)
            torch.cuda.synchronize()
            save_json(args.output / "teacher.json", {"views": 2, "images": len(labels),
                      "seconds": round(time.perf_counter() - started, 2), "teacher_sha256": plan["parent_sha256"]})
        train_branch(args.output / name, plan, name, weights, images, labels, targets, teacher, dtype)


def verify_training(directory, name, plan):
    record = json_from(directory / "training.json")
    if (record["branch"] != name or record["medium_sha256"] != digest_file(directory / "medium.pt")
            or [(r["epoch"], r["updates"]) for r in record["history"]] !=
               [(i + 1, plan["updates_per_epoch"]) for i in range(plan["epochs"])]):
        raise ValueError("Incomplete training record or mismatched backbone")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--parent", type=Path, default=Path("outputs/lab04_gpu_rotation"))
    parser.add_argument("--output", type=Path, default=Path(".local/lab04-self-distill"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261130)
    train(parser.parse_args())
