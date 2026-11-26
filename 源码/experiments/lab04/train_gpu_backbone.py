"""Train an independently initialized lightweight ResNet12 on the 64 base classes."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import digest_file, load_split, selection_digest
from .models import ResNet12, distribution_alignment


def batches_by_class(labels, rng):
    """Visit every image once. Each batch contains 16 classes with eight images each."""
    groups = np.stack([rng.permutation(np.flatnonzero(labels == c)) for c in range(64)])
    for offset in range(0, groups.shape[1], 8):
        for classes in rng.permutation(64).reshape(4, 16):
            yield groups[classes, offset:offset + 8].reshape(-1)


def training_view(images, side):
    count = len(images)
    device = images.device
    affine = torch.zeros(count, 2, 3, device=device)
    scale = torch.empty(count, device=device).uniform_(0.73, 1.05)
    affine[:, 0, 0] = scale * (2 * torch.randint(2, (count,), device=device) - 1)
    affine[:, 1, 1] = scale
    affine[:, :, 2] = (torch.rand(count, 2, device=device) * 2 - 1) * (1.09 - scale[:, None])
    grid = F.affine_grid(affine, (count, 3, side, side), align_corners=False)
    x = F.grid_sample(images.float().mul_(1 / 255), grid, padding_mode="reflection", align_corners=False)
    gain = torch.empty(count, 3, 1, 1, device=device).uniform_(0.82, 1.18)
    x = (x * gain).clamp_(0, 1)
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def episode_loss(features, shots):
    # Batch order is randomized per epoch. Three disjoint five-way tasks use 15 classes.
    task = F.normalize(features[:120].reshape(3, 5, 8, -1), dim=-1)
    prototypes = F.normalize(task[:, :, :shots].mean(2), dim=-1)
    query = task[:, :, shots:].flatten(1, 2)
    scores = query @ prototypes.transpose(1, 2) / .08
    target = torch.arange(5, device=features.device).repeat_interleave(8 - shots).repeat(3)
    return F.cross_entropy(scores.flatten(0, 1), target)


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA-enabled Python environment")
    if args.epochs < 2 or args.meta_epochs < 0:
        raise ValueError("At least two classification epochs are required")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    gpu = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__)
    protocol = json.loads((args.artifacts / "protocol.json").read_bytes())
    plan = {"widths": [16, 32, 64, 128], "seed": args.seed, "epochs": args.epochs,
            "meta_epochs": args.meta_epochs, "images": 38400, "batch": 128, "paired_views_per_batch": 32,
            "train_selection_sha256": protocol["train_selection_sha256"], "external_weights": False,
            "precision": str(dtype), "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "code_sha256": {p.name: digest_file(p) for p in (source, source.with_name("models.py"),
                                                             source.with_name("data.py"))}}
    plan_path = args.output / "backbone_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_bytes()) != plan:
        raise ValueError("Use a separate output directory for a different training plan")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    model = ResNet12(64, plan["widths"]).to(gpu, memory_format=torch.channels_last)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=.1, mode="fan_out", nonlinearity="leaky_relu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.01, fused=True)
    history = []
    training_state = args.output / "resume.pt"
    if training_state.exists():
        saved = torch.load(training_state, map_location="cpu", weights_only=True)
        if saved["plan"] != plan:
            raise ValueError("Saved optimizer does not match the current plan")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
    if len(history) < args.epochs + args.meta_epochs:
        print("Loading the complete base split", flush=True)
        base = load_split(args.data_root, "train")
        if (selection_digest(base) != plan["train_selection_sha256"]
                or np.bincount(base.labels).tolist() != [600] * 64):
            raise ValueError("Training IDs or counts differ from the released split")
        images = torch.from_numpy(base.images).permute(0, 3, 1, 2).to(gpu)
        labels = base.labels
        targets = torch.from_numpy(labels).to(gpu)
        del base
    for epoch in range(len(history), args.epochs + args.meta_epochs):
        started = time.perf_counter()
        rng = np.random.default_rng(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        phase2 = epoch >= args.epochs
        fraction = epoch / args.epochs
        side = 40 if fraction < .25 else 56 if fraction < .5 else 72 if fraction < .75 else 84
        if phase2:
            lr = .00010 + .00020 * (1 + math.cos(math.pi * (epoch - args.epochs) / max(args.meta_epochs - 1, 1))) / 2
        else:
            cosine = (1 + math.cos(math.pi * max(0, epoch - 2) / max(args.epochs - 3, 1))) / 2
            lr = (.00012 + .00138 * cosine) * min(1., (epoch + 1) / 2)
        optimizer.param_groups[0]["lr"] = lr
        total = torch.zeros(2, device=gpu)
        model.train()
        for step, indices in enumerate(batches_by_class(labels, rng)):
            index = torch.as_tensor(indices, device=gpu)
            x = training_view(images[index], side)
            y = targets[index]
            # Sample two examples of every class for the extra paired alignment view.
            paired = torch.arange(0, len(y), 4, device=gpu)
            expanded = torch.cat([x, x[paired].flip(-1)]).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=dtype):
                scores, features = model(expanded, return_features=True)
            scores, features = scores.float(), features.float()
            loss = F.cross_entropy(scores, torch.cat([y, y[paired]]), label_smoothing=.05)
            unit = F.normalize(features)
            loss = loss + .5 * distribution_alignment(unit[paired], unit[len(y):])
            if phase2:
                loss = loss + .25 * episode_loss(features[:len(y)], 1 if step % 2 == 0 else 5)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total[0] += loss.detach()
            total[1] += (scores[:len(y)].detach().argmax(1) == y).sum()
        loss_sum, correct = total.tolist()
        if not math.isfinite(loss_sum):
            raise FloatingPointError("Training loss became nonfinite")
        row = {"epoch": epoch + 1, "phase": 2 if phase2 else 1, "side": side, "lr": lr,
               "loss": loss_sum / 300, "augmented_train_accuracy": correct / len(labels),
               "seconds": round(time.perf_counter() - started, 2)}
        history.append(row)
        atomic_save(training_state, {"plan": plan, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                     "scaler": scaler.state_dict(), "history": history})
        print(json.dumps(row), flush=True)
    atomic_save(args.output / "fresh.pt", {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": 64, "widths": plan["widths"], "purpose": "independent base-class training",
                "metadata": {"plan": plan}})
    (args.output / "backbone_training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--output", type=Path, default=Path(".local/lab04-gpu-backbone"))
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--meta-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261206)
    train(parser.parse_args())
