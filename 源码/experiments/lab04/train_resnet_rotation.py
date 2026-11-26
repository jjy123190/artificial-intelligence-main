"""Continue the base-only ResNet12 with CutMix and joint class/rotation prediction."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .models import distribution_alignment
from .train_gpu_backbone import atomic_save, batches_by_class, episode_loss, training_view
from .train_gpu_metric import save_json
from .train_medium_gpu import medium_model


def cutmix(images, rng):
    h, w = images.shape[-2:]
    size = math.sqrt(1 - rng.beta(1., 1.))
    cx, cy = rng.integers(w), rng.integers(h)
    x0, x1 = max(0, cx - int(w * size / 2)), min(w, cx + int(w * size / 2))
    y0, y1 = max(0, cy - int(h * size / 2)), min(h, cy + int(h * size / 2))
    shuffled = torch.randperm(len(images), device=images.device)
    result = images.clone()
    result[:, :, y0:y1, x0:x1] = images[shuffled, :, y0:y1, x0:x1]
    return result, shuffled, (x1 - x0) * (y1 - y0) / (h * w)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("The training command requires CUDA")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    original = args.previous / "medium.pt"
    history_plan = json_from(args.previous / "plan.json")
    original_hash = digest_file(original)
    if (history_plan["external_pretraining"] or history_plan["training_plan"]["external_weights"]
            or original_hash != history_plan["backbone_sha256"]
            or history_plan["training_plan"]["train_ids"] != json_from(args.artifacts / "protocol.json")["train_selection_sha256"]):
        raise ValueError("Expected the verified base-class-only medium ResNet12")
    if args.epochs < 4:
        raise ValueError("Use at least four training epochs")
    source = Path(__file__)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    plan = {"seed": args.seed, "epochs": args.epochs, "parent_sha256": original_hash,
            "parent_training": history_plan["training_plan"], "external_weights": False,
            "train_ids": history_plan["training_plan"]["train_ids"], "widths": [32, 64, 128, 256],
            "input_size": 84, "batch": 128, "gpu": torch.cuda.get_device_name(),
            "dtype": str(dtype), "torch": str(torch.__version__),
            "source": {p.name: digest_file(p) for p in (source, source.with_name("train_gpu_backbone.py"),
                       source.with_name("train_medium_gpu.py"), source.with_name("models.py"))}}
    plan_file = args.output / "training_plan.json"
    if plan_file.exists() and json_from(plan_file) != plan:
        raise ValueError("The output directory already holds a different plan")
    save_json(plan_file, plan)
    model = medium_model().to("cuda", memory_format=torch.channels_last)
    model.load_state_dict(torch.load(original, map_location="cpu", weights_only=True))
    rotation = nn.Linear(256, 64 * 4).cuda()
    optimizer = torch.optim.AdamW([*model.parameters(), *rotation.parameters()], lr=.0006,
                                  weight_decay=.03, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    history, checkpoint = [], args.output / "resume.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["plan"] != plan:
            raise ValueError("The saved optimizer belongs to a different plan")
        model.load_state_dict(state["backbone"])
        rotation.load_state_dict(state["rotation"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        del state
    if len(history) < args.epochs:
        base = load_archive(args.data_root, args.artifacts, "train")
        images = torch.from_numpy(base.images).permute(0, 3, 1, 2).cuda()
        labels = base.labels
        targets = torch.from_numpy(labels).cuda()
        del base
        print("Training on 38,400 base images using", plan["gpu"], flush=True)
    for epoch in range(len(history), args.epochs):
        start = time.perf_counter()
        rng = np.random.default_rng(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        lr = (.000035 + .000565 * (1 + math.cos(math.pi * epoch / (args.epochs - 1))) / 2)
        optimizer.param_groups[0]["lr"] = lr * min(1., (epoch + 1) / 2)
        total = torch.zeros((), device="cuda")
        for step, indices in enumerate(batches_by_class(labels, rng)):
            idx = torch.as_tensor(indices, device="cuda")
            x, y = training_view(images[idx], 84), targets[idx]
            # Rotate a balanced subset of the clean batch. Rotation labels are generated from the view.
            select = torch.randperm(len(x), device="cuda")[:32]
            parts = x[select].chunk(4)
            rotated = torch.cat([torch.rot90(part, k, (-2, -1)) for k, part in enumerate(parts)])
            rotation_target = 4 * y[select] + torch.arange(4, device="cuda").repeat_interleave(8)
            mixed = step % 2 == 0
            supervised = x
            if mixed:
                supervised, perm, area = cutmix(x, rng)
            pair = torch.arange(step % 8, len(x), 8, device="cuda")
            views = torch.cat([supervised, rotated, supervised[pair].flip(-1)]).contiguous(
                memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=dtype):
                logits, features = model(views, return_features=True)
                rotation_logits = rotation(features[len(y):len(y) + len(rotated)])
            scores = logits[:len(y)].float()
            ce = F.cross_entropy(scores, y, label_smoothing=.04)
            if mixed:
                ce = (1 - area) * ce + area * F.cross_entropy(scores, y[perm], label_smoothing=.04)
            z = F.normalize(features.float())
            loss = ce + .5 * F.cross_entropy(rotation_logits.float(), rotation_target, label_smoothing=.04)
            loss = loss + .5 * distribution_alignment(z[pair], z[-len(pair):])
            if not mixed:
                loss = loss + .2 * episode_loss(z[:len(y)], 1 if step % 4 == 1 else 5)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach()
        value = total.item() / (step + 1)
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite training loss")
        history.append({"epoch": epoch + 1, "loss": value, "seconds": round(time.perf_counter() - start, 2)})
        atomic_save(checkpoint, {"plan": plan, "backbone": model.state_dict(), "rotation": rotation.state_dict(),
                                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "history": history})
        print(history[-1], flush=True)
    atomic_save(args.output / "medium.pt", {k: v.cpu() for k, v in model.state_dict().items()})
    save_json(args.output / "training.json", history)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_medium"))
    p.add_argument("--output", type=Path, default=Path(".local/lab04-rotation"))
    p.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    p.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=20261128)
    train(p.parse_args())
