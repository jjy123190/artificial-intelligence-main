"""Train the 2M-parameter ResNet12 independently, retaining train-only CE and alignment."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .accuracy90_data import load_archive, json_from
from .data import digest_file
from .models import ResNet12, distribution_alignment
from .train_gpu_backbone import atomic_save, batches_by_class, episode_loss, training_view
from .train_gpu_metric import save_json


def medium_model():
    model = ResNet12(64, (32, 64, 128, 256))
    for block in model.encoder:
        for group in (block.convolutions, block.shortcut):
            for i, layer in enumerate(group):
                if isinstance(layer, nn.GroupNorm):
                    group[i] = nn.BatchNorm2d(layer.num_channels)
    return model


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled Python runtime is required")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs
    if epochs < 8:
        raise ValueError("At least eight epochs are required for this schedule")
    source = Path(__file__)
    plan = {"source": {p.name: digest_file(p) for p in
                       [source, source.with_name("models.py"), source.with_name("train_gpu_backbone.py"),
                        source.with_name("accuracy90_data.py")]},
            "seed": args.seed, "epochs": epochs, "batch": 128, "widths": [32, 64, 128, 256],
            "external_weights": False, "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "train_ids": json_from(args.artifacts / "protocol.json")["train_selection_sha256"]}
    plan_file = output / "training_plan.json"
    if plan_file.exists() and json_from(plan_file) != plan:
        raise ValueError("Existing training directory belongs to a different plan")
    save_json(plan_file, plan)
    checkpoint = output / "resume.pt"
    model = medium_model().to("cuda", memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.025, fused=True)
    history = []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["plan"] != plan:
            raise ValueError("Mismatched training checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        del state
    if len(history) < epochs:
        print("Loading verified training archive", flush=True)
        base = load_archive(args.data_root, args.artifacts, "train")
        images = torch.from_numpy(base.images).permute(0, 3, 1, 2).cuda()
        labels = base.labels
        targets = torch.from_numpy(labels).cuda()
        del base
    for epoch in range(len(history), epochs):
        started = time.monotonic()
        rng = np.random.default_rng(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        progress = epoch / (epochs - 1)
        side = 48 if progress < .28 else 64 if progress < .72 else 80
        lr = (.00009 + .0014 * (1 + math.cos(math.pi * progress)) / 2) * min(1., (epoch + 1) / 3)
        optimizer.param_groups[0]["lr"] = lr
        totals = torch.zeros(2, device="cuda")
        for step, indices in enumerate(batches_by_class(labels, rng)):
            idx = torch.as_tensor(indices, device="cuda")
            y = targets[idx]
            x = training_view(images[idx], side)
            paired = torch.arange(step % 8, len(y), 8, device="cuda")
            combined = torch.cat([x, x[paired].flip(-1)]).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, z = model(combined, return_features=True)
            logits, z = logits.float(), z.float()
            loss = F.cross_entropy(logits, torch.cat([y, y[paired]]), label_smoothing=.06)
            z = F.normalize(z)
            loss = loss + .5 * distribution_alignment(z[paired], z[len(y):])
            if progress >= .80:
                loss = loss + .2 * episode_loss(z[:len(y)], 1 if step % 2 == 0 else 5)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals[0] += loss.detach()
            totals[1] += (logits[:len(y)].detach().argmax(1) == y).sum()
        loss_sum, correct = totals.tolist()
        if not math.isfinite(loss_sum):
            raise FloatingPointError("Training loss became nonfinite")
        row = {"epoch": epoch + 1, "train_accuracy": correct / len(labels), "loss": loss_sum / 300,
               "input_size": side, "seconds": round(time.monotonic() - started, 2)}
        history.append(row)
        atomic_save(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "history": history, "plan": plan})
        print(row, flush=True)
    atomic_save(output / "medium.pt", {k: v.cpu() for k, v in model.state_dict().items()})
    save_json(output / "training.json", history)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path(".local/lab04-medium"))
    p.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    p.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261127)
    train(p.parse_args())
