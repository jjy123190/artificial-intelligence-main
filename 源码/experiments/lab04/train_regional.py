"""Train the base-only ResNet12 with an auxiliary classifier on intermediate image regions."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import update_bn

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .evaluate_gpu_tracks import verified_files
from .train_gpu_backbone import atomic_save, batches_by_class, training_view
from .train_gpu_metric import save_json
from .train_image_augmentation import augment_training_batch
from .train_medium_gpu import medium_model
from .train_rotation_views import calibration_batches, objective


def regional_loss(region_head, maps, labels):
    regions = F.adaptive_avg_pool2d(maps, (3, 3)).flatten(2).transpose(1, 2)
    scores = region_head(regions).float()
    targets = labels[:, None].expand(-1, scores.size(1))
    losses = F.cross_entropy(scores.transpose(1, 2), targets, label_smoothing=.04, reduction="none")
    # An image-level class need not describe every background region.
    return losses.topk(6, largest=False).values.mean()


def train(args):
    torch.set_num_threads(2)
    archive = json_from(args.control / "plan.json")
    base_plan = archive["training"]
    parent = args.parent / "medium.pt"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if (archive["track"] != "image_augmentation" or base_plan["external_weights"] or
            base_plan["train_ids"] != json_from(args.artifacts / "protocol.json")["train_selection_sha256"] or
            base_plan["parent_sha256"] != digest_file(parent) or
            base_plan["rotation_state_sha256"] != digest_file(args.rotation_state) or
            base_plan["gpu"] != torch.cuda.get_device_name() or
            base_plan["torch"] != str(torch.__version__) or base_plan["precision"] != str(dtype)):
        raise ValueError("Expected the matched base-only augmentation run and its starting weights")
    verified_files(Path(__file__).parent, base_plan["source_sha256"])
    plan = {"matched_training": base_plan, "matched_plan_sha256": digest_file(args.control / "plan.json"),
            "source_sha256": digest_file(Path(__file__)), "auxiliary_weight": .35, "head_lr_multiple": 8,
            "regions": "Block 3, 3x3 adaptive average pooling, 6 lowest patch CEs per unmixed image",
            "selection": "One backbone and readout for both shots. Maximize summed fixed validation correct. "
                         "Tie order: augmented, rotation_views, regional, augmented_kernel, regional_kernel.",
            "kernel": {"weight": .2, "temperature": .2, "grid": 3}}
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / "training_plan.json"
    if plan_file.exists() and json_from(plan_file) != plan:
        raise ValueError("Use a separate directory for a different training plan")
    save_json(plan_file, plan)
    plan_hash = digest_file(plan_file)
    record = out / "training.json"
    if record.exists():
        log = json_from(record)
        if log["plan_sha256"] != plan_hash:
            raise ValueError("Training plan differs from its export")
        verified_files(out, log["files"])
        return

    torch.manual_seed(base_plan["seed"])
    model = medium_model().to("cuda", memory_format=torch.channels_last)
    model.load_state_dict(torch.load(parent, map_location="cpu", weights_only=True))
    rotation = nn.Linear(256, 256).cuda()
    rotation.load_state_dict(torch.load(args.rotation_state, map_location="cpu", weights_only=True)["head"])
    region_head = nn.Linear(128, 64).cuda()
    optimizer = torch.optim.AdamW([{"params": [*model.parameters(), *rotation.parameters()]},
                                   {"params": region_head.parameters()}], lr=base_plan["lr_max"],
                                  weight_decay=.03, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    resume = out / "resume.pt"
    history = []
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["plan_sha256"] != plan_hash:
            raise ValueError("Mismatched regional training checkpoint")
        for name, component in (("model", model), ("rotation", rotation), ("region_head", region_head),
                                ("optimizer", optimizer), ("scaler", scaler)):
            component.load_state_dict(state[name])
        history = state["history"]
        del state
    print("Loading base classes on", base_plan["gpu"], flush=True)
    base = load_archive(args.data_root, args.artifacts, "train")
    if np.bincount(base.labels).tolist() != [600] * 64:
        raise ValueError("Expected all 38,400 base images")
    images = torch.from_numpy(base.images).permute(0, 3, 1, 2).cuda()
    targets = torch.from_numpy(base.labels).cuda()
    control_log = json_from(args.control / "training.json")["augmented"]["history"]
    epochs = base_plan["epochs"]
    for epoch in range(len(history), epochs):
        started = time.perf_counter()
        seed = base_plan["seed"] + epoch
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        extra_rng = torch.Generator(device="cuda").manual_seed(seed + 1000000)
        cosine = (1 + math.cos(math.pi * epoch / max(1, epochs - 1))) / 2
        lr = base_plan["lr_min"] + (base_plan["lr_max"] - base_plan["lr_min"]) * cosine
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * plan["head_lr_multiple"]
        total = torch.zeros(2, device="cuda")
        seen = rotated = blurred = auxiliary_steps = 0
        for step, batch in enumerate(batches_by_class(base.labels, rng)):
            idx = torch.as_tensor(batch, device="cuda")
            y = targets[idx]
            x, (nr, nb) = augment_training_batch(training_view(images[idx], 84), extra_rng, base_plan["augmentation"])
            seen += len(idx)
            rotated += nr
            blurred += nb
            optimizer.zero_grad(set_to_none=True)
            if step % 2:  # Only these batches contain 128 single-label images before the extra views.
                block_outputs = []
                with model.encoder[2].register_forward_hook(lambda module, inputs, output: block_outputs.append(output)):
                    loss, _ = objective(model, rotation, x, y, step, rng, dtype)
                with torch.autocast("cuda", dtype=dtype):
                    local_ce = regional_loss(region_head, block_outputs.pop()[:len(y)], y)
                loss = loss + plan["auxiliary_weight"] * local_ce
                total[1] += local_ce.detach()
                auxiliary_steps += 1
            else:
                loss, _ = objective(model, rotation, x, y, step, rng, dtype)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total[0] += loss.detach()
        loss_sum, local_sum = total.tolist()
        counts = dict(epoch=epoch + 1, updates=step + 1, images=seen, rotated=rotated, blurred=blurred, lr=lr)
        if (any(control_log[epoch][key] != value for key, value in counts.items()) or
                auxiliary_steps != 150 or not all(map(math.isfinite, (loss_sum, local_sum)))):
            raise RuntimeError("Augmentation, budget or learning rate differs from the matched run, or loss is nonfinite")
        row = dict(**counts, loss=loss_sum / (step + 1), region_ce=local_sum / auxiliary_steps,
                   region_updates=auxiliary_steps, seconds=round(time.perf_counter() - started, 2))
        history.append(row)
        atomic_save(resume, {"plan_sha256": plan_hash, "model": model.state_dict(), "rotation": rotation.state_dict(),
                             "region_head": region_head.state_dict(), "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "history": history})
        print(row, flush=True)
    del images, targets
    with torch.autocast("cuda", dtype=dtype):
        update_bn(calibration_batches(base.images, base.labels, base_plan["seed"]), model, device="cuda")
    atomic_save(out / "medium.pt", {k: v.cpu() for k, v in model.state_dict().items()})
    save_json(record, {"plan_sha256": plan_hash, "history": history, "files": {"medium.pt": digest_file(out / "medium.pt")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--control", type=Path, default=Path("outputs/lab04_gpu_image_augmentation"))
    parser.add_argument("--parent", type=Path, default=Path("outputs/lab04_gpu_rotation_views"))
    parser.add_argument("--rotation-state", type=Path, default=Path(".local/lab04-rotation-views/resume.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--output", type=Path, default=Path(".local/lab04-regional"))
    train(parser.parse_args())
