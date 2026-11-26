"""Continue the base-only ResNet12 with joint rotation labels and self-distillation across rotations."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, update_bn

from .accuracy90_data import json_from, load_archive
from .data import digest_file, normalized_tensor
from .evaluate_gpu_tracks import verified_files
from .models import distribution_alignment
from .train_gpu_backbone import atomic_save, batches_by_class, episode_loss, training_view
from .train_gpu_metric import save_json
from .train_medium_gpu import medium_model
from .train_resnet_rotation import cutmix


def rotation_targets(joint_logits):
    # View-major blocks; joint label 4 * class + rotation. Pool only matching angles.
    by_view = joint_logits.reshape(4, -1, 64, 4)
    class_logits = torch.stack([by_view[k, :, :, k] for k in range(4)]).mean(0)
    return (class_logits.detach() / 2).softmax(-1)


def objective(model, head, x, y, step, rng, dtype):
    selected = torch.arange(step % 8, len(x), 8, device=x.device)
    upright = x[selected]
    rotated = torch.cat([torch.rot90(upright, k, (-2, -1)) for k in range(4)])
    rotation_ids = torch.arange(4, device=x.device).repeat_interleave(len(selected))
    labels = 4 * y[selected].repeat(4) + rotation_ids
    mixed = step % 2 == 0
    supervised = x
    if mixed:
        supervised, perm, area = cutmix(x, rng)
    views = torch.cat((supervised, rotated, upright.flip(-1))).contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=dtype):
        logits, features = model(views, return_features=True)
        joint = head(features[len(y):len(y) + len(rotated)]).float()
    main = logits[:len(y)].float()
    ce = F.cross_entropy(main, y, label_smoothing=.04)
    if mixed:
        ce = (1 - area) * ce + area * F.cross_entropy(main, y[perm], label_smoothing=.04)
    original_logits = logits[len(y):len(y) + len(upright)].float()
    distill = 4 * F.kl_div((original_logits / 2).log_softmax(1), rotation_targets(joint), reduction="batchmean")
    joint_ce = F.cross_entropy(joint, labels, label_smoothing=.04)
    z = F.normalize(features.float())
    align = distribution_alignment(z[len(y):len(y) + len(upright)], z[-len(upright):])
    episode = z.new_zeros(()) if mixed else episode_loss(z[:len(y)], 1 if step % 4 == 1 else 5)
    loss = ce + .5 * joint_ce + .5 * distill + .5 * align + .2 * episode
    return loss, torch.stack((loss, ce, joint_ce, distill, align, episode)).detach()


def calibration_batches(images, labels, seed):
    # Balanced clean images and flips, one pass over all base IDs on each export.
    for idx in batches_by_class(labels, np.random.default_rng(seed)):
        view = normalized_tensor(images[idx])
        yield torch.cat((view, view.flip(-1))).contiguous(memory_format=torch.channels_last)


def train(args):
    if not torch.cuda.is_available() or args.epochs < 4:
        raise ValueError("At least four epochs and a CUDA GPU are required")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    parent = json_from(args.parent / "results.json")
    verified_files(args.parent, {n: parent["files"][n] for n in ("medium.pt", "plan.json", "continuation_training.json")})
    previous = json_from(args.parent / "plan.json")
    source = args.rotation_state
    state = torch.load(source, map_location="cpu", weights_only=True)
    weights = torch.load(args.parent / "medium.pt", map_location="cpu", weights_only=True)
    ids = json_from(args.artifacts / "protocol.json")["train_selection_sha256"]
    if (previous["external_pretraining"] or previous["training"]["external_weights"]
            or previous["training"]["parent_training"]["external_weights"] or previous["training"]["train_ids"] != ids
            or previous["training"] != state["plan"] or state["history"] != json_from(args.parent / "continuation_training.json")
            or state["backbone"].keys() != weights.keys() or any(not torch.equal(v, state["backbone"][k]) for k, v in weights.items())):
        raise ValueError("Expected the base-only backbone and its matching trained rotation head")
    deps = ("train_rotation_views.py", "train_resnet_rotation.py", "train_gpu_backbone.py", "train_medium_gpu.py",
            "models.py", "data.py", "accuracy90_data.py")
    first_average = args.epochs - args.epochs // 2 + 1
    plan = {"epochs": args.epochs, "average_from_epoch": first_average, "seed": args.seed, "batch": 128,
            "train_ids": ids, "external_weights": False, "rotation_state_sha256": digest_file(source),
            "parent_sha256": parent["files"]["medium.pt"], "torch": str(torch.__version__), "dtype": str(dtype),
            "gpu": torch.cuda.get_device_name(), "source": {n: digest_file(Path(__file__).with_name(n)) for n in deps},
            "selection": "Single backbone with most correct across both fixed validation shots. Ties favor parent, last, then average. Fixed parent views and inference.",
            "bn_calibration": "For both trained candidates, one balanced pass over every clean base image and its flip."}
    file = out / "training_plan.json"
    if file.exists() and json_from(file) != plan:
        raise ValueError("Use a separate output directory for another training plan")
    save_json(file, plan)
    if (out / "training.json").exists():
        record = json_from(out / "training.json")
        if record["plan_sha256"] != digest_file(file):
            raise ValueError("Training plan differs from export")
        verified_files(out, record["files"])
        return
    model = medium_model().to("cuda", memory_format=torch.channels_last)
    model.load_state_dict(weights)
    head = nn.Linear(256, 256).cuda()
    head.load_state_dict(state["rotation"])
    del state, weights
    averaged = AveragedModel(model)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=.00045, weight_decay=.03, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    checkpoint = out / "resume.pt"
    history = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["plan"] != plan:
            raise ValueError("Continuation checkpoint does not match the plan")
        model.load_state_dict(saved["model"])
        head.load_state_dict(saved["head"])
        averaged.load_state_dict(saved["average"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        del saved
    print("Loading base images on", plan["gpu"], flush=True)
    base = load_archive(args.data_root, args.artifacts, "train")
    if np.bincount(base.labels).tolist() != [600] * 64:
        raise ValueError("Expected all 64 base classes")
    images = torch.from_numpy(base.images).permute(0, 3, 1, 2).cuda()
    targets = torch.from_numpy(base.labels).cuda()
    for ep in range(len(history), args.epochs):
        started = time.perf_counter()
        rng = np.random.default_rng(args.seed + ep)
        torch.manual_seed(args.seed + ep)
        lr = (.000045 + .000405 * (1 + math.cos(math.pi * ep / (args.epochs - 1))) / 2) * min(1., (ep + 1) / 2)
        optimizer.param_groups[0]["lr"] = lr
        total = torch.zeros(6, device="cuda")
        for step, ids_batch in enumerate(batches_by_class(base.labels, rng)):
            ids_batch = torch.as_tensor(ids_batch, device="cuda")
            x, y = training_view(images[ids_batch], 84), targets[ids_batch]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = objective(model, head, x, y, step, rng, dtype)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += parts
        values = (total / (step + 1)).tolist()
        if step != 299 or not all(map(math.isfinite, values)):
            raise RuntimeError("Non-finite loss or incomplete base epoch")
        if ep + 1 >= first_average:
            averaged.update_parameters(model)
        row = dict(zip(("loss", "ce", "rotation_ce", "kd", "alignment", "episodic"), values))
        row.update(epoch=ep + 1, updates=step + 1, lr=lr, seconds=round(time.perf_counter() - started, 2))
        history.append(row)
        atomic_save(checkpoint, {"plan": plan, "model": model.state_dict(), "head": head.state_dict(),
                                 "average": averaged.state_dict(), "optimizer": optimizer.state_dict(),
                                 "scaler": scaler.state_dict(), "history": history})
        print(row, flush=True)
    del images, targets
    exports = {}
    if int(averaged.n_averaged) != args.epochs - first_average + 1:
        raise RuntimeError("Incorrect number of averaged checkpoints")
    for name, backbone in (("last", model), ("average", averaged.module)):
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype):
            update_bn(calibration_batches(base.images, base.labels, args.seed), backbone, device="cuda")
        directory = out / name
        directory.mkdir(exist_ok=True)
        atomic_save(directory / "medium.pt", {k: v.cpu() for k, v in backbone.state_dict().items()})
        exports[name] = round(time.perf_counter() - started, 2)
    save_json(out / "training.json", {"history": history, "bn_seconds": exports, "plan_sha256": digest_file(file),
              "files": {f"{n}/medium.pt": digest_file(out / n / "medium.pt") for n in exports}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--parent", type=Path, default=Path("outputs/lab04_gpu_rotation"))
    parser.add_argument("--rotation-state", type=Path, default=Path(".local/lab04-rotation/resume.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--output", type=Path, default=Path(".local/lab04-rotation-views"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261201)
    train(parser.parse_args())
