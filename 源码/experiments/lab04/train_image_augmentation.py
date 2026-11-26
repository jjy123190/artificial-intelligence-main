"""Matched continuation of the base-only ResNet12 with optional blur and small rotations."""
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
from .train_medium_gpu import medium_model
from .train_rotation_views import calibration_batches, objective


BRANCHES = ("control", "augmented")
POLICY = {"rotation_probability": 0.4, "degrees": 12.0, "blur_probability": 0.2, "sigma": 0.8}


def rotate_images(images, degrees):
    radians = torch.deg2rad(degrees)
    cosine, sine = radians.cos(), radians.sin()
    zero = torch.zeros_like(radians)
    affine = torch.stack((cosine, -sine, zero, sine, cosine, zero), dim=1).view(-1, 2, 3)
    grid = F.affine_grid(affine, images.shape, align_corners=False)
    return F.grid_sample(images, grid, padding_mode="reflection", align_corners=False)


def blur_images(images, sigma):
    coords = images.new_tensor([-1, 0, 1])
    kernel = (-coords.square() / (2 * sigma ** 2)).exp()
    kernel = kernel / kernel.sum()
    channels = images.size(1)
    weights = torch.outer(kernel, kernel).expand(channels, 1, 3, 3).contiguous()
    return F.conv2d(F.pad(images, (1, 1, 1, 1), mode="reflect"), weights, groups=channels)


def augment_training_batch(images, generator, policy=POLICY):
    """Operate on private training views, drawing only from the extra augmentation RNG."""
    rotation_draw, angle_draw, blur_draw = torch.rand(
        (3, len(images)), device=images.device, generator=generator
    )
    rotate = (rotation_draw < policy["rotation_probability"]).nonzero().flatten()
    blur = (blur_draw < policy["blur_probability"]).nonzero().flatten()
    result = images.clone()
    if rotate.numel():
        angles = (2 * angle_draw[rotate] - 1) * policy["degrees"]
        result[rotate] = rotate_images(result[rotate], angles)
    if blur.numel():
        result[blur] = blur_images(result[blur], policy["sigma"])
    return result, (rotate.numel(), blur.numel())


def train_branch(name, out, plan, state, weights, base, images, targets, dtype):
    directory = out / name
    directory.mkdir(exist_ok=True)
    plan_hash = digest_file(out / "training_plan.json")
    record_file = directory / "training.json"
    if record_file.exists():
        record = json_from(record_file)
        if record["plan_sha256"] != plan_hash or record["branch"] != name:
            raise ValueError("The branch export belongs to another training plan")
        verified_files(directory, record["files"])
        return
    model = medium_model().to("cuda", memory_format=torch.channels_last)
    model.load_state_dict(weights)
    head = nn.Linear(256, 256).cuda()
    head.load_state_dict(state["head"])
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=plan["lr_max"],
                                  weight_decay=.03, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    history = []
    checkpoint = directory / "resume.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["plan_sha256"] != plan_hash or saved["branch"] != name:
            raise ValueError("The checkpoint belongs to another branch or run")
        model.load_state_dict(saved["model"])
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        del saved
    for epoch in range(len(history), plan["epochs"]):
        started = time.perf_counter()
        seed = plan["seed"] + epoch
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        extra_rng = torch.Generator(device="cuda").manual_seed(seed + 1000000)
        cosine = (1 + math.cos(math.pi * epoch / max(1, plan["epochs"] - 1))) / 2
        lr = plan["lr_min"] + (plan["lr_max"] - plan["lr_min"]) * cosine
        optimizer.param_groups[0]["lr"] = lr
        rotated = blurred = seen = 0
        total_loss = torch.zeros((), device="cuda")
        for step, batch in enumerate(batches_by_class(base.labels, rng)):
            indices = torch.as_tensor(batch, device="cuda")
            x = training_view(images[indices], 84)
            if name == "augmented":
                x, (n_rotate, n_blur) = augment_training_batch(x, extra_rng)
                rotated += n_rotate
                blurred += n_blur
            seen += len(indices)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(model, head, x, targets[indices], step, rng, dtype)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach()
        loss_value = (total_loss / (step + 1)).item()
        if seen != 38400 or step != 299 or not math.isfinite(loss_value):
            raise RuntimeError("Incomplete epoch or non-finite loss")
        row = dict(epoch=epoch + 1, updates=step + 1, images=seen, rotated=rotated,
                   blurred=blurred, loss=loss_value, lr=lr, seconds=round(time.perf_counter() - started, 2))
        history.append(row)
        atomic_save(checkpoint, {"plan_sha256": plan_hash, "branch": name, "model": model.state_dict(),
                                "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(), "history": history})
        print(name, row, flush=True)
    with torch.autocast("cuda", dtype=dtype):
        update_bn(calibration_batches(base.images, base.labels, plan["seed"]), model, device="cuda")
    atomic_save(directory / "medium.pt", {k: v.cpu() for k, v in model.state_dict().items()})
    save_json(record_file, {"plan_sha256": plan_hash, "branch": name, "history": history,
                            "files": {"medium.pt": digest_file(directory / "medium.pt")}})


def train(args):
    if not torch.cuda.is_available() or args.epochs < 1:
        raise ValueError("A CUDA device and positive training budget are required")
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    parent = json_from(args.parent / "results.json")
    verified_files(args.parent, {n: parent["files"][n] for n in ("medium.pt", "plan.json", "training.json")})
    parent_plan = json_from(args.parent / "plan.json")["training"]
    parent_log = json_from(args.parent / "training.json")
    state = torch.load(args.rotation_state, map_location="cpu", weights_only=True)
    weights = torch.load(args.parent / "medium.pt", map_location="cpu", weights_only=True)
    if (parent["track"] != "rotation_views" or parent["winner"] != "last" or parent["external_pretraining"]
            or parent_plan["external_weights"] or state["plan"] != parent_plan or state["history"] != parent_log["history"]
            or parent_plan["train_ids"] != json_from(args.artifacts / "protocol.json")["train_selection_sha256"]
            or any(not torch.equal(weights[k], state["model"][k]) for k, _ in medium_model().named_parameters())):
        raise ValueError("Expected the exported base-only model and its matching rotation head")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    plan = {"branches": list(BRANCHES), "epochs": args.epochs, "batch": 128, "seed": args.seed,
            "lr_max": 0.00016, "lr_min": 0.000025, "augmentation": POLICY, "external_weights": False,
            "train_ids": parent_plan["train_ids"], "parent_sha256": parent["files"]["medium.pt"],
            "rotation_state_sha256": digest_file(args.rotation_state), "gpu": torch.cuda.get_device_name(),
            "precision": str(dtype), "torch": str(torch.__version__),
            "source_sha256": {n: digest_file(Path(__file__).with_name(n)) for n in
                              ("train_image_augmentation.py", "train_rotation_views.py", "train_gpu_backbone.py",
                               "train_resnet_rotation.py", "train_medium_gpu.py")},
            "selection": "Highest sum of 1-shot and 5-shot fixed validation correct counts; ties favor parent, control, augmented."}
    file = out / "training_plan.json"
    if file.exists() and json_from(file) != plan:
        raise ValueError("Training plan changed; use a new output directory")
    save_json(file, plan)
    print("Loading 64 base classes on", plan["gpu"], flush=True)
    base = load_archive(args.data_root, args.artifacts, "train")
    if np.bincount(base.labels).tolist() != [600] * 64:
        raise ValueError("Expected 600 training images per base class")
    images = torch.from_numpy(base.images).permute(0, 3, 1, 2).cuda()
    targets = torch.from_numpy(base.labels).cuda()
    for name in BRANCHES:
        train_branch(name, out, plan, state, weights, base, images, targets, dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--parent", type=Path, default=Path("outputs/lab04_gpu_rotation_views"))
    parser.add_argument("--rotation-state", type=Path, default=Path(".local/lab04-rotation-views/resume.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--output", type=Path, default=Path(".local/lab04-image-augmentation"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261209)
    train(parser.parse_args())
