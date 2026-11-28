"""Matched CE/alignment training and exact all-parameter MAML on mini-ImageNet base classes."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .models import distribution_alignment, maml_adapt
from .train_gpu_backbone import atomic_save, batches_by_class, training_view
from .train_gpu_metric import save_json
from .train_medium_gpu import medium_model

SEED = 20261128
WARM, EPOCHS = 8, 20
META_EPOCHS, META_TASKS = 6, 64
INNER_LR, INNER_STEPS = .06, 1
META_CANDIDATES = (3, 6)
ALIGN = .75
DEFAULT_WORK = Path(".local/lab04-unified256")
ARTIFACTS = Path("outputs/lab04")
DATA = Path(".local/mini-imagenet")


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError("This training/replay recipe requires CUDA.")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False


def schedule(epoch):
    p = (epoch - 1) / (EPOCHS - 1)
    lr = (.00013 + .00145 * (1 + math.cos(math.pi * p)) / 2) * min(1., epoch / 2)
    return (48 if epoch <= WARM else 64 if epoch <= 16 else 80), lr


def cpu_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def init_maml(ce, seed=SEED):
    torch.manual_seed(seed)
    model = medium_model()
    model.load_state_dict(ce)
    model.classifier = nn.Linear(256, 5)
    nn.init.normal_(model.classifier.weight, std=.01)
    nn.init.zeros_(model.classifier.bias)
    # Freeze running statistics, NOT trainable BN affine parameters. All tasks
    # start from the same buffers; no support/query batch updates or leakage.
    return model.eval()


def plan(artifacts):
    source = Path(__file__).parent
    files = [Path(__file__).name, "accuracy90_data.py", "models.py", "train_medium_gpu.py",
             "train_gpu_backbone.py", "train_gpu_metric.py", "data.py"]
    return {
        "seed": SEED, "channels": [32, 64, 128, 256], "normalization": "BatchNorm2d",
        "embedding": "global average pooling -> 256", "external_weights": False,
        "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "train_ids_sha256": json_from(artifacts / "protocol.json")["train_selection_sha256"],
        "train_images": 38400, "classes": 64,
        "source_sha256": {f: digest_file(source / f) for f in files},
        "supervised": {"init": "seeded PyTorch defaults; shared CE warmup, fork model AND AdamW",
                       "shared_epochs": WARM, "epochs_total_per_arm": EPOCHS, "batch": 128,
                       "extra_views": 32, "sampler": "all 38400 once/epoch; 16 classes x 8",
                       "augment": "independent training_view for 128 images and 32 paired images",
                       "schedule": [{"epoch": i, "pixels": schedule(i)[0], "lr": schedule(i)[1]}
                                    for i in range(1, EPOCHS + 1)],
                       "optimizer": "AdamW, betas .9/.999, weight_decay .025, fused",
                       "precision": "bf16 autocast; fp32 CE, moments and parameters",
                       "ce_label_smoothing": .05, "alignment_weights": {"ce": 0., "da": ALIGN},
                       "alignment": "mean+diagonal variance MSE on L2 pooled features x sqrt(256)",
                       "checkpoint_rule": "fixed final epoch in both arms; no per-arm selection"},
        "maml": {"init": "final CE encoder+BN buffers; fresh 5-way linear head N(0,.01), zero bias",
                 "epochs": META_EPOCHS, "train_tasks_per_epoch": META_TASKS,
                 "samples": "alternating 5-way 1/5-shot; five disjoint queries/class, base classes only",
                 "outer_tasks_per_update": 2, "inner_steps": INNER_STEPS, "inner_lr": INNER_LR,
                 "inner_parameters": "ALL; exact second order create_graph=True",
                 "training_pixels": 64, "precision": "fp32, TF32 permitted",
                 "bn": "eval for support AND query; affine weights still adapted",
                 "optimizer": "AdamW, weight_decay .01, clip norm 5",
                 "outer_lr_by_epoch": [meta_lr(e) for e in range(1, META_EPOCHS + 1)],
                 "candidates": list(META_CANDIDATES),
                 "validation_rule": "max correct across first 60 released validation tasks per shot; tie earlier"},
    }


def persist_plan(path, value):
    if path.exists() and json_from(path) != value:
        raise ValueError(f"Plan mismatch; use a new directory: {path}")
    save_json(path, value)


def supervised(name, work, images, labels, plan_hash):
    path = work / (name + "_resume.pt")
    begin, end = (1, WARM) if name == "warm" else (WARM + 1, EPOCHS)
    torch.manual_seed(SEED)
    model = medium_model().to("cuda", memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.025, fused=True)
    history = []
    source = path if path.exists() else work / "warm_resume.pt" if name != "warm" else None
    if source is not None:
        state = checkpoint(source)
        if state["plan"] != plan_hash:
            raise ValueError("Supervised checkpoint does not match training plan")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        history = state["history"]
        del state
    y_all = torch.as_tensor(labels, dtype=torch.long, device="cuda")
    align_weight = ALIGN if name == "da" else 0.
    for epoch in range(max(begin, len(history) + 1), end + 1):
        tic = time.perf_counter()
        model.train()
        side, lr = schedule(epoch)
        opt.param_groups[0]["lr"] = lr
        rng = np.random.default_rng(SEED + epoch)
        torch.manual_seed(SEED + epoch)
        fingerprint = hashlib.sha256()
        totals = torch.zeros(3, device="cuda")
        steps = 0
        for step, ids in enumerate(batches_by_class(labels, rng)):
            fingerprint.update(np.asarray(ids, dtype="<i8").tobytes())
            idx = torch.as_tensor(ids, device="cuda", dtype=torch.long)
            raw, y = images[idx], y_all[idx]
            paired = torch.arange(step % 4, len(y), 4, device="cuda")
            x = torch.cat((training_view(raw, side), training_view(raw[paired], side)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, z = model(x.contiguous(memory_format=torch.channels_last), return_features=True)
            logits = logits.float()
            ce = F.cross_entropy(logits, torch.cat((y, y[paired])), label_smoothing=.05)
            z = F.normalize(z.float()) * math.sqrt(256)
            da = distribution_alignment(z[paired], z[len(y):])
            loss = ce + align_weight * da
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals += torch.stack((ce.detach(), da.detach(), (logits[:len(y)].argmax(1) == y).sum()))
            steps += 1
        ce, da, right = totals.tolist()
        if not math.isfinite(ce + da) or steps != 300:
            raise RuntimeError("Incomplete or nonfinite supervised epoch")
        row = {"epoch": epoch, "phase": name, "lr": lr, "pixels": side, "steps": steps,
               "ce": ce / steps, "alignment_stat": da / steps, "alignment_weight": align_weight,
               "loss": (ce + align_weight * da) / steps, "train_accuracy": right / len(labels),
               "draw_sha256": fingerprint.hexdigest(), "seconds": round(time.perf_counter() - tic, 2)}
        history.append(row)
        atomic_save(path, {"model": cpu_state(model), "opt": opt.state_dict(), "history": history,
                           "plan": plan_hash})
        print(json.dumps(row), flush=True)
    if [r["epoch"] for r in history] != list(range(1, end + 1)):
        raise ValueError("Incomplete epoch history")
    save_json(work / (name + "_history.json"), history)
    if name != "warm" and not (work / (name + ".pt")).exists():
        atomic_save(work / (name + ".pt"), cpu_state(model))


def meta_lr(epoch):
    return .000035 + .000145 * (1 + math.cos(math.pi * (epoch - 1) / (META_EPOCHS - 1))) / 2


def draw_meta(groups, rng, shots):
    classes = rng.choice(len(groups), 5, replace=False)
    rows = np.stack([rng.choice(groups[c], shots + 5, replace=False) for c in classes])
    return rows[:, :shots].reshape(-1), rows[:, shots:].reshape(-1)


def meta_train(work, images, labels, plan_hash):
    path = work / "maml_resume.pt"
    ce = work / "ce.pt"
    init_hash = digest_file(ce)
    model = init_maml(checkpoint(ce)).to("cuda", memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=meta_lr(1), weight_decay=.01)
    groups = [np.flatnonzero(labels == c) for c in range(64)]
    history = []
    if path.exists():
        state = checkpoint(path)
        if state["plan"] != plan_hash or state["ce_sha256"] != init_hash:
            raise ValueError("Mismatched MAML checkpoint")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        history = state["history"]
        del state
    for epoch in range(len(history) + 1, META_EPOCHS + 1):
        tic = time.perf_counter()
        lr = meta_lr(epoch)
        opt.param_groups[0]["lr"] = lr
        rng = np.random.default_rng(SEED + 9000 + epoch)
        torch.manual_seed(SEED + 9000 + epoch)
        totals = torch.zeros(3, device="cuda")
        opt.zero_grad(set_to_none=True)
        for t in range(META_TASKS):
            shots = 1 if t % 2 == 0 else 5
            si, qi = draw_meta(groups, rng, shots)
            sx, qx = [training_view(images[torch.as_tensor(ids, dtype=torch.long, device="cuda")], 64)
                      for ids in (si, qi)]
            sy = torch.arange(5, device="cuda").repeat_interleave(shots)
            qy = torch.arange(5, device="cuda").repeat_interleave(5)
            fast = maml_adapt(model, sx, sy, steps=INNER_STEPS, learning_rate=INNER_LR, second_order=True)
            logits = torch.func.functional_call(model, fast, (qx,))
            loss = F.cross_entropy(logits, qy)
            (loss / 2).backward()
            totals[0] += loss.detach()
            totals[1] += (logits.argmax(1) == qy).sum()
            if t % 2:
                norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                totals[2] += norm.detach()
                opt.step()
                opt.zero_grad(set_to_none=True)
            del fast, logits, loss
        ce_loss, right, grad = totals.tolist()
        row = {"epoch": epoch, "tasks": META_TASKS, "outer_steps": META_TASKS // 2, "outer_lr": lr,
               "query_ce": ce_loss / META_TASKS, "query_accuracy": right / (META_TASKS * 25),
               "outer_grad_norm_preclip": grad / (META_TASKS // 2),
               "seconds": round(time.perf_counter() - tic, 2)}
        history.append(row)
        if epoch in META_CANDIDATES:
            atomic_save(work / f"maml_e{epoch}.pt", cpu_state(model))
        atomic_save(path, {"model": cpu_state(model), "opt": opt.state_dict(), "history": history,
                           "ce_sha256": init_hash, "plan": plan_hash})
        print(json.dumps(row), flush=True)
    save_json(work / "maml_history.json", {"ce_sha256": init_hash, "epochs": history})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    setup()
    args.work.mkdir(parents=True, exist_ok=True)
    path = args.work / "training_plan.json"
    persist_plan(path, plan(args.artifacts))
    h = digest_file(path)
    base = load_archive(args.data_root, args.artifacts, "train")
    if np.bincount(base.labels).tolist() != [600] * 64:
        raise ValueError("Wrong base split")
    labels = base.labels
    images = torch.from_numpy(base.images).permute(0, 3, 1, 2).cuda()
    del base
    for name in ("warm", "ce", "da"):
        supervised(name, args.work, images, labels, h)
    meta_train(args.work, images, labels, h)


if __name__ == "__main__":
    main()

