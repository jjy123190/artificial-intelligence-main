"""Learn channel-distance pooling from base-class examples of a frozen ResNet12."""
import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .accuracy90_data import json_from, load_archive
from .data import digest_file, normalized_tensor
from .evaluate_gpu_tracks import verified_files
from .rotation_study import load_extractor
from .train_episode_metric import BATCH_TASKS, SEED, data_args, prototype_logits, sample_indices
from .train_gpu_metric import save_json


CHANNELS = 48
STEPS = 700
BATCH = 32
DIMENSION = 192


class DistanceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(256, CHANNELS, 1, bias=False)
        self.norm = nn.BatchNorm2d(CHANNELS)
        self.project = nn.Linear(CHANNELS * (CHANNELS + 1) // 2, DIMENSION, bias=False)
        self.register_buffer("upper", torch.triu_indices(CHANNELS, CHANNELS), persistent=False)

    def forward(self, maps):
        x = self.norm(self.reduce(maps)).flatten(2)
        gram = x @ x.transpose(1, 2) / x.size(2)
        diag = gram.diagonal(dim1=1, dim2=2)
        distances = (diag.unsqueeze(1) + diag.unsqueeze(2) - 2 * gram).clamp_min(1e-8).sqrt()
        # A channel is a vector of spatial responses. Double-center their distance matrix.
        # Explicitly zero its diagonal after the stabilized square root.
        distances = distances.masked_fill(torch.eye(CHANNELS, device=x.device, dtype=torch.bool), 0.)
        mean = distances.mean(2, keepdim=True)
        centered = distances - mean - mean.transpose(1, 2) + mean.mean(1, keepdim=True)
        pooled = centered[:, self.upper[0], self.upper[1]]
        return F.normalize(self.project(F.normalize(pooled)))


@torch.inference_mode()
def map_batches(encoder, images, device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    encoder.eval().to(device, memory_format=torch.channels_last)
    for start in range(0, len(images), BATCH):
        x = normalized_tensor(images[start:start + BATCH]).to(device, memory_format=torch.channels_last)
        yield encoder(x), encoder(x.flip(-1))


@torch.inference_mode()
def image_features(encoder, head, images, device):
    return torch.cat([F.normalize(head(a) + head(b)).cpu()
                      for a, b in map_batches(encoder, images, device)]).numpy()


def fuse(original, distance, weight):
    return np.concatenate([np.sqrt(1 - weight) * original, np.sqrt(weight) * distance],
                          axis=-1).astype(np.float32)


def prepare_maps(args):
    cache = args.map_cache
    cache.mkdir(parents=True, exist_ok=True)
    prior = json_from(args.previous / "results.json")
    if prior["external_pretraining"] or prior["track"] != "rotation_views":
        raise ValueError("Expected the self-trained parent")
    verified_files(args.previous, {"medium.pt": prior["files"]["medium.pt"]})
    plan = {"backbone": prior["files"]["medium.pt"], "manifest": digest_file(args.artifacts / "dataset_manifest.json"),
            "train_ids": json_from(args.artifacts / "protocol.json")["train_selection_sha256"],
            "format": "train archive order; original and horizontal flip; NCHW float16 on disk"}
    if (cache / "maps.json").exists():
        record = json_from(cache / "maps.json")
        if record["plan"] != plan:
            raise ValueError("Map cache differs from current data/backbone")
        verified_files(cache, record["files"])
        return
    base = load_archive(args.data_root, args.artifacts, "train")
    if len(base.class_names) != 64 or len(base.images) != 38400:
        raise ValueError("Expected all base classes")
    encoder = load_extractor(args.previous, "cuda").model.encoder
    files = ("maps_original.npy", "maps_flipped.npy", "labels.npy")
    arrays = [np.lib.format.open_memmap(cache / name, mode="w+", dtype=np.float16,
                                        shape=(len(base.images), 256, 5, 5)) for name in files[:2]]
    for i, pair in enumerate(map_batches(encoder, base.images, "cuda")):
        for arr, batch in zip(arrays, pair):
            arr[i * BATCH:i * BATCH + len(batch)] = batch.half().cpu().numpy()
        if (i + 1) % 300 == 0:
            print("Final spatial maps", (i + 1) * BATCH, "/", len(base.images), flush=True)
    for arr in arrays:
        arr.flush()
    np.save(cache / files[2], base.labels)
    save_json(cache / "maps.json", {"plan": plan, "files": {n: digest_file(cache / n) for n in files}})


def train_head(args):
    prepare_maps(args)
    labels = np.load(args.map_cache / "labels.npy")
    groups = np.stack([np.flatnonzero(labels == c) for c in np.unique(labels)])
    if groups.shape != (64, 600):
        raise ValueError("Base-class counts do not match")
    maps = [torch.from_numpy(np.load(args.map_cache / n)).cuda()
            for n in ("maps_original.npy", "maps_flipped.npy")]
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    head = DistanceHead().cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0007, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.00005)
    targets = torch.arange(5, device="cuda").repeat_interleave(15).repeat(BATCH_TASKS)
    history = []
    totals = torch.zeros(2, device="cuda")
    for step in range(STEPS):
        shots = (1, 5)[step % 2]
        ix = torch.as_tensor(sample_indices(groups, rng, shots), device="cuda").flatten()
        both = torch.cat([m[ix].float() for m in maps])
        a, b = head(both).chunk(2)
        z = F.normalize(a + b).reshape(BATCH_TASKS, 5, shots + 15, -1)
        logits = prototype_logits(z[:, :, :shots], z[:, :, shots:].flatten(1, 2), False)
        ce = F.cross_entropy(logits.flatten(0, 1), targets, label_smoothing=.04)
        alignment = ((a.mean(0) - b.mean(0)).square().sum() +
                     (a.std(0, unbiased=False) - b.std(0, unbiased=False)).square().sum())
        loss = ce + .25 * alignment
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        totals += torch.stack((ce, alignment)).detach()
        if (step + 1) % 100 == 0:
            values = (totals / 100).tolist()
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite head training")
            history.append({"step": step + 1, "ce": values[0], "view_alignment": values[1]})
            totals.zero_()
            print("Spatial head", history[-1], flush=True)
    return head.cpu().eval(), history


if __name__ == "__main__":
    from pathlib import Path
    p = argparse.ArgumentParser(description="Cache final feature maps of the 64 base classes")
    data_args(p)
    p.add_argument("--map-cache", type=Path, default=Path(".local/lab04-distance-covariance"))
    torch.set_num_threads(2)
    prepare_maps(p.parse_args())
