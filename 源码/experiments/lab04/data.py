"""Validated mini-ImageNet image loading and disjoint episode sampling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


@dataclass
class ImageSplit:
    images: np.ndarray
    labels: np.ndarray
    ids: list[str]
    class_names: list[str]
    split: str

    @property
    def by_class(self):
        return {int(label): np.flatnonzero(self.labels == label)
                for label in np.unique(self.labels)}


def image_records(data_root: str | Path, split: str):
    root = Path(data_root)
    alternatives = ("val", "validation") if split == "val" else (split,)
    folder = next((root / name for name in alternatives if (root / name).is_dir()), None)
    if folder is None:
        raise FileNotFoundError(f"Missing {split} folder in {root}")
    records = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        relative = path.relative_to(folder)
        match = re.search(r"n\d{8}", str(relative))
        if match:
            label = match.group()
        elif len(relative.parts) > 1:
            label = relative.parts[0]
        else:
            raise ValueError(f"Cannot infer class from {path}")
        records.append((label, path, f"{split}/{relative.as_posix()}"))
    if not records:
        raise ValueError(f"No images found in {folder}")
    return records


def verify_splits(data_root):
    expected = {"train": 64, "val": 16, "test": 20}
    class_sets = {}
    details = {}
    for split, count in expected.items():
        records = image_records(data_root, split)
        names = sorted({row[0] for row in records})
        if len(names) != count:
            raise ValueError(f"Expected {count} classes in {split}, found {len(names)}")
        class_sets[split] = set(names)
        details[split] = {"classes": len(names), "images": len(records), "class_names": names}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = class_sets[first] & class_sets[second]
        if overlap:
            raise ValueError(f"Class leakage between {first} and {second}: {overlap}")
    return details


def load_split(data_root, split, *, max_per_class=None, seed=20261122):
    records = image_records(data_root, split)
    names = sorted({row[0] for row in records})
    mapping = {name: index for index, name in enumerate(names)}
    rng = np.random.default_rng(seed)
    selected = []
    for name in names:
        rows = [row for row in records if row[0] == name]
        if max_per_class is not None and len(rows) > max_per_class:
            indices = np.sort(rng.choice(len(rows), max_per_class, replace=False))
            rows = [rows[int(index)] for index in indices]
        selected.extend(rows)
    images = np.empty((len(selected), 84, 84, 3), dtype=np.uint8)
    for index, (_, path, _) in enumerate(selected):
        with Image.open(path) as image:
            images[index] = np.asarray(image.convert("RGB").resize((84, 84), Image.Resampling.BICUBIC))
    return ImageSplit(images=images,
                      labels=np.array([mapping[row[0]] for row in selected], dtype=np.int64),
                      ids=[row[2] for row in selected], class_names=names, split=split)


def normalized_tensor(images, *, augment=False, generator=None):
    inputs = torch.from_numpy(np.asarray(images)).permute(0, 3, 1, 2).float() / 255
    if augment:
        flip = torch.rand(len(inputs), generator=generator) < 0.5
        inputs[flip] = inputs[flip].flip(-1)
        # Reflection crop and mild photometric augmentation are applied only
        # during training. No augmentation is applied when extracting results.
        inputs = F.pad(inputs, (4, 4, 4, 4), mode="reflect")
        offset = torch.randint(0, 9, (2,), generator=generator)
        inputs = inputs[:, :, int(offset[0]):int(offset[0]) + 84,
                        int(offset[1]):int(offset[1]) + 84]
        gain = 0.85 + 0.3 * torch.rand(len(inputs), 3, 1, 1, generator=generator)
        inputs = (inputs * gain).clamp(0, 1)
    mean = inputs.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = inputs.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return ((inputs - mean) / std).contiguous()


def sample_episode(labels, rng, *, ways=5, shots=1, queries=15):
    labels = np.asarray(labels)
    available = np.unique(labels)
    if len(available) < ways:
        raise ValueError("Not enough classes for episode")
    selected_classes = rng.choice(available, ways, replace=False)
    support_ids, query_ids, support_labels, query_labels = [], [], [], []
    for episode_label, original in enumerate(selected_classes):
        candidates = np.flatnonzero(labels == original)
        if len(candidates) < shots + queries:
            raise ValueError("Not enough independent support/query examples")
        chosen = rng.choice(candidates, shots + queries, replace=False)
        support_ids.extend(chosen[:shots])
        query_ids.extend(chosen[shots:])
        support_labels.extend([episode_label] * shots)
        query_labels.extend([episode_label] * queries)
    # Shuffle queries so position cannot reveal their labels to prediction.
    query_order = rng.permutation(len(query_ids))
    return {"classes": selected_classes.astype(int),
            "support_indices": np.array(support_ids, dtype=int),
            "query_indices": np.array(query_ids, dtype=int)[query_order],
            "support_labels": np.array(support_labels, dtype=int),
            "query_labels": np.array(query_labels, dtype=int)[query_order]}


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_digest(split: ImageSplit):
    return hashlib.sha256("\n".join(split.ids).encode("utf-8")).hexdigest()
