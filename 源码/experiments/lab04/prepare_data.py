"""Fetch and verify the standard Ravi/Larochelle mini-ImageNet split.

The dataset is for non-commercial research/education under ImageNet's terms.
Only JPEG/PNG files are extracted; archive links and path traversal are rejected.
No pickle or executable dataset payload is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.request

from PIL import Image


SOURCE_PAGE = "https://yaoyaoliu.web.illinois.edu/projects/mtl/download/Lmzjm9tX.html"
SPLITS = {
    "train": {"id": "107FTosYIeBn5QbynR46YG91nHcJ70whs", "bytes": 125900800, "md5": "62af9b3c839974dad2d474e6325795af", "classes": 64},
    "val": {"id": "1hSMUMj5IRpf-nQs1OwgiQLmGZCN0KDWl", "bytes": 30791680, "md5": "ab02f050b0bf66823e7acb0c1ac1bc6b", "classes": 16},
    "test": {"id": "1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v", "bytes": 39178240, "md5": "318185fc3e3bf8bc57de887d9682c666", "classes": 20},
}
CSV_BASE = "https://raw.githubusercontent.com/yaoyao-liu/mini-imagenet-tools/master/csv_files/"
CLASS_NAMES_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"


def hashes(path: Path) -> dict:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest(), "bytes": path.stat().st_size}


def download(url: str, target: Path, expected_bytes: int | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    started = time.monotonic()
    next_report = 4 * 1024 * 1024
    print(f"Downloading {target.name}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type and target.suffix == ".tar":
            raise RuntimeError("Google Drive returned an HTML confirmation/error page, not an archive")
        with partial.open("wb") as stream:
            total = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
                total += len(chunk)
                if total >= next_report:
                    print(f"{target.name}: {total / 1024**2:.1f} MiB, {time.monotonic()-started:.1f}s", flush=True)
                    next_report += 4 * 1024 * 1024
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise ValueError(f"Wrong archive size for {target.name}: {partial.stat().st_size} != {expected_bytes}")
    partial.replace(target)


def official_classes(root: Path, split: str) -> tuple[set[str], dict]:
    csv_path = root / "archives" / f"{split}.csv"
    url = CSV_BASE + f"{split}.csv"
    if not csv_path.exists():
        download(url, csv_path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = {row["label"] for row in rows}
    return labels, {"url": url, "csv_rows": len(rows), **hashes(csv_path)}


def extract_images(archive: Path, root: Path, split: str, labels: set[str]) -> int:
    output = root / split
    output.mkdir(parents=True, exist_ok=True)
    files = 0
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsafe non-file member: {member.name}")
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if name.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"Unexpected archive member: {member.name}")
            candidates = [part for part in name.parts if part in labels]
            if not candidates:
                candidates = [label for label in labels if name.name.startswith(label)]
            if len(candidates) != 1:
                raise ValueError(f"Cannot identify unique class: {member.name}")
            target = output / candidates[0] / name.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            files += 1
    return files


def validate_split(root: Path, split: str, expected_labels: set[str]) -> dict:
    split_root = root / split
    actual_labels = {item.name for item in split_root.iterdir() if item.is_dir()}
    if actual_labels != expected_labels:
        raise ValueError(f"Split class mismatch: {split}, extra={actual_labels-expected_labels}, missing={expected_labels-actual_labels}")
    entries, counts, sizes, modes = [], {}, {}, {}
    for label in sorted(actual_labels):
        paths = sorted((split_root / label).iterdir())
        if len(paths) != 600:
            raise ValueError(f"{split}/{label}: expected 600 images, got {len(paths)}")
        counts[label] = len(paths)
        for path in paths:
            with Image.open(path) as image:
                image.load()
                size_key = f"{image.width}x{image.height}"
                sizes[size_key] = sizes.get(size_key, 0) + 1
                modes[image.mode] = modes.get(image.mode, 0) + 1
                if image.size != (84, 84):
                    raise ValueError(f"Image not 84x84: {path}")
            entries.append({"path": path.relative_to(root).as_posix(), "label": label})
    with (root / f"{split}_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(entries)
    return {"classes": len(counts), "images": len(entries), "class_counts": counts, "sizes": sizes, "modes": modes, "all_images_decoded": True}


def prepare(root: Path, selected: list[str]) -> dict:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "dataset_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {
        "name": "mini-ImageNet", "split_protocol": "Ravi & Larochelle (2017), 64/16/20 disjoint classes",
        "source_page": SOURCE_PAGE, "purpose": "Non-commercial educational experiment",
        "splits": {},
    }
    split_labels = {}
    for split in selected:
        expected = SPLITS[split]
        labels, csv_info = official_classes(root, split)
        if len(labels) != expected["classes"]:
            raise ValueError(f"Unexpected official CSV class count: {split}")
        split_labels[split] = labels
        url = f"https://drive.usercontent.google.com/download?id={expected['id']}&export=download&confirm=t"
        archive = root / "archives" / f"{split}.tar"
        if not archive.exists():
            download(url, archive, expected["bytes"])
        archive_info = hashes(archive)
        if archive_info["md5"] != expected["md5"] or archive_info["bytes"] != expected["bytes"]:
            raise ValueError(f"Archive does not match original author's published MD5/size: {split}")
        marker = root / split / ".extraction_complete"
        if not marker.exists() or marker.read_text(encoding="ascii").strip() != archive_info["sha256"]:
            count = extract_images(archive, root, split, labels)
            print(f"Extracted {split}: {count} images", flush=True)
            marker.write_text(archive_info["sha256"], encoding="ascii")
        validation = validate_split(root, split, labels)
        summary["splits"][split] = {"archive": {"url": url, **archive_info}, "official_split_csv": csv_info, **validation}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Verified {split}: {validation['classes']} classes / {validation['images']} decoded images", flush=True)
    names = list(summary["splits"])
    for index, left in enumerate(names):
        left_labels = set(summary["splits"][left]["class_counts"])
        for right in names[index + 1:]:
            if left_labels & set(summary["splits"][right]["class_counts"]):
                raise ValueError(f"Overlapping split classes: {left}/{right}")
    summary["class_splits_disjoint"] = True
    class_index_path = root / "archives" / "imagenet_class_index.json"
    if not class_index_path.exists():
        download(CLASS_NAMES_URL, class_index_path)
    class_index = json.loads(class_index_path.read_text(encoding="utf-8"))
    class_names = {item[0]: item[1] for item in class_index.values()}
    used_labels = {label for split_info in summary["splits"].values() for label in split_info["class_counts"]}
    missing_names = used_labels - class_names.keys()
    if missing_names:
        raise ValueError(f"Class names missing from ImageNet index: {missing_names}")
    (root / "class_names.json").write_text(json.dumps({label: class_names[label] for label in sorted(used_labels)}, indent=2), encoding="utf-8")
    summary["class_names_source"] = {"url": CLASS_NAMES_URL, **hashes(class_index_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    args = parser.parse_args()
    prepare(args.root, args.splits)
