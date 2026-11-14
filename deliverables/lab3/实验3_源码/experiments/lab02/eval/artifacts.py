"""统一的 UTF-8 输出、CSV、代码/数据指纹与统计汇总。"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execution_manifest(root: Path) -> dict:
    excluded = {"__pycache__", "output", "node_modules", "dist", "build", ".git", ".venv", "venv", "_归档", "outputs", ".state", "人工智能作业一", "人工智能作业二"}
    paths = []
    for directory, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if d not in excluded]
        paths.extend(Path(directory) / name for name in filenames if name.endswith(".py"))
    code = {str(p.relative_to(root)).replace("\\", "/"): fingerprint(p) for p in sorted(paths)}
    data = {p.name: fingerprint(p) for p in sorted((root / "data" / "raw").glob("*.csv"))}
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "source_sha256": code, "snapshot_sha256": data,
            "runtime_note": "壁钟时间是本机实测；与硬件、系统负载、解释器有关，不能保证跨机器逐值一致"}


def aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, values in groups.items():
        result = dict(zip(keys, key))
        result["repetitions"] = len(values)
        for metric in metrics:
            data = [v[metric] for v in values if v.get(metric) is not None]
            result[metric + "_mean"] = statistics.mean(data) if data else None
            result[metric + "_sd"] = statistics.stdev(data) if len(data) > 1 else 0.0 if data else None
        out.append(result)
    return out


def seal_manifest(output: Path, manifest: dict) -> None:
    """在所有结果写完后封存文本/数值产物。PNG由独立绘图流程生成，不绑定旧预览图。"""
    extensions = {".csv", ".json", ".jsonl", ".txt", ".md"}
    files = sorted(p for p in output.rglob("*") if p.is_file()
                   and p.name != "manifest.json" and p.suffix in extensions)
    manifest["artifact_sha256"] = {p.relative_to(output).as_posix(): fingerprint(p) for p in files}
    manifest["artifact_scope"] = "本次生成的文本、模型、原始输入与数值结果；不含独立重绘PNG或manifest自身"
    write_json(output / "manifest.json", manifest)
