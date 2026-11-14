"""Check a source-only workspace; never delete files or write experiment outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.export_source import source_files


def inspect_source(root: Path = ROOT) -> dict:
    root = root.resolve()
    expected = {path.relative_to(root).as_posix() for path in source_files(root)}
    actual = set()
    forbidden_directories = []
    symlinks = []
    generated_names = {"outputs", "deliverables", "node_modules", "__pycache__",
                       ".state", ".venv", "venv", ".pytest_cache", "dist"}
    for current, directories, files in os.walk(root, followlinks=False):
        current = Path(current)
        for name in list(directories):
            child = current / name
            if child.is_symlink():
                symlinks.append(child.relative_to(root).as_posix())
                directories.remove(name)
            elif child == root / ".git":
                directories.remove(name)  # Keep existing version history, never export it.
            elif name in generated_names:
                forbidden_directories.append(child.relative_to(root).as_posix())
        for name in files:
            path = current / name
            if path.is_symlink():
                symlinks.append(path.relative_to(root).as_posix())
            actual.add(path.relative_to(root).as_posix())
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    return {"status": "passed" if not (unexpected or missing or symlinks or forbidden_directories) else "failed",
            "root": str(root), "files": len(actual),
            "bytes": sum((root / path).stat().st_size for path in actual if (root / path).is_file()),
            "unexpected_files": unexpected, "missing_files": missing,
            "generated_directories": sorted(forbidden_directories), "symlinks": sorted(symlinks),
            "git_metadata_exempt": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = inspect_source(args.root)
    except (OSError, ValueError) as exc:
        result = {"status": "failed", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
