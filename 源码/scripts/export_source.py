"""Export all three experiments as source only, with explicit input/configuration files.

Generated results, reports, recordings, archives, installed dependencies and private
state never enter the export. The destination must not exist and cannot be inside
this repository. Material-generation Python tools are source, not their outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "__init__.py", "main.py", "README.md", ".gitignore", ".gitattributes",
    "requirements.txt", "requirements-web.txt", "requirements-plots.txt",
    "requirements-lab01.txt", "requirements-lab02.txt", "requirements-lab03.txt",
    "start_web.ps1", "stop_web.ps1", "run_lab02.ps1",
    "打开网站.cmd", "停止网站.cmd", "启动实验二.cmd", "运行实验.cmd",
)
DATA_FILES = (
    "data/source_hashes.json", "data/example_graph.json", "data/example_observations.json",
    "data/lab03/benchmark.json", "data/lab03/brief.json",
    "data/lab03/ga_warm_start.json", "data/lab03/llm_model_manifest.json",
)
WEB_FILES = ("package.json", "package-lock.json", "tsconfig.json", "vite.config.ts", "index.html")
CODE_FOLDERS = {
    "core": {".py"}, "backend": {".py"}, "experiments": {".py", ".ps1", ".cmd", ".txt"},
    "tests": {".py"}, "scripts": {".py", ".ps1", ".cmd"},
    "web/src": {".ts", ".tsx", ".css"}, "web/public": {".svg"},
    "web/scripts": {".mjs"}, "data/raw": {".csv"},
}


def source_files(root: Path = ROOT):
    root = root.resolve()
    paths = {root / name for name in ROOT_FILES + DATA_FILES}
    paths.update(root / "web" / name for name in WEB_FILES)
    for folder, suffixes in CODE_FOLDERS.items():
        paths.update(path for path in (root / folder).rglob("*")
                     if path.suffix.lower() in suffixes and "__pycache__" not in path.parts)
    for path in sorted(paths):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"缺少源码/输入配置，或路径不是项目内普通文件：{relative}")
        yield path


def export(destination: Path, root: Path = ROOT) -> dict:
    root, destination = root.resolve(), destination.resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("源码导出目标必须位于项目目录之外")
    if destination.exists():
        raise FileExistsError(f"目标目录已存在，拒绝覆盖：{destination}")
    files = list(source_files(root))
    destination.mkdir(parents=True)
    for source in files:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    hashes = {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(destination.rglob("*")) if path.is_file()}
    return {"destination": str(destination), "files": len(hashes), "source_sha256": hashes,
            "all_three_labs": True, "generated_artifacts_included": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export(args.destination)
    print(json.dumps({key: value for key, value in result.items() if key != "source_sha256"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
