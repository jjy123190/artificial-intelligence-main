"""按明确的文件白名单导出独立源码，不打包文档、生成结果或已安装依赖。

此脚本保留在开发项目中；导出的目录仅含运行代码、测试源码与必要输入配置。
目标目录必须不存在，避免覆盖已有提交材料。验收记录由调用方保存在包外。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "__init__.py", "main.py", "requirements.txt", "requirements-web.txt",
    "requirements-lab01.txt", "requirements-lab02.txt", "requirements-plots.txt",
    "start_web.ps1", "stop_web.ps1", "run_lab02.ps1",
    "打开网站.cmd", "停止网站.cmd", "启动实验二.cmd", "运行实验.cmd",
)
RUNTIME_SCRIPTS = ("__init__.py", "web_python.ps1", "prepare_lab02.py", "package_project.py")
LAB01_FILES = {"__init__.py", "dataset.py", "main.py", "service.py", "benchmark.py", "viz.py", "打开网站.cmd"}


def source_files(root: Path):
    root = root.resolve()
    paths = {root / name for name in ROOT_FILES}
    for folder in ("core", "backend", "tests"):
        paths.update((root / folder).rglob("*.py"))
    paths.add(root / "experiments/__init__.py")
    for path in (root / "experiments/lab02").rglob("*"):
        if path.suffix in {".py", ".ps1", ".cmd"}:
            paths.add(path)
    for path in (root / "experiments/lab01").rglob("*"):
        relative = path.relative_to(root / "experiments/lab01")
        if (len(relative.parts) == 1 and path.name in LAB01_FILES) or (
                relative.parts[0] in {"kr", "reasoning"} and path.suffix == ".py"):
            paths.add(path)
    paths.update(root / "scripts" / name for name in RUNTIME_SCRIPTS)
    paths.update((root / "data/raw").glob("*.csv"))
    paths.update(root / "data" / name for name in ("source_hashes.json", "example_graph.json", "example_observations.json"))
    paths.update(root / "web" / name for name in ("package.json", "package-lock.json", "tsconfig.json", "vite.config.ts", "index.html"))
    for folder, suffixes in (("src", {".ts", ".tsx", ".css"}), ("public", {".svg"}), ("scripts", {".mjs"})):
        paths.update(p for p in (root / "web" / folder).rglob("*") if p.suffix in suffixes)
    for path in sorted(paths):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"源码缺失或不是本项目内的普通文件：{relative}")
        yield path


def export(destination: Path, root: Path = ROOT) -> dict:
    root, destination = root.resolve(), destination.resolve()
    files = list(source_files(root))
    if destination.exists():
        raise FileExistsError(f"目标目录已存在，拒绝覆盖：{destination}")
    destination.mkdir(parents=True)
    for source in files:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # 提交的是实验二，默认入口直达实验二；公共平台和所需的共享源码完整保留。
    launcher = destination / "打开网站.cmd"
    launcher.write_bytes((destination / "启动实验二.cmd").read_bytes())
    hashes = {str(p.relative_to(destination)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(destination.rglob("*")) if p.is_file()}
    return {"destination": str(destination), "files": len(hashes), "source_sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.destination), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
