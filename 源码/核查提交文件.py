"""Read-only validation of the shipped file inventory; no training or network."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT / "文件校验清单.json").read_text(encoding="utf-8"))
    missing, changed, unsafe = [], [], []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
            unsafe.append(name)
        elif not path.is_file():
            missing.append(name)
        elif path.stat().st_size != expected["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            changed.append(name)
    ignored = {".venv", "venv", "node_modules", "__pycache__", ".state", ".git", ".pytest_cache", ".local"}
    extra = []
    import os
    for base, dirs, names in os.walk(ROOT):
        relative = Path(base).relative_to(ROOT)
        dirs[:] = [d for d in dirs if d not in ignored and not (relative == Path("outputs") and d == "runtime")]
        for name in names:
            path = Path(base) / name
            rel = path.relative_to(ROOT).as_posix()
            if rel not in manifest["files"] and rel != "文件校验清单.json" and not name.endswith((".pyc", ".tsbuildinfo")):
                extra.append(rel)
    result = {"status": "passed" if not (missing or changed or unsafe) else "failed", "checked_files": len(manifest["files"]), "missing": missing, "changed": changed, "unsafe_paths": unsafe, "extra_files": extra, "note": "额外文件单独列出；不修改或删除任何文件。"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1

if __name__ == "__main__":
    sys.exit(main())
