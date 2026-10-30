"""Replay sealed lab02 results without rewriting their provenance.

The original strict CLI fingerprints every Python file, including newly added
media helpers. This command first requires every sealed source and raw data file
to match byte-for-byte, rejects unexpected additions, then invokes the unmodified
strict verifier against an exact read-only-source copy of that recorded scope.
It does not regenerate benchmarks, alter hashes, or claim the current whole-tree
fingerprint equals the historical one.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs/qa/lab02-ppt-video-3min"
sys.path.insert(0, str(ROOT))

from experiments.lab02.eval.artifacts import execution_manifest, fingerprint
from experiments.lab02.eval.verification import verify_results

ALLOWED_ADDITIONS = {
    "scripts/create_lab02_ppt_video.py", "scripts/record_lab02_demo.py",
    "scripts/build_lab02_three_minute.py", "scripts/verify_lab02_recorded_snapshot.py",
    "scripts/test_lab02_ppt_video.py",
}


def main():
    folders = ["demo", "experiments", "end_to_end", "benchmarks", "governance"]
    manifests = {folder: json.loads((ROOT / "outputs/lab02" / folder / "manifest.json").read_text(encoding="utf-8-sig")) for folder in folders}
    before = {folder: fingerprint(ROOT / "outputs/lab02" / folder / "manifest.json") for folder in folders}
    recorded = manifests["demo"]
    current = execution_manifest(ROOT)
    for folder, manifest in manifests.items():
        if manifest["source_sha256"] != recorded["source_sha256"] or manifest["snapshot_sha256"] != recorded["snapshot_sha256"]:
            raise RuntimeError(f"Recorded source scopes differ: {folder}")
    changed = [name for name, expected in recorded["source_sha256"].items() if current["source_sha256"].get(name) != expected]
    additions = sorted(set(current["source_sha256"]) - set(recorded["source_sha256"]))
    if changed or set(additions) - ALLOWED_ADDITIONS:
        raise RuntimeError(f"Algorithm/source drift: changed={changed}, unexpected additions={sorted(set(additions) - ALLOWED_ADDITIONS)}")
    if current["snapshot_sha256"] != recorded["snapshot_sha256"]:
        raise RuntimeError("Raw data changed")
    scope = WORK / ("recorded-source-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    scope.mkdir(parents=True, exist_ok=False)
    for name in recorded["source_sha256"]:
        source = (ROOT / name).resolve()
        destination = (scope / name).resolve()
        source.relative_to(ROOT.resolve())
        destination.relative_to(scope.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (scope / "data/raw").mkdir(parents=True)
    for name in recorded["snapshot_sha256"]:
        shutil.copyfile(ROOT / "data/raw" / name, scope / "data/raw" / name)
    audit = {"strict_whole_tree_command": "not equivalent: explicitly allowed added media helpers",
             "recorded_source_files_unchanged": len(recorded["source_sha256"]),
             "raw_data_files_unchanged": len(recorded["snapshot_sha256"]), "added_media_helpers": additions,
             "scope_copy": str(scope.relative_to(ROOT)), "old_manifests_rewritten": False}
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    result = verify_results(ROOT / "outputs/lab02", scope, full_replay=True)
    after = {folder: fingerprint(ROOT / "outputs/lab02" / folder / "manifest.json") for folder in folders}
    if before != after:
        raise RuntimeError("Sealed manifests unexpectedly changed")
    audit["replay"] = result
    audit["manifest_hashes_unchanged"] = after
    (WORK / "source-scope-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
