"""Stable file inventory for this report delivery (not the application workspace)."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = BASE / "evidence/delivery_checksums.json"
EXCLUDED_RECORDS = {
    "evidence/delivery_checksums.json",  # A manifest cannot hash itself.
    "evidence/delivery_validation.json",  # Refreshed by read-only verification.
    "evidence/gallery_validation.json",  # Refreshed by read-only browser QA.
}
POLICY = (
    "All report, figure, frozen evidence and reproduction files are protected. "
    "Exclude this manifest, refreshable verification records, bytecode and generated "
    "reproduce/qa previews. The original protected_before.json baseline IS included."
)


def delivery_files():
    files = {}
    for path in sorted(BASE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(BASE).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc" or rel in EXCLUDED_RECORDS:
            continue
        if rel.startswith("reproduce/qa/") and rel != "reproduce/qa/protected_before.json":
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(BASE):
            raise ValueError(f"Unexpected link outside delivery: {rel}")
        files[rel] = path
    return files