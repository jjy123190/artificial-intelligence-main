"""Create an explicit SHA-256 delivery manifest after successful read-only checks.
Use --refresh-reviewed only after reviewing intentional changes to an existing seal.
This script never updates a stale validation record or changes experiment results.
"""
import argparse
import datetime
import hashlib
import json
from delivery_files import BASE, CHECKSUM_FILE, POLICY, delivery_files


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-reviewed", action="store_true")
    args = parser.parse_args()
    if CHECKSUM_FILE.exists() and not args.refresh_reviewed:
        parser.error("Manifest already exists. Review changes, then use --refresh-reviewed explicitly.")
    checks = json.loads((BASE / "evidence/delivery_validation.json").read_text(encoding="utf-8"))
    assert checks["status"] == "passed"
    assert checks.get("offline_gallery_verified") is True, "Run the final delivery verifier first"
    inventory = delivery_files()
    hashes = {rel: sha(path) for rel, path in inventory.items()}
    assert hashes == checks["verified_delivery_files_sha256"], "Delivery changed after verification; re-verify it."
    record = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "algorithm": "sha256",
        "policy": POLICY,
        "reviewed_validation_at": checks["verified_at"],
        "file_count": len(hashes),
        "files": hashes,
    }
    CHECKSUM_FILE.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Sealed {len(hashes)} reviewed delivery files: {CHECKSUM_FILE}")


if __name__ == "__main__":
    main()