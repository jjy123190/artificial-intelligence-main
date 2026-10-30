"""Regenerate all formal evidence, verify it, then install the checked files locally."""
from __future__ import annotations

import csv
import io
import json
import math
import shutil
import sys
import zipfile
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.lab02.eval.artifacts import execution_manifest, fingerprint, write_json
from experiments.lab02.eval.benchmark import run_benchmarks
from experiments.lab02.eval.end_to_end import EndToEndConfig, run_end_to_end
from experiments.lab02.eval.experiments import ExperimentConfig, run_experiments
from experiments.lab02.eval.governance_experiments import run_governance_experiments
from experiments.lab02.eval.verification import verify_results
from experiments.lab02.main import _plots, run_demo

OUT = Path(__file__).resolve().parent


def compare_saved_predictions(archive, stage):
    comparisons = []
    for relative, keys in (
        ("experiments/predictions.csv", ("seed", "scenario", "sample_id")),
        ("experiments/conflict_predictions.csv", ("seed", "conflict_rate", "sample_id")),
        ("end_to_end/predictions.csv", ("seed", "scenario", "sample_id", "model")),
    ):
        before = list(csv.DictReader(io.StringIO(archive.read("outputs/lab02/" + relative).decode("utf-8-sig"))))
        with (stage / relative).open(encoding="utf-8-sig", newline="") as handle:
            after = list(csv.DictReader(handle))
        lookup = {tuple(row[k] for k in keys): row for row in after}
        assert len(before) == len(after) == len(lookup)
        checks = 0
        for row in before:
            current = lookup[tuple(row[k] for k in keys)]
            assert set(current) == set(row)
            for field, old_value in row.items():
                if field.endswith(("_ms", "_us")):
                    continue
                new_value = current[field]
                if old_value != new_value:
                    assert math.isclose(float(old_value), float(new_value), rel_tol=1e-12, abs_tol=1e-12), (relative, field)
                checks += 1
        comparisons.append({"file": relative, "rows": len(after), "non_timing_values_compared": checks})
    return comparisons


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tests = json.loads((OUT / "test-summary.json").read_text(encoding="utf-8"))
    current = execution_manifest(ROOT)
    assert tests["successful"] and tests["source_sha256"] == current["source_sha256"]
    stage = OUT / ("formal-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    stage.mkdir(exist_ok=False)
    before = json.loads((OUT / "before-fixes.json").read_text(encoding="utf-8"))
    with (OUT / "formal-rerun.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        def progress(message):
            line = f"{datetime.now(timezone.utc).isoformat()} {message}"
            print(line, flush=True)
            print(line, file=sys.__stdout__, flush=True)

        progress("Regenerating formal demo")
        run_demo(stage)
        progress("Regenerating indicator experiments and explicit ablation records")
        run_experiments(stage / "experiments", ExperimentConfig(), progress=progress)
        progress("Regenerating graph-level experiments")
        run_end_to_end(stage / "end_to_end", EndToEndConfig(), progress=progress)
        progress("Regenerating governance experiments")
        run_governance_experiments(stage / "governance", sizes=(8, 32, 64, 128), repeats=3, progress=progress)
        progress("Regenerating deterministic benchmarks")
        run_benchmarks(stage / "benchmarks", groups=(1, 4, 8, 16), repeats=3, progress=progress)
        progress("Regenerating all figures from the new CSVs")
        figures = _plots(stage, required=True)
        progress("Verifying hashes, all scores and all graph decisions")
        replay = verify_results(stage, ROOT, full_replay=True)
        write_json(stage / "qa/programming_acceptance.json", replay)
        with zipfile.ZipFile(ROOT / before["archive"]) as archive:
            comparison = compare_saved_predictions(archive, stage)
        for relative, digest in before["protected_sha256"].items():
            assert fingerprint(ROOT / relative) == digest, relative
        assert execution_manifest(ROOT)["source_sha256"] == current["source_sha256"]
        progress("Full replay passed; installing the verified formal files")
        destination = ROOT / "outputs/lab02"
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        for path in files:
            target = destination / path.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            assert fingerprint(target) == fingerprint(path)
        report = {"completed_utc": datetime.now(timezone.utc).isoformat(), "successful": True,
                  "staged_results": stage.relative_to(ROOT).as_posix(),
                  "installed_files": len(files), "figures": len(figures),
                  "test_count": tests["tests_run"], "full_replay": replay,
                  "prediction_comparisons": comparison,
                  "non_timing_predictions_unchanged_tolerance": 1e-12,
                  "protected_inputs_and_deliverables_unchanged": True,
                  "source_sha256": current["source_sha256"],
                  "backup": before["archive"]}
        write_json(OUT / "formal-results-check.json", report)
        progress(json.dumps({k: report[k] for k in ("successful", "installed_files", "figures", "test_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
