"""Freeze the submitted implementation and its latest measured results for the report."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

WORK = Path(__file__).resolve().parent
PROJECT = WORK.parents[2]
RELEASE = Path(r"C:\Users\Administrator\Desktop\人工智能实验2\第31组_实验二源码")
OUTPUT = RELEASE / "outputs/lab02"
sys.path.insert(0, str(RELEASE))
from experiments.lab02.eval.artifacts import execution_manifest

def read(name):
    with (OUTPUT / name).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))

tables = {key: read(name) for key, name in {
    "forward": "benchmarks/forward_summary.csv", "slicing": "benchmarks/goal_slicing.csv",
    "resolution": "benchmarks/resolution.csv", "indicator": "experiments/summary.csv",
    "sensitivity": "experiments/sensitivity.csv", "ablation": "experiments/ablations.csv",
    "conflict": "experiments/conflict_summary.csv", "graph": "end_to_end/summary.csv",
    "equity": "governance/equity_summary.csv",
}.items()}
current = execution_manifest(RELEASE)
for family in ("demo", "experiments", "end_to_end", "benchmarks", "governance"):
    manifest = json.loads((OUTPUT / family / "manifest.json").read_text(encoding="utf-8"))
    assert current["source_sha256"] == manifest["source_sha256"], family
    assert current["snapshot_sha256"] == manifest["snapshot_sha256"], family
    for relative, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((OUTPUT / family / relative).read_bytes()).hexdigest() == expected, relative

def row(table, **conditions):
    return next(r for r in tables[table] if all(r[k] == v for k, v in conditions.items()))

speedup = float(row("forward", groups="16", model="indexed_naive")["elapsed_ms_mean"]) / float(row("forward", groups="16", model="semi_naive")["elapsed_ms_mean"])
groups = defaultdict(list)
for r in tables["sensitivity"]:
    groups[(r["model"], r["parameter"], float(r["value"]))].append(float(r["f1"]))
sensitivity = [{"model": k[0], "parameter": k[1], "value": k[2], "mean": statistics.mean(v), "sd": statistics.stdev(v)} for k, v in sorted(groups.items())]
models = {name: json.loads((OUTPUT / name / "models.json").read_text(encoding="utf-8")) for name in ("experiments", "end_to_end")}
snapshot = {
    "release": str(RELEASE), "tables": tables, "speedup": speedup, "sensitivity_summary": sensitivity,
    "models": models, "source_files": len(current["source_sha256"]),
    "source_sha256": current["source_sha256"], "raw_data_sha256": current["snapshot_sha256"],
    "runtime_script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RELEASE.glob("*.ps1")},
    "ablation_counts": dict(Counter(r["kind"] for r in tables["ablation"])),
    "acceptance": json.loads((OUTPUT / "qa/programming_acceptance.json").read_text(encoding="utf-8")),
    "tests": json.loads((PROJECT / "outputs/qa/source-submission/unit-tests.json").read_text(encoding="utf-8")),
    "manifest_environment": {k: current[k] for k in ("python", "platform")},
}
assert snapshot["tests"]["passed"] and snapshot["tests"]["tests"] == 204
(WORK / "evidence.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
brief = json.loads((PROJECT / "outputs/qa/lab02-materials/brief.json").read_text(encoding="utf-8"))
(WORK / "brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT / "deliverables/lab02").glob("*.pptx")}
(WORK / "protected-pptx.json").write_text(json.dumps(protected, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"source_files": snapshot["source_files"], "tests": 204, "speedup": speedup,
                  "all_saved_artifact_hashes_match": True, "ablation_counts": snapshot["ablation_counts"],
                  "sensitivity": sensitivity,
                  "graph_policies": {k: v["adaptive_model"]["review_policy"]["minimum_support"] for k, v in models["end_to_end"]["models_by_seed"].items()}}, ensure_ascii=False, indent=2))
