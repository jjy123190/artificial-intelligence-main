"""Validate the relocated source without writing runtime files into the release."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

qa = Path(__file__).resolve().parent
release = Path(sys.argv[1]).resolve()
expected = json.loads((qa / "ready-inventory.json").read_text(encoding="utf-8"))["source_sha256"]


def inventory():
    return {p.relative_to(release).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(release.rglob("*")) if p.is_file()}


assert inventory() == expected
commands = [
    ["lab02", "demo", "--output-dir", str(qa / "relocated-demo")],
    ["lab02", "analyze", "--input", str(release / "data/example_graph.json"),
     "--output-dir", str(qa / "relocated-analysis")],
]
for number, arguments in enumerate(commands, 1):
    with (qa / f"relocated-command-{number}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8",
                                 str(release / "main.py"), *arguments], cwd=qa,
                                stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (number, result.returncode)
demo = json.loads((qa / "relocated-demo/demo/summary.json").read_text(encoding="utf-8"))
assert str(release / "data/raw") in demo["source"]
assert demo["forward"]["converged"]
assert inventory() == expected
report = {
    "release": str(release), "files": len(expected), "all_hashes_match_tested_source": True,
    "isolated_python_mode": True, "foreign_working_directory": str(qa),
    "standalone_demo_and_graph_analysis": "passed", "runtime_files_in_release": 0,
    "tests": json.loads((qa / "unit-tests.json").read_text(encoding="utf-8"))["tests"],
    "mathematical_reference_checks": len(json.loads((qa / "independent-checks.json").read_text(encoding="utf-8"))),
    "formal_verification": json.loads((qa / "formal-run.json").read_text(encoding="utf-8"))["verification"],
    "browser_checks": json.loads((qa / "browser-checks.json").read_text(encoding="utf-8"))["checks"],
    "source_sha256": expected,
}
(qa / "release-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))
