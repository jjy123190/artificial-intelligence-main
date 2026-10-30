"""Exercise the shipped Windows runner; keep validation logs outside its source tree."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(sys.argv[1]).resolve()
    qa = Path(__file__).resolve().parent
    started = time.perf_counter()
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               str(root / "run_lab02.ps1"), "-PythonPath", str(root / ".venv/Scripts/python.exe")]
    with (qa / "formal-run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=qa, stdout=log, stderr=subprocess.STDOUT)
    report = {"exit_code": result.returncode, "elapsed_seconds": time.perf_counter() - started,
              "cwd": str(qa), "source_root": str(root)}
    if result.returncode == 0:
        report["verification"] = json.loads((root / "outputs/lab02/qa/programming_acceptance.json").read_text(encoding="utf-8"))
        report["figures"] = [str(p.relative_to(root)) for p in (root / "outputs/lab02").rglob("*.png")]
        assert len(report["figures"]) == 13
    (qa / "formal-run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if result.returncode:
        print((qa / "formal-run.log").read_text(encoding="utf-8", errors="replace")[-7000:])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
