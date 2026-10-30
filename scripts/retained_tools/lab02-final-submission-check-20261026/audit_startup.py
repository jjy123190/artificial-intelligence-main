"""Run the actual submission launcher from a clean copy and a foreign cwd."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error

WORK = Path(__file__).resolve().parent
ROOT = WORK / "isolated-source"
PYTHON = WORK.parent / "source-submission/staging-source/.venv/Scripts/python.exe"
URL = "http://127.0.0.1:8793"
EXPECTED = hashlib.sha256(str(ROOT).casefold().encode()).hexdigest()[:16]


def main():
    started = time.perf_counter()
    log_path = WORK / "first-startup.log"
    command = ["cmd.exe", "/d", "/c", str(ROOT / "打开网站.cmd"),
               "-NoBrowser", "-Port", "8793", "-PythonPath", str(PYTHON)]
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=WORK, stdout=output, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW,
                                   env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    (WORK / "startup-process.json").write_text(json.dumps({"pid": process.pid, "command": command,
                                                        "source": str(ROOT), "url": URL}, ensure_ascii=False, indent=2), encoding="utf-8")
    ready_at = None
    last_stage = None
    while process.poll() is None:
        if ready_at is None:
            try:
                with urllib.request.urlopen(URL + "/api/health", timeout=1) as response:
                    health = json.load(response)
                assert health["workspace_id"] == EXPECTED, health
                ready_at = time.perf_counter() - started
                print(f"Website ready after {ready_at:.1f}s, before full experiment completion.", flush=True)
            except urllib.error.URLError:
                pass
        current = log_path.read_text(encoding="utf-8", errors="replace")
        stages = re.findall(r"\[[1-7]/7\].*", current)
        if stages and stages[-1] != last_stage:
            last_stage = stages[-1]
            print(last_stage, flush=True)
        time.sleep(2)
    report = {"command": command, "exit_code": process.returncode,
              "elapsed_seconds": time.perf_counter() - started,
              "website_ready_after_seconds": ready_at,
              "launched_from_other_directory": True,
              "source": str(ROOT), "url": URL}
    if process.returncode == 0:
        report["verification"] = json.loads((ROOT / "outputs/lab02/qa/programming_acceptance.json").read_text(encoding="utf-8"))
        report["figure_count"] = len(list((ROOT / "outputs/lab02").rglob("*.png")))
        (WORK / "first-startup-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        assert report["verification"]["successful"] and report["verification"]["full_replay"]
        assert report["figure_count"] == 13, report["figure_count"]
        assert ready_at is not None
    else:
        print(log_path.read_text(encoding="utf-8", errors="replace")[-7000:], flush=True)
    (WORK / "first-startup-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
