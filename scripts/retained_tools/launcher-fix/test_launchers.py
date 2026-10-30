"""Run both .cmd files with Explorer's environment and a foreign working directory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import psutil

root = Path(r"C:\Users\Administrator\Desktop\人工智能实验2\第31组_实验二源码")
qa = Path(__file__).resolve().parent
explorer = next(p for p in psutil.process_iter(["name"])
                if (p.info["name"] or "").lower() == "explorer.exe")
environment = explorer.environ()
system_root = environment.get("SystemRoot", r"C:\Windows")
results = []
first_pid = None
for filename in ("打开网站.cmd", "启动实验二.cmd"):
    started = time.perf_counter()
    process = subprocess.run([str(Path(system_root) / "System32/cmd.exe"), "/d", "/c",
                              str(root / filename), "-NoBrowser"],
                             cwd=Path(system_root) / "System32", env=environment,
                             input="\r\n", capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=90)
    output = process.stdout + process.stderr
    (qa / (filename + ".log")).write_text(output, encoding="utf-8")
    print(filename, process.returncode, output, flush=True)
    assert process.returncode == 0
    state = json.loads((root / "outputs/runtime/server.json").read_text(encoding="utf-8-sig"))
    with urllib.request.urlopen(state["url"] + "/api/health", timeout=5) as response:
        health = json.load(response)
    expected_id = hashlib.sha256(str(root).casefold().encode()).hexdigest()[:16]
    assert health["workspace_id"] == expected_id
    with urllib.request.urlopen(state["url"], timeout=5) as response:
        assert response.status == 200 and b'<div id="root"></div>' in response.read()
    if first_pid is None:
        first_pid = state["pid"]
    else:
        assert state["pid"] == first_pid
    results.append({"launcher": filename, "exit_code": 0, "seconds": time.perf_counter() - started,
                    "url": state["url"] + "/#/labs/lab02", "pid": state["pid"],
                    "explorer_environment": True, "foreign_cwd": True, "http_status": 200})
(qa / "launcher-verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
