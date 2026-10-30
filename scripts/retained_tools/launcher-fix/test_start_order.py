"""Regression: a slow first experiment must not delay the web server or a second launcher."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

qa = Path(__file__).resolve().parent
release = Path(r"C:\Users\Administrator\Desktop\人工智能实验2\第31组_实验二源码")
with tempfile.TemporaryDirectory(prefix="startup-order-", dir=qa) as temporary:
    root = Path(temporary).resolve()
    assert root.is_relative_to(qa)
    (root / "scripts").mkdir()
    (root / "web/dist").mkdir(parents=True)
    (root / "web/dist/index.html").write_text("test frontend", encoding="utf-8")
    shutil.copy2(release / "start_web.ps1", root / "start_web.ps1")
    shutil.copy2(release / "stop_web.ps1", root / "stop_web.ps1")
    (root / "scripts/web_python.ps1").write_text(
        "function Resolve-WebPython { param($Root, $PythonPath); return $PythonPath }", encoding="utf-8-sig")
    (root / "scripts/prepare_lab02.py").write_text(
        "from pathlib import Path\nimport sys\nsys.exit(0 if (Path(__file__).resolve().parents[1] / 'ready.flag').exists() else 1)\n",
        encoding="utf-8")
    (root / "run_lab02.ps1").write_text(
        "param($PythonPath)\nStart-Sleep -Seconds 12\nSet-Content -LiteralPath (Join-Path $PSScriptRoot 'ready.flag') -Value 'done'\n",
        encoding="utf-8-sig")
    (root / "main.py").write_text('''from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import hashlib, json, sys
root = Path(__file__).resolve().parent
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({'app': 'equity-reasoning-web', 'workspace_id': hashlib.sha256(str(root).casefold().encode()).hexdigest()[:16]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args): pass
HTTPServer(('127.0.0.1', int(sys.argv[-1])), Handler).serve_forever()
''', encoding="utf-8")
    with socket.socket() as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               str(root / "start_web.ps1"), "-Port", str(port), "-NoBrowser", "-Entry", "lab02",
               "-PythonPath", sys.executable]
    started = time.perf_counter()
    with (qa / "startup-order.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=qa, stdout=log, stderr=subprocess.STDOUT)
        try:
            while time.perf_counter() - started < 10:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=.5):
                        break
                except OSError:
                    time.sleep(.1)
            else:
                raise AssertionError("Web server waited for the slow experiment")
            web_ready_seconds = time.perf_counter() - started
            assert not (root / "ready.flag").exists()
            second_started = time.perf_counter()
            second = subprocess.run(command, cwd=qa, stdout=log, stderr=subprocess.STDOUT, timeout=5)
            assert second.returncode == 0 and not (root / "ready.flag").exists()
            second_seconds = time.perf_counter() - second_started
            assert process.wait(timeout=30) == 0
            assert (root / "ready.flag").exists()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                            str(root / "stop_web.ps1")], cwd=qa, stdout=log, stderr=subprocess.STDOUT, timeout=15)
    report = {"web_ready_before_slow_initialization": True, "web_ready_seconds": web_ready_seconds,
              "second_launcher_did_not_wait_for_initialization": True, "second_launcher_seconds": second_seconds}
    (qa / "startup-order-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
