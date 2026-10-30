"""Verify the double-click menu without recomputing the user's completed experiments."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import psutil

root = Path(r"C:\Users\Administrator\Desktop\人工智能实验2\第31组_实验二源码")
qa = Path(__file__).resolve().parent
explorer = next(p for p in psutil.process_iter(["name"])
                if (p.info["name"] or "").lower() == "explorer.exe")
environment = explorer.environ()
protected = [root / "outputs/lab02/experiments/models.json", root / "outputs/lab02/end_to_end/models.json"]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}


def run(directory, responses, *arguments):
    result = subprocess.run(["cmd.exe", "/d", "/c", str(directory / "运行实验.cmd"), *arguments],
                            env=environment, cwd=qa, input=responses, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=35)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


exit_output = run(root, "0\n")
assert "实验二：实验运行与结果查看" in exit_output
assert "重新运行完整实验" in exit_output
view_output = run(root, "2\n\n")
assert "已打开结果文件夹" in view_output
cached_output = run(root, "", "-PythonPath", sys.executable)
assert "已有实验结果" in cached_output and "请输入选项" not in cached_output
assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}

# 确认菜单1会传递--force，避免在真实结果上再耗时重跑；替身只记录参数。
with tempfile.TemporaryDirectory(prefix="menu-fixture-", dir=qa) as temporary:
    fixture = Path(temporary).resolve()
    assert fixture.is_relative_to(qa)
    (fixture / "scripts").mkdir()
    (fixture / "outputs/lab02").mkdir(parents=True)
    for name in ("运行实验.cmd", "run_lab02.ps1"):
        shutil.copy2(root / name, fixture / name)
    helper = "function Resolve-WebPython { param($Root, $PythonPath); return '" + sys.executable + "' }\n"
    helper += "function Start-Process { param($FilePath) }\n"
    (fixture / "scripts/web_python.ps1").write_text(helper, encoding="utf-8-sig")
    (fixture / "scripts/prepare_lab02.py").write_text('''import json, sys
from pathlib import Path
if '--check' not in sys.argv:
    Path(__file__).resolve().parents[1].joinpath('arguments.json').write_text(json.dumps(sys.argv[1:]))
''', encoding="utf-8")
    run_output = run(fixture, "1\n\n")
    assert "完整实验已完成" in run_output
    assert "--force" in json.loads((fixture / "arguments.json").read_text())

report = {"exit_menu": "passed", "view_existing_results": "passed", "force_option_forwarded": "passed",
          "noninteractive_startup_unchanged": "passed", "existing_models_unchanged": True}
(qa / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
(qa / "menu-output.log").write_text(exit_output + "\n" + view_output + "\n" + cached_output + "\n" + run_output, encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
