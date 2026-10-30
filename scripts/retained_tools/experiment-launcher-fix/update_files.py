from pathlib import Path
import shutil

development = Path(__file__).resolve().parents[3]
release = Path(r"C:\Users\Administrator\Desktop\人工智能实验2\第31组_实验二源码")
batch = r'''@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_lab02.ps1" -Interactive
) else (
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_lab02.ps1" %*
)
set "experimentExit=%errorlevel%"
if not "%experimentExit%"=="0" (
    echo.
    echo Experiment execution failed. Please copy the error above.
    pause
)
exit /b %experimentExit%
'''
(release / "运行实验.cmd").write_bytes(batch.replace("\n", "\r\n").encode("utf-8"))
text = (release / "run_lab02.ps1").read_text(encoding="utf-8-sig")
(release / "run_lab02.ps1").write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
for name in ("运行实验.cmd", "run_lab02.ps1"):
    shutil.copy2(release / name, development / name)
print("Updated the interactive experiment launcher and synchronized development sources.")
