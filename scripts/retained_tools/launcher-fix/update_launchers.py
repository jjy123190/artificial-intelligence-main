from pathlib import Path
import shutil

release = Path(r"C:\Users\Administrator\Desktop\人工智能实验2\第31组_实验二源码")
development = Path(__file__).resolve().parents[3]
batch = r'''@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo Starting the local website. Please keep this window open during first-time setup.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web.ps1" -Entry lab02 %*
set "launchExit=%errorlevel%"
if not "%launchExit%"=="0" (
    echo.
    echo Website startup failed. Please copy the error above.
    pause
)
exit /b %launchExit%
'''
for name in ("打开网站.cmd", "启动实验二.cmd"):
    (release / name).write_bytes(batch.replace("\n", "\r\n").encode("utf-8"))
    if name == "启动实验二.cmd":
        shutil.copy2(release / name, development / name)
# 保持开发项目的总入口，同时让以后导出的实验二入口也使用修复后的脚本。
(development / "打开网站.cmd").write_bytes(batch.replace("-Entry lab02", "-Entry home").replace("\n", "\r\n").encode("utf-8"))
text = (release / "start_web.ps1").read_text(encoding="utf-8-sig")
(release / "start_web.ps1").write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
shutil.copy2(release / "start_web.ps1", development / "start_web.ps1")
print("Updated both release launchers and the development copies with Windows line endings.")
