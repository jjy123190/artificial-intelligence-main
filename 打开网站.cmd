@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo Starting the local website. Please keep this window open during first-time setup.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web.ps1" -Entry home %*
set "launchExit=%errorlevel%"
if not "%launchExit%"=="0" (
    echo.
    echo Website startup failed. Please copy the error above.
    pause
)
exit /b %launchExit%
