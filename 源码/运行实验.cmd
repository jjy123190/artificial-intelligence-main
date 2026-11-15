@echo off
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
