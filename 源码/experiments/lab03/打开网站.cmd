@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\start_web.ps1" -Entry lab03 %*
if errorlevel 1 pause
