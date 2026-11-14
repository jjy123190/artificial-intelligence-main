@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\start_web.ps1" -Entry lab01 %*
if errorlevel 1 pause
