@echo off
setlocal
chcp 65001 >nul
call "%~dp0源码\停止网站.cmd" %*
exit /b %errorlevel%
