@echo off
rem Double-click to set up (first run only) and open Flutter Lab in your browser.
rem Options are passed through, e.g.:  start.bat -Test   start.bat -Evaluate   start.bat -Train
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 pause
