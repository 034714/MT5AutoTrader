@echo off
setlocal
cd /d "%~dp0"
rem Python priority: project .venv -> bundled runtime -> fallback_python.txt (local override) -> PATH python
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if defined PY goto have_py
if exist "runtime\python.exe" set "PY=%~dp0runtime\python.exe"
if defined PY goto have_py
if exist "fallback_python.txt" set /p FALLBACK_PY=<fallback_python.txt
if defined FALLBACK_PY if exist "%FALLBACK_PY%" set "PY=%FALLBACK_PY%"
:have_py
if not defined PY set "PY=python"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8900" ^| findstr "LISTENING"') do set "PORTPID=%%P"
if defined PORTPID (
  start "" "http://127.0.0.1:8900"
  exit /b 0
)
start "MT5AutoTrader" /min "%PY%" "%~dp0src\app.py"
for /l %%N in (1,1,30) do (
  timeout /t 1 /nobreak >nul
  netstat -ano | findstr ":8900" | findstr "LISTENING" >nul
  if not errorlevel 1 (
    start "" "http://127.0.0.1:8900"
    exit /b 0
  )
)
echo Dashboard did not start. Check logs\dashboard.log.
pause
exit /b 1
