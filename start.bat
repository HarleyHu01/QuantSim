@echo off
rem ============================================================
rem  QuantSim one-click launcher
rem  Backend API (FastAPI :8000) + Web UI (Streamlit, port 8501-8505)
rem  First run creates a virtual environment and installs deps.
rem  ASCII-only + CRLF, no dependency on activate.bat.
rem ============================================================
cd /d %~dp0
title QuantSim Launcher

rem ---- 1. check python ----
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found. Please install Python 3.11+ and add it to PATH.
    pause
    exit /b 1
)

rem ---- 2. venv and dependencies (use venv python directly, no activate needed) ----
if not exist .venv\Scripts\python.exe (
    echo [1/3] First run: creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Please check your Python installation.
        pause
        exit /b 1
    )
    echo [2/3] Installing dependencies, this may take a few minutes...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install core dependencies. Check your network and retry.
        pause
        exit /b 1
    )
    .venv\Scripts\python.exe -m pip install -r requirements-optional.txt 2>nul
)

rem ---- 3. kill stale QuantSim backend on :8000 before starting (avoid old process
rem        holding the port so the NEW code never comes up and UI talks to stale API) ----
echo [3/3] Checking port 8000 for stale QuantSim API processes...
rem 3a. precise kill: match command line (uvicorn / quant.api) so we do not kill other apps
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue; if ($c) { foreach ($x in $c) { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $x.OwningProcess) -ErrorAction SilentlyContinue; if ($p -and $p.CommandLine -match 'uvicorn|quant\.api') { Write-Host ('Killing stale QuantSim API (PID ' + $p.ProcessId + ')'); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } else { Write-Host ('WARN: port 8000 held by PID ' + $x.OwningProcess + ' (' + $p.ProcessName + ') - not killed, close it manually') } } } else { Write-Host 'Port 8000 is free.' }"
rem 3b. fallback: if :8000 is still listening (e.g. PowerShell cannot see it),
rem      force-kill whatever holds it (this machine's :8000 belongs to QuantSim)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    echo   Forcing kill of PID %%p on :8000
    taskkill /F /PID %%p >nul 2>nul
)
timeout /t 1 /nobreak >nul 2>&1

rem ---- 4. start backend API in background, log to api.log ----
echo [4/4] Starting backend API: http://127.0.0.1:8000  (log: api.log)
start "QuantSim-API" /min cmd /c ".venv\Scripts\python.exe -m uvicorn quant.api.app:app --host 127.0.0.1 --port 8000 > api.log 2>&1"

rem ---- 5. wait for backend health (max 60s) ----
set /a tries=0
:wait_api
timeout /t 1 /nobreak >nul 2>&1
set /a tries+=1
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri http://127.0.0.1:8000/api/health -TimeoutSec 2 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 goto api_ok
set /a mod=%tries% %% 5
if %mod% equ 0 echo   waiting for backend (%tries%s)...
if %tries% lss 60 goto wait_api
echo [ERROR] Backend API did not become ready within 60s. See api.log.
pause
exit /b 1

:api_ok
echo Backend API is ready.
rem ---- 5b. verify the running backend actually has backtest routes (openapi check).
rem        If an OLD process was still holding :8000, /api/backtest/combo would 404.
rem        NOTE: on failure we WARN but DO NOT exit - the UI must still start. ----
powershell -NoProfile -Command "try { $o = Invoke-RestMethod -Uri http://127.0.0.1:8000/openapi.json -TimeoutSec 30; if ($o.paths.'/api/backtest/combo') { Write-Host 'Backend routes OK - backtest available.'; exit 0 } else { Write-Host 'WARN: backend missing backtest routes - stale process?'; exit 1 } } catch { Write-Host ('WARN: openapi check failed - ' + $_.Exception.Message); exit 1 }"
if errorlevel 1 goto bt_warn
echo Backend routes OK.
goto ui_start
:bt_warn
echo [WARN] Backend route check did not confirm backtest routes.
echo        The process on :8000 may be stale - backtest could return 404.
echo        Verify at http://127.0.0.1:8000/docs ; the UI will still start.
:ui_start

rem ---- 6. find a free port for the Web UI and start Streamlit ----
set PORT=8501
:find_port
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>nul
if errorlevel 1 goto port_busy
goto port_free
:port_busy
set /a PORT+=1
if %PORT% gtr 8505 (
    echo [ERROR] No free port in range 8501-8505. Close some apps and retry.
    pause
    exit /b 1
)
goto find_port
:port_free
echo Web UI will run on port %PORT%.
start "" http://127.0.0.1:%PORT%
echo Starting Streamlit on http://127.0.0.1:%PORT% ... (close this window to stop it)
.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port %PORT% --server.headless true
if errorlevel 1 (
    echo [ERROR] Streamlit failed to start. See the error above.
)

echo.
echo Stopped. You may close the QuantSim-API window manually; backend log is in api.log
pause
