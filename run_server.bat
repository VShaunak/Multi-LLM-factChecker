@echo off
setlocal
cd /d "%~dp0"
title Multi-LLM FactChecker & 3D Consensus Evaluator - Port 8020
echo ================================================================
echo  Multi-LLM FactChecker & 3D Consensus Evaluator
echo  Server running at: http://127.0.0.1:8020
echo  Press Ctrl+C to stop the server
echo ================================================================
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 app.py --port 8020
) else (
  python app.py --port 8020
)
pause
