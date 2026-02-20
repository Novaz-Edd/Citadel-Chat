@echo off
REM ─────────────────────────────────────────────
REM Citadel-Chat Windows Startup Script
REM ─────────────────────────────────────────────
echo.
echo  *** Citadel-Chat: Starting up... ***
echo.

REM Check if .env file has been configured
findstr /C:"your_groq_api_key_here" .env >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo  [WARNING] GROQ_API_KEY is not set in .env file!
    echo  Please open .env and replace 'your_groq_api_key_here' with your real key.
    echo  Get a free key at: https://console.groq.com
    echo.
    pause
    exit /b 1
)

echo  Starting Citadel-Chat on http://localhost:8000 ...
echo  Press Ctrl+C to stop the server.
echo.

python main.py

pause
