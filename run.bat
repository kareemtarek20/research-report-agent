@echo off
REM Starts the Research Intelligence Agent web UI and opens it in your browser.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment and installing dependencies, first run only...
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

if not exist ".env" (
    echo No .env found. Open .env and set LLM_API_KEY before running.
    pause
    exit /b 1
)

start "" http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true
pause

:error
echo Something went wrong. Check the messages above.
pause
