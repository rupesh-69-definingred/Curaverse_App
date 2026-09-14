@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv || goto :error
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt || goto :error
echo.
echo Starting Curaverse at http://127.0.0.1:5000
python app.py
goto :eof
:error
echo.
echo Setup failed. Make sure Python is installed and available as ^"python^" in PATH.
pause
exit /b 1
