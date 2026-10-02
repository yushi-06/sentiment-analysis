@echo off
echo Activating Sentiment Analysis Virtual Environment...
echo ================================================

REM Check if virtual environment exists
if not exist "sentiment_env\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv sentiment_env
    echo Virtual environment created!
)

REM Activate the virtual environment
call sentiment_env\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo Python location: %VIRTUAL_ENV%
echo.
echo Available commands:
echo   python fixed_main.py          - Enhanced bias-fixed system
echo   python start_system.py        - System launcher  
echo   python intelligent_main.py    - Full intelligent interface
echo   python simple_verify.py       - Verify system setup
echo.
echo To deactivate: deactivate
echo ================================================
