@echo off
echo ================================================================================
echo Credit Card Fraud Detection - Quick Start
echo ================================================================================
echo.

echo [1/4] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python found
echo.

echo [2/4] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

echo [3/4] Creating output directory...
if not exist "output" mkdir output
echo [OK] Output directory ready
echo.

echo [4/4] Running fraud detection pipeline...
echo.
python main.py

if errorlevel 1 (
    echo.
    echo ERROR: Pipeline failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [OK] Pipeline completed successfully!
echo Check the 'output' folder for results and visualizations.
echo ================================================================================
pause
