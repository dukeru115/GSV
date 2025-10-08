@echo off
REM Development Environment Setup Script for GSV 2.0 (Windows)
REM This script sets up a complete development environment on Windows

echo ==================================================
echo GSV 2.0 - Development Environment Setup (Windows)
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

REM Display Python version
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Python %PYTHON_VERSION% detected
echo.

REM Create virtual environment
set /p VENV="Create virtual environment? (y/n): "
if /i "%VENV%"=="y" (
    echo Creating virtual environment...
    python -m venv gsv_env
    call gsv_env\Scripts\activate.bat
    echo Virtual environment created and activated
    echo.
)

REM Install core dependencies
echo Installing core dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Core dependencies installed
echo.

REM Install development tools
set /p DEVTOOLS="Install development tools? (pytest, black, flake8, mypy) (y/n): "
if /i "%DEVTOOLS%"=="y" (
    echo Installing development tools...
    pip install pytest pytest-cov black flake8 mypy
    echo Development tools installed
    echo.
)

REM Run installation check
echo Verifying installation...
python check_installation.py

if errorlevel 1 (
    echo.
    echo ERROR: Setup verification failed
    echo Please check error messages above
    pause
    exit /b 1
)

echo.
echo ==================================================
echo Setup complete!
echo ==================================================
echo.
echo Next steps:
echo   1. Run tests: python gsv2_tests.py
echo   2. Try examples: python examples\simple_example.py
echo   3. Read CONTRIBUTING.md for guidelines
echo   4. Start coding!
echo.
if /i "%VENV%"=="y" (
    echo To activate virtual environment in future:
    echo   gsv_env\Scripts\activate.bat
    echo.
)
pause
