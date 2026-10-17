@echo off
rem UltraTranslate installer: double-click me. Finds (or installs) Python 3.12, then runs install.py.
cd /d "%~dp0"
title UltraTranslate installer
chcp 65001 >nul
set PYEXE=
py -3.12 -c "import sys" >nul 2>nul && set PYEXE=py -3.12
if not defined PYEXE (
  python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>nul && set PYEXE=python
)
if not defined PYEXE (
  echo Python 3.12 is not installed.
  choice /M "Install Python 3.12 now with winget"
  if errorlevel 2 goto :nopython
  winget install -e --id Python.Python.3.12 --accept-package-agreements
  py -3.12 -c "import sys" >nul 2>nul && set PYEXE=py -3.12
)
if not defined PYEXE goto :nopython
%PYEXE% install.py %*
exit /b

:nopython
echo.
echo Please install Python 3.12 from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
echo then double-click install.cmd again.
pause
