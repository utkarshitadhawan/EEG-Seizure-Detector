@echo off
setlocal
cd /d "%~dp0"
title EEG Seizure Detector
set "ENVDIR=%USERPROFILE%\envs\seizure"
set "VPY=%ENVDIR%\Scripts\python.exe"
set "SKV=1.6.1"
if not exist "%~dp0gui.py" goto NOFILES
if not exist "%~dp0eegseizure\__init__.py" goto NOFILES
if not exist "%~dp0models" goto NOFILES
if exist "%VPY%" "%VPY%" -c "import numpy, pandas, scipy, joblib, matplotlib, PIL, mne, tkinter, sklearn; assert sklearn.__version__=='%SKV%'" >nul 2>&1 && goto RUN
set "PY="
python --version >nul 2>&1 && set "PY=python"
if not defined PY py --version >nul 2>&1 && set "PY=py"
if not defined PY goto NOPY
echo.
echo First-time setup (about 3-6 minutes, needs internet). Please wait...
echo.
%PY% -m venv "%ENVDIR%" || goto FAIL
"%VPY%" -m ensurepip --upgrade >nul 2>&1
"%VPY%" -m pip install --upgrade pip >nul 2>&1
"%VPY%" -m pip install "scikit-learn==%SKV%" numpy pandas scipy joblib matplotlib pillow mne || goto FAIL
"%VPY%" -c "import tkinter" >nul 2>&1 || goto NOTK
:RUN
start "" "%ENVDIR%\Scripts\pythonw.exe" "%~dp0gui.py"
exit /b 0
:NOFILES
echo Keep START_HERE.bat, gui.py, the eegseizure folder and the models folder together in ONE folder.
pause
exit /b 1
:NOPY
echo Python is not installed. Trying to install it automatically...
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements || start "" https://www.python.org/downloads/
echo When Python is installed, CLOSE this window and double-click START_HERE.bat again.
echo (If installing by hand from python.org, tick "Add python.exe to PATH".)
pause
exit /b 1
:FAIL
echo SETUP FAILED. Check your internet connection (turn off VPN / try another network), then run this file again.
echo If scikit-learn==%SKV% cannot be found, install Python 3.11 or 3.12.
pause
exit /b 1
:NOTK
echo Python was installed without Tkinter. Re-run the Python installer, choose Modify, tick "tcl/tk and IDLE".
pause
exit /b 1
