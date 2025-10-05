@echo off
setlocal

rem Find Python 3 via py launcher, then fallback to PATH
call :FindPython38Plus
if not defined PYTHON_EXE (
  echo [ERROR] Python 3 not found. Install it from https://www.python.org/downloads/windows/
  exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo Using Python: "%PYTHON_EXE%"

rem Create venv if it does not exist
if not exist "%VENV_PY%" (
  echo Creating virtual environment at "%VENV_DIR%"...
  "%PYTHON_EXE%" -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
) else (
  echo Virtual environment already exists at "%VENV_DIR%".
)

rem Upgrade pip
echo Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip.
  exit /b 1
)

echo Installing package "vxpy"...
"%VENV_PY%" -m pip install "vxpy"
if errorlevel 1 (
  echo [ERROR] Failed to install vxpy.
  exit /b 1
)

echo Done.
exit /b 0



:FindPython38Plus
  set "PYTHON_EXE="
  rem Prefer highest available 3.x >= 3.8 via Python Launcher
  for %%V in (3.13 3.12 3.11 3.10 3.9 3.8) do (
    for /f "delims=" %%I in ('py -%%V -c "import sys; print(sys.executable)" 2^>NUL') do (
      if exist "%%I" (
        set "PYTHON_EXE=%%I"
        goto :_fp_done
      )
    )
  )

  rem Fallback: scan PATH and require >= 3.8
  for /f "delims=" %%I in ('where python 2^>NUL') do (
    "%%I" -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,8) else 1)" >NUL 2>NUL
    if not errorlevel 1 (
      set "PYTHON_EXE=%%I"
      goto :eof
    )
  )

:_fp_done
  if defined PYTHON_EXE (
    for %%P in ("%PYTHON_EXE%") do set "PYTHON_EXE=%%~fP"
  )
  goto :eof
