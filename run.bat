@echo off
setlocal

rem Resolve script and venv paths
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

rem Default config or first argument
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=%SCRIPT_DIR%configurations\example.yaml"
rem Normalize slashes to backslashes
set "CONFIG=%CONFIG:/=\%"

rem Validate venv
if not exist "%VENV_PY%" (
  echo [ERROR] Virtual environment not found at "%VENV_DIR%".
  echo Re-run install to create it.
  exit /b 1
)
if not exist "%ACTIVATE%" (
  echo [ERROR] Activation script not found at "%ACTIVATE%".
  exit /b 1
)

rem Activate venv
call "%ACTIVATE%"
if errorlevel 1 (
  echo [ERROR] Failed to activate the virtual environment.
  exit /b 1
)


rem Prefer calling the installed console script
where vxpy >NUL 2>&1
if errorlevel 1 (
  rem Fallback to module invocation if console script is missing
  "%VENV_PY%" -m vxpy -c "%CONFIG%" run
  set "EXITCODE=%ERRORLEVEL%"
) else (
  vxpy -c "%CONFIG%" run
  set "EXITCODE=%ERRORLEVEL%"
)


echo VxPy closed with exit code %EXITCODE%.`
timeout /t 1 /nobreak >nul

endlocal & exit /b %EXITCODE%
