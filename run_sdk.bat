@echo off
rem OneShot MV through the Comfy SDK, on this PC's ComfyUI. Start ComfyUI first.
rem   run_sdk.bat                      asks for the song, the World line and the length
rem   run_sdk.bat --song x.wav --theme "..." --seconds 30 [--preview] [--fx intense] ...
rem The first run installs comfy-sdk and comfy-api-proxy into .sdk\ next to this file;
rem ComfyUI's own Python packages are left alone.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%~dp0..\..\..\python_embeded\python.exe"
if not exist "%PY%" set "PY=python"

if exist ".sdk\comfy_sdk" goto :installed
echo Installing comfy-sdk and comfy-api-proxy into .sdk ...
"%PY%" -m pip install --disable-pip-version-check --target .sdk comfy-sdk comfy-api-proxy
if errorlevel 1 goto :failed
:installed

if not "%~1"=="" goto :withargs
set "SONG="
set /p "SONG=Song file (drag it into this window, then Enter): "
set "SONG=%SONG:"=%"
if "%SONG%"=="" goto :failed
set "THEME="
set /p "THEME=World line (Enter for none): "
set "SECS=30"
set /p "SECS=Seconds (Enter for 30, 0 = whole song): "
"%PY%" sdk_run.py --song "%SONG%" --theme "%THEME%" --seconds %SECS%
goto :end

:withargs
"%PY%" sdk_run.py %*
goto :end

:failed
echo.
echo Something went wrong (see above).
:end
echo.
pause
