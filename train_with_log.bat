@echo off
setlocal EnableDelayedExpansion

set PYTHONIOENCODING=utf-8

set "TS=%date%_%time%"
set "TS=%TS: =0%"
set "TS=%TS:/=%"
set "TS=%TS:\=%"
set "TS=%TS::=%"
set "TS=%TS:.=%"
set "LOGDIR=runs\detect"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOGFILE=%LOGDIR%\train_%TS%.log"

echo Logging to %LOGFILE%
echo.

set "ROOT=%~dp0"
set "PROJECT=%ROOT%runs\\detect"
set "DEFAULT_ARGS=model=models/yolov8_flower.yaml data=data/flower.yaml imgsz=640 epochs=200 batch=8 device=0 workers=0 project=%PROJECT% name=train exist_ok=False"
yolo train %DEFAULT_ARGS% %* > "%LOGFILE%" 2>&1
echo.
echo Done. Log saved to %LOGFILE%
