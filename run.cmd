@echo off
rem UltraTranslate launcher (pythonw from PATH)
cd /d "%~dp0"
if "%~1"=="" (
  start "" pythonw -m ultra_translate
) else (
  start "" pythonw -m ultra_translate --file "%~f1"
)
