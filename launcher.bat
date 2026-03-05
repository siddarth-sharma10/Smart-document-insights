@echo off
cd /d "F:\d files\my projects\smart document insights"
call venv\Scripts\activate.bat
start "" "index.html"
start cmd /k "cd /d "F:\d files\my projects\smart document insights" && call venv\Scripts\activate.bat && python app.py"