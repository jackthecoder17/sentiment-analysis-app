@echo off
echo Installing requirements (FastAPI, Uvicorn, etc.)...
python -m pip install -r requirements.txt
echo.
echo Starting Custom Sentiment Analysis API Server...
echo.
echo If this is the first time, it might take a moment to load libraries.
echo The app will be available at http://127.0.0.1:8000
echo.
python -m uvicorn src.main:app --reload
pause
