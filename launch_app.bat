@echo off
cd /d "C:\Users\shyam\OneDrive\Desktop\Sarcasm Detection"
call venv\Scripts\activate.bat
python -m streamlit run app.py
pause
