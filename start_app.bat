@echo off
echo ==========================================
echo Quantum Intelligent Viral Genomics Suite
echo ==========================================
echo.
echo This will launch the app with the correct Python environment.
echo.

REM Set the correct Python path directly
set PYTHON_PATH=C:\Users\nahid_vv0xche\AppData\Local\Programs\Python\Python310
set PYTHON_EXE=%PYTHON_PATH%\python.exe
set STREAMLIT_EXE=%PYTHON_PATH%\Scripts\streamlit.exe

echo Checking Python installation...
if exist "%PYTHON_EXE%" (
    echo ✅ Found Python: %PYTHON_EXE%
) else (
    echo ❌ Python not found at: %PYTHON_EXE%
    echo.
    echo Please update the PYTHON_PATH in this batch file to point to your Python installation.
    pause
    exit /b 1
)

echo.
echo Checking for BioPython...
"%PYTHON_EXE%" -c "from Bio import SeqIO; print('✅ BioPython is installed')" 2>nul
if errorlevel 1 (
    echo ❌ BioPython is not installed in this Python environment.
    echo.
    echo Installing required packages...
    "%PYTHON_EXE%" -m pip install biopython scipy scikit-learn networkx plotly streamlit pandas numpy matplotlib seaborn
    echo.
    echo Installation complete!
    timeout /t 2 /nobreak
)

echo.
echo Starting the application...
echo.

REM Check if streamlit.exe exists
if exist "%STREAMLIT_EXE%" (
    "%STREAMLIT_EXE%" run script.py
) else (
    "%PYTHON_EXE%" -m streamlit run script.py
)

echo.
echo Application closed.
pause