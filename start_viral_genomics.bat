@echo off
echo ==========================================
echo Quantum Intelligent Viral Genomics Suite
echo ==========================================
echo.

REM Clear any existing Python paths from environment
set PYTHONPATH=

REM Set the correct Python paths
set PATH=C:\Users\nahid_vv0xche\AppData\Local\Programs\Python\Python310;C:\Users\nahid_vv0xche\AppData\Local\Programs\Python\Python310\Scripts;%PATH%

echo Checking environment...
where python
python --version

echo.
echo Testing BioPython...
python -c "from Bio import SeqIO; print('✅ BioPython is working!')"

echo.
echo Starting the application...
echo.
streamlit run script.py

pause