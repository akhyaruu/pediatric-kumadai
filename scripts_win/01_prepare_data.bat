@echo off
REM DeepFill -> crop -> data dictionary -> train/test split -> imputation (diagnosis).
REM Activate the conda environment first
REM Output: data_app\ inside the project folder. Adjust RAW and XLSX if the data live elsewhere.
REM Extra options are passed on to prepare_data.py, e.g.:
REM   scripts_win\01_prepare_data.bat --steps split impute
REM   scripts_win\01_prepare_data.bat --fresh          (redo DeepFill from scratch)
cd /d "%~dp0.."
set RAW=%USERPROFILE%\Downloads\claude context\US_Pictures
set XLSX=%USERPROFILE%\Downloads\claude context\pediatric.xlsx
python prepare_data.py --raw "%RAW%" --xlsx "%XLSX%" --out data_app %*
pause
