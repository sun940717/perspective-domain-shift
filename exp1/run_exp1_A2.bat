@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM ======================================================================
REM  PDS A2 - repeat Exp 1 (level protocol), A1 and A1b with a second backbone
REM  Backbone: Swin-T (vision transformer). Results go to results_swin_t\
REM  Reuses ..\exp0\.venv and the crops in data\
REM  Time: about 45-75 min on an RTX 4050
REM ======================================================================

set "PY=..\exp0\.venv\Scripts\python.exe"
set "BB=swin_t"
set "OUT=results_swin_t"
REM Transformers are usually fine-tuned with a lower learning rate than CNNs.
set "LR=1e-4"

if not exist "%PY%" (
  echo [ERROR] Python environment not found. Run ..\exp0\run_exp0.bat first.
  echo         找不到 Python 環境，請先執行 ..\exp0\run_exp0.bat。
  pause
  exit /b 1
)
if not exist "data\index.csv" (
  echo [ERROR] data\index.csv not found. Run run_exp1.bat first.
  echo         找不到 data\index.csv，請先執行 run_exp1.bat。
  pause
  exit /b 1
)

echo [1/4] Train + evaluate %BB%, level protocol, 3 frames x 5 folds / 訓練與評估...
"%PY%" pds_exp1.py run --data data --out %OUT% --backbone %BB% --lr %LR% --protocols level --frames cam alpha allo --workers 2
if errorlevel 1 goto :fail

echo [2/4] A1 de-rotation (reuses saved models) / A1 轉正（重用已存模型）...
"%PY%" pds_exp1.py derot --data data --out %OUT% --backbone %BB% --lr %LR% --workers 2
if errorlevel 1 goto :fail

echo [3/4] A1b noisy-roll de-rotation / A1b 含誤差側傾轉正...
"%PY%" pds_exp1.py derotnoise --data data --out %OUT% --backbone %BB% --lr %LR% --workers 2
if errorlevel 1 goto :fail

echo [4/4] Report / 報告...
"%PY%" pds_exp1.py report --results %OUT%
if errorlevel 1 goto :fail

echo.
echo Done. See %OUT%\summary.md and %OUT%\tilt_degradation.png
echo 完成。請看 %OUT%\summary.md 與 %OUT%\tilt_degradation.png
explorer %OUT%
pause
exit /b 0

:fail
echo.
echo [ERROR] A step failed. Copy the error text above and send it to Claude.
echo         有步驟失敗，請把上方錯誤訊息複製給 Claude。
pause
exit /b 1
