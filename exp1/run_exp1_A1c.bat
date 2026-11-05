@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM ======================================================================
REM  PDS A1c - corner-fill check for the A1 de-rotation
REM  Re-runs the oracle de-rotation with mirrored corners instead of black ones,
REM  for all three model sets. No training: reuses saved checkpoints.
REM    results\                  ResNet-18, lr 3e-4
REM    results_swin_t\           Swin-T,    lr 1e-4
REM    results_resnet18_lr1e-4\  ResNet-18, lr 1e-4
REM  Time: about 25-40 min on an RTX 4050
REM ======================================================================

set "PY=..\exp0\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python environment not found. Run ..\exp0\run_exp0.bat first.
  echo         找不到 Python 環境，請先執行 ..\exp0\run_exp0.bat。
  pause
  exit /b 1
)

echo [0/3] Geometry self-test / 幾何自我測試...
"%PY%" pds_exp1.py selftest
if errorlevel 1 goto :fail

echo [1/3] ResNet-18 lr 3e-4, reflect fill / 鏡射填補...
"%PY%" pds_exp1.py derot --data data --out results --backbone resnet18 --lr 3e-4 --fill reflect --workers 2
if errorlevel 1 goto :fail
"%PY%" pds_exp1.py report --results results > nul
if errorlevel 1 goto :fail

echo [2/3] Swin-T lr 1e-4, reflect fill / 鏡射填補...
"%PY%" pds_exp1.py derot --data data --out results_swin_t --backbone swin_t --lr 1e-4 --fill reflect --workers 2
if errorlevel 1 goto :fail
"%PY%" pds_exp1.py report --results results_swin_t > nul
if errorlevel 1 goto :fail

echo [3/3] ResNet-18 lr 1e-4, reflect fill / 鏡射填補...
"%PY%" pds_exp1.py derot --data data --out results_resnet18_lr1e-4 --backbone resnet18 --lr 1e-4 --fill reflect --workers 2
if errorlevel 1 goto :fail
"%PY%" pds_exp1.py report --results results_resnet18_lr1e-4 > nul
if errorlevel 1 goto :fail

echo.
echo ===== Section 5b from each report / 各報告的 5b 節 =====
for %%D in (results results_swin_t results_resnet18_lr1e-4) do (
  echo.
  echo ----- %%D -----
  "%PY%" -c "import sys; t=open(sys.argv[1], encoding='utf-8').read(); print(t[t.find('## 5b'):t.find('## 6')])" "%%D\summary.md"
)
echo.
echo Done. Send everything above to Claude.
echo 完成。把上面全部內容傳給 Claude。
pause
exit /b 0

:fail
echo.
echo [ERROR] A step failed. Copy the error text above and send it to Claude.
echo         有步驟失敗，請把上方錯誤訊息複製給 Claude。
pause
exit /b 1
