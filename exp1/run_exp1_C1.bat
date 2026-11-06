@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM ======================================================================
REM  PDS C1 - de-rotation with a roll ESTIMATED from the image (court trial 3)
REM  Roll is estimated by GeoCalib (ECCV 2024), an off-the-shelf model that was
REM  never trained on our data. No retraining of our models: reuses checkpoints.
REM  Run this AFTER run_exp1_A3.bat has finished (both use the GPU).
REM  First run installs GeoCalib and downloads its weights (internet needed).
REM  Time: mini about 30-60 min; trainval part may take several hours.
REM  Resumable: if it stops, double-click again.
REM ======================================================================

set "PY=..\exp0\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python environment not found. Run ..\exp0\run_exp0.bat first.
  echo         找不到 Python 環境，請先執行 ..\exp0\run_exp0.bat。
  pause
  exit /b 1
)

"%PY%" -c "import geocalib" >nul 2>&1
if errorlevel 1 (
  echo [0/5] Installing GeoCalib / 安裝 GeoCalib...
  "%PY%" -m pip install kornia "git+https://github.com/cvg/GeoCalib"
  if errorlevel 1 goto :fail
)
"%PY%" pds_exp1.py selftest
if errorlevel 1 goto :fail

echo [1/5] Mini, ResNet-18 lr 3e-4 (also estimates roll for every mini image) / 估計側傾...
"%PY%" pds_exp1.py derotest --data data --out results --backbone resnet18 --lr 3e-4 --workers 2
if errorlevel 1 goto :fail
"%PY%" pds_exp1.py report --results results > nul
if errorlevel 1 goto :fail

echo [2/5] Mini, Swin-T lr 1e-4 / 迷你資料，Swin-T...
"%PY%" pds_exp1.py derotest --data data --out results_swin_t --backbone swin_t --lr 1e-4 --workers 2
if errorlevel 1 goto :fail
"%PY%" pds_exp1.py report --results results_swin_t > nul
if errorlevel 1 goto :fail

echo [3/5] Mini, ResNet-18 lr 1e-4 / 迷你資料，ResNet-18 lr 1e-4...
"%PY%" pds_exp1.py derotest --data data --out results_resnet18_lr1e-4 --backbone resnet18 --lr 1e-4 --workers 2
if errorlevel 1 goto :fail
"%PY%" pds_exp1.py report --results results_resnet18_lr1e-4 > nul
if errorlevel 1 goto :fail

echo.
echo ===== Section 5c, mini / 迷你資料的 5c 節 =====
for %%D in (results results_swin_t results_resnet18_lr1e-4) do (
  echo.
  echo ----- %%D -----
  "%PY%" -c "import sys; t=open(sys.argv[1], encoding='utf-8').read(); print(t[t.find('## 5c'):t.find('## 5b')])" "%%D\summary.md"
)

if not exist "results_trainval\ckpt" (
  echo.
  echo [4/5] results_trainval not found: run run_exp1_A3.bat first, then run this again.
  echo       找不到 results_trainval：請先跑完 run_exp1_A3.bat，再執行一次。
  goto :done
)
echo [4/5] Trainval 85 scenes, ResNet-18 lr 3e-4 (long) / 85 個場景（較久）...
"%PY%" pds_exp1.py derotest --data data_trainval --out results_trainval --backbone resnet18 --lr 3e-4 --workers 2
if errorlevel 1 goto :fail
echo [5/5] Report / 報告...
"%PY%" pds_exp1.py report --results results_trainval > nul
if errorlevel 1 goto :fail
echo.
echo ----- results_trainval -----
"%PY%" -c "import sys; t=open(sys.argv[1], encoding='utf-8').read(); print(t[t.find('## 5c'):t.find('## 5b')])" "results_trainval\summary.md"

:done
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
