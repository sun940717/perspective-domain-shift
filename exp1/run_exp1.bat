@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM ======================================================================
REM  PDS Experiment 1 - label frame x camera tilt, one-click runner
REM  PDS 實驗 1 - 標籤座標系 x 攝影機傾斜，一鍵執行
REM
REM  Reuses the Python environment and nuScenes data from ..\exp0
REM  沿用 ..\exp0 的 Python 環境與 nuScenes 資料
REM  Disk needed: about 1 GB for rendered crops / 需要約 1 GB 存放裁切
REM ======================================================================

set "PY=..\exp0\.venv\Scripts\python.exe"
set "DATAROOT=..\exp0\nuscenes"

if not exist "%PY%" (
  echo [ERROR] Python environment not found. Run ..\exp0\run_exp0.bat first.
  echo         找不到 Python 環境，請先執行 ..\exp0\run_exp0.bat。
  pause
  exit /b 1
)
"%PY%" -c "import cv2" >nul 2>&1 || "%PY%" -m pip install opencv-python

echo [1/5] Geometry self-test / 幾何自我測試...
"%PY%" pds_exp1.py selftest
if errorlevel 1 goto :fail

if not exist "data\geometry.csv" (
  echo [2/5] Reading nuScenes geometry / 讀取 nuScenes 幾何資料...
  "%PY%" pds_exp1.py prepare --dataroot "%DATAROOT%" --out data
  if errorlevel 1 goto :fail
) else (
  echo [2/5] data\geometry.csv exists, skipping / 已存在，略過
)

if not exist "data\index.csv" (
  echo [3/5] Rendering tilted crops, about 10-20 min / 產生傾斜裁切，約 10-20 分鐘...
  "%PY%" pds_exp1.py render --data data
  if errorlevel 1 goto :fail
) else (
  echo [3/5] data\index.csv exists, skipping / 已存在，略過
)

echo [4/5] Training 4 frames x 2 protocols x 5 folds, resumable / 訓練 40 次，可續跑...
"%PY%" pds_exp1.py run --data data --out results --workers 2
if errorlevel 1 goto :fail

echo [4b] A1 mechanism check: de-rotate rolled crops, about 15 min / 機制檢驗：把側傾裁切轉回，約 15 分鐘...
"%PY%" pds_exp1.py derot --data data --out results --workers 2
if errorlevel 1 goto :fail

echo [4c] A1b noisy-roll de-rotation, about 10 min / 含誤差側傾轉正，約 10 分鐘...
"%PY%" pds_exp1.py derotnoise --data data --out results --workers 2
if errorlevel 1 goto :fail

echo [5/5] Building report / 產生報告...
"%PY%" pds_exp1.py report --results results
if errorlevel 1 goto :fail

echo.
echo Done. See results\summary.md and results\tilt_degradation.png
echo 完成。請看 results\summary.md 與 results\tilt_degradation.png
explorer results
pause
exit /b 0

:fail
echo.
echo [ERROR] A step failed. Copy the error text above and send it to Claude.
echo         有步驟失敗，請把上方錯誤訊息複製給 Claude。
pause
exit /b 1
