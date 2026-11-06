@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM ======================================================================
REM  PDS A3 - repeat the core result on 85 nuScenes trainval scenes
REM  Data: ..\exp0\nuscenes_trainval (metadata + keyframe blobs part 1)
REM  Backbone: ResNet-18, lr 3e-4 (the main setup). Level protocol only.
REM  Steps: prepare, render (no aug copies), train + evaluate, de-rotation
REM  with black and mirrored corners, noisy roll with mirrored corners, report.
REM  Every step is resumable: if it stops, double-click again.
REM  Time: several hours on an RTX 4050 - run it overnight.
REM ======================================================================

set "PY=..\exp0\.venv\Scripts\python.exe"
set "DATAROOT=..\exp0\nuscenes_trainval"
set "DATA=data_trainval"
set "OUT=results_trainval"

if not exist "%PY%" (
  echo [ERROR] Python environment not found. Run ..\exp0\run_exp0.bat first.
  echo         找不到 Python 環境，請先執行 ..\exp0\run_exp0.bat。
  pause
  exit /b 1
)
if not exist "%DATAROOT%\v1.0-trainval" (
  echo [ERROR] %DATAROOT%\v1.0-trainval not found. Extract the trainval metadata first.
  echo         找不到 trainval metadata，請先解壓。
  pause
  exit /b 1
)

echo [0/7] Geometry self-test / 幾何自我測試...
"%PY%" pds_exp1.py selftest
if errorlevel 1 goto :fail

if not exist "%DATA%\geometry.csv" (
  echo [1/7] Reading trainval geometry, only scenes on disk / 讀取 trainval 幾何資料，只用已下載的場景...
  "%PY%" pds_exp1.py prepare --dataroot "%DATAROOT%" --version v1.0-trainval --out %DATA%
  if errorlevel 1 goto :fail
) else (
  echo [1/7] %DATA%\geometry.csv exists, skipping / 已存在，略過
)

if not exist "%DATA%\index.csv" (
  echo [2/7] Rendering tilted crops, 1-3 hours / 產生傾斜裁切，約 1-3 小時...
  "%PY%" pds_exp1.py render --data %DATA% --aug-copies 0
  if errorlevel 1 goto :fail
) else (
  echo [2/7] %DATA%\index.csv exists, skipping / 已存在，略過
)

echo [3/7] Train + evaluate, 3 frames x 5 folds / 訓練與評估...
"%PY%" pds_exp1.py run --data %DATA% --out %OUT% --backbone resnet18 --lr 3e-4 --protocols level --frames cam alpha allo --workers 2
if errorlevel 1 goto :fail

echo [4/7] De-rotation, black corners / 轉正，黑色角落...
"%PY%" pds_exp1.py derot --data %DATA% --out %OUT% --backbone resnet18 --lr 3e-4 --workers 2
if errorlevel 1 goto :fail

echo [5/7] De-rotation, mirrored corners / 轉正，鏡射角落...
"%PY%" pds_exp1.py derot --data %DATA% --out %OUT% --backbone resnet18 --lr 3e-4 --fill reflect --workers 2
if errorlevel 1 goto :fail

echo [6/7] Noisy roll, mirrored corners / 含誤差側傾，鏡射角落...
"%PY%" pds_exp1.py derotnoise --data %DATA% --out %OUT% --backbone resnet18 --lr 3e-4 --fill reflect --workers 2
if errorlevel 1 goto :fail

echo [7/7] Report / 報告...
"%PY%" pds_exp1.py report --results %OUT%
if errorlevel 1 goto :fail

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
