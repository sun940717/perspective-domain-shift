@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM ======================================================================
REM  PDS Experiment 0 - one-click runner for Windows + NVIDIA GPU
REM  PDS 實驗 0 - Windows + NVIDIA 顯示卡一鍵執行
REM
REM  Folder layout / 資料夾結構:
REM    exp0\
REM      run_exp0.bat
REM      pds_exp0.py
REM      nuscenes\            <- v1.0-mini extracted here / v1.0-mini 解壓在這裡
REM        v1.0-mini\  samples\  sweeps\  maps\
REM ======================================================================

set "DATAROOT=%~dp0nuscenes"
set "PY=.venv\Scripts\python.exe"

if not exist "%DATAROOT%\v1.0-mini" (
  echo [ERROR] Cannot find "%DATAROOT%\v1.0-mini"
  echo         Extract nuScenes v1.0-mini into the nuscenes folder next to this file.
  echo         請把 nuScenes v1.0-mini 解壓到本檔案旁的 nuscenes 資料夾。
  pause
  exit /b 1
)

if exist "%PY%" goto :have_env

REM ---- Find a PyTorch-compatible Python: 3.10 > 3.11 > 3.12 ----------------
REM ---- 尋找 PyTorch 相容的 Python：3.10 > 3.11 > 3.12 -----------------------
set "BASEPY="
for %%V in (3.10 3.11 3.12) do (
  if not defined BASEPY (
    py -%%V -c "import sys" >nul 2>&1 && set "BASEPY=py -%%V"
  )
)
if not defined BASEPY (
  python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set "BASEPY=python"
)
if not defined BASEPY (
  echo [ERROR] No Python 3.10 / 3.11 / 3.12 found.
  echo         找不到 Python 3.10 / 3.11 / 3.12。
  echo.
  echo         Detected Pythons / 偵測到的 Python:
  py -0 2>nul
  python --version 2>nul
  echo.
  echo         Install Python 3.10 from python.org, tick "Add python.exe to PATH", then rerun.
  echo         請從 python.org 安裝 Python 3.10，勾選「Add python.exe to PATH」，再重新執行。
  pause
  exit /b 1
)

echo [1/5] Creating environment with: %BASEPY%
echo       建立環境，使用：%BASEPY%
%BASEPY% -m venv .venv
if errorlevel 1 goto :fail
"%PY%" -m pip install --upgrade pip
echo [2/5] Installing CUDA PyTorch / 安裝 CUDA 版 PyTorch...
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :fail
"%PY%" -m pip install nuscenes-devkit pandas numpy pillow matplotlib
if errorlevel 1 goto :fail

:have_env
echo [check] Python and GPU status / Python 與顯示卡狀態:
"%PY%" --version
"%PY%" -c "import torch; ok=torch.cuda.is_available(); print('  CUDA available:', ok); print('  GPU:', torch.cuda.get_device_name(0) if ok else 'NONE - will run on CPU (slow)')"

if not exist "crops\meta.csv" (
  echo [3/5] Extracting car crops / 擷取車輛裁切...
  "%PY%" pds_exp0.py extract --dataroot "%DATAROOT%" --out crops
  if errorlevel 1 goto :fail
) else (
  echo [3/5] crops\meta.csv exists, skipping extract / 已有裁切，略過
)

echo [4/5] Training 30 runs, resumable / 訓練 30 次，可續跑...
"%PY%" pds_exp0.py run --crops crops --out results --workers 2
if errorlevel 1 goto :fail

echo [5/5] Building report / 產生報告...
"%PY%" pds_exp0.py report --crops crops --results results
if errorlevel 1 goto :fail

echo.
echo Done. Results are in results\summary.md and results\err_vs_ray.png
echo 完成。結果在 results\summary.md 與 results\err_vs_ray.png
explorer results
pause
exit /b 0

:fail
echo.
echo [ERROR] A step failed. Copy the error text above and send it to Claude.
echo         有步驟失敗，請把上方錯誤訊息複製給 Claude。
pause
exit /b 1
