# PDS Experiment 1 — Label frame × camera tilt
# PDS 實驗 1 —— 標籤座標系 × 攝影機傾斜

**Question:** a heading model is trained on level cameras; at test time the camera pitches or rolls and its extrinsics are unknown. How much accuracy does each label frame keep?
**問題：**朝向模型以水平相機訓練；測試時攝影機俯仰或側傾且外參未知。每種標籤座標系能保留多少準確度？

## Design / 設計

| Item | Choice | 說明 |
|---|---|---|
| Label frames | `world`, `cam`, `alpha`, `allo` (6D rotation relative to the pixel ray) | 世界 yaw、相機 yaw、觀測角、相對像素視線的 6D 旋轉 |
| Decoders | `none` (no extrinsics), `grav` (allo: object up-axis as gravity), `ext` (exact), `ext_noisy` (±2° noise) | 無外參、以物體上方向估重力、精確外參、含 2° 誤差外參 |
| Protocols | `level` (train level only), `aug` (train level + random tilt) | 只用水平訓練；水平 + 隨機傾斜訓練 |
| Tilt test grid | pitch 0, ±1, ±2, ±3, ±5, ±8, ±12, ±15°; roll 0 … ±20° | 俯仰到 ±15°，側傾到 ±20° |
| Tilt simulation | exact homography `H = K R K⁻¹`; labels rotated by the same `R` | 精確單應變形；標籤以同一個 R 旋轉 |
| Fair comparison | only objects visible in **every** tilt condition | 只用在**所有**傾斜條件下都可見的物體 |
| Folds | 5-fold leave-scenes-out, same split as Exp 0 | 與實驗 0 相同的 5 折場景切分 |

Pitch is capped at ±15° because nuScenes cameras see only ~39° vertically; beyond that, cars near the horizon leave the frame.
俯仰上限 ±15°，因為 nuScenes 相機垂直視野只有約 39°；超過後地平線附近的車會離開畫面。

## Verified before training / 訓練前已驗證

`python pds_exp1.py selftest` runs 14 geometry checks: the warp equals a true 3D rotation, perfect predictions decode back exactly with known extrinsics, and `allo/grav` recovers the heading exactly under pure pitch or pure roll **without** extrinsics.
`selftest` 執行 14 項幾何檢查：變形等同真正的 3D 旋轉、已知外參時完美預測可精確解碼回來、且 `allo/grav` 在純俯仰或純側傾下**不需外參**就能精確還原朝向。

**Geometric floor / 幾何下限：** with perfect predictions and no extrinsics, yaw-only frames lose at most 1.1° at 20° of tilt. So any large loss under tilt comes from what the model learned, not from label geometry.
**幾何下限：**預測完美且無外參時，只有 yaw 的座標系在 20° 傾斜下最多只損失 1.1°。所以傾斜下的大幅損失來自模型學到的東西，而不是標籤幾何。

## Run / 執行

Run `..\exp0\run_exp0.bat` first (it creates the Python environment), then double-click `run_exp1.bat`.
先執行 `..\exp0\run_exp0.bat`（建立 Python 環境），再雙擊 `run_exp1.bat`。

Outputs: `results\summary.md`, `results\tilt_level.png`, `results\tilt_aug.png`.
輸出：`results\summary.md`、`results\tilt_level.png`、`results\tilt_aug.png`。
