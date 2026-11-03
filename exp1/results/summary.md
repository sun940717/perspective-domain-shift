# PDS Experiment 1 — Label frame × camera tilt / 標籤座標系 × 攝影機傾斜

5 folds · 1814 objects · every number below is PAIRED: error at tilt minus the same objects' error with a level camera, mean ± std over folds.
共 5 折、1814 個物體。下表皆為配對差：傾斜下誤差減去同一批物體在水平相機下的誤差（各折平均 ± 標準差）。

## 1. Degradation under tilt, extrinsics unknown / 傾斜造成的退化（外參未知）

| protocol | frame | level err (°) | pitch ±5 | pitch ±15 | roll ±5 | roll ±12 | roll ±20 |
|---|---|---|---|---|---|---|---|
| level | world | 90.3 ± 8.9 | -1.1 ± 1.0 | -1.3 ± 1.2 | -3.5 ± 2.8 | -5.8 ± 5.9 | -4.7 ± 9.3 |
| level | cam | 46.4 ± 3.2 | +0.3 ± 0.6 | +0.4 ± 0.7 | +1.8 ± 3.6 | +9.7 ± 7.6 | +19.5 ± 10.1 |
| level | alpha | 39.0 ± 4.8 | +0.3 ± 1.2 | +1.2 ± 2.8 | +3.5 ± 4.1 | +12.0 ± 7.1 | +20.7 ± 7.6 |
| level | allo | 41.0 ± 9.5 | -0.8 ± 0.9 | +1.7 ± 3.5 | +4.2 ± 3.7 | +12.0 ± 4.6 | +18.1 ± 5.1 |
| aug | world | 91.2 ± 9.8 | -1.0 ± 2.1 | -1.2 ± 1.8 | -0.8 ± 1.0 | +0.5 ± 2.1 | +0.2 ± 3.0 |
| aug | cam | 44.1 ± 3.6 | -1.1 ± 3.5 | -0.6 ± 3.9 | -1.4 ± 5.1 | -0.2 ± 6.6 | +1.6 ± 8.7 |
| aug | alpha | 31.4 ± 6.0 | +0.9 ± 0.5 | +1.6 ± 1.2 | +2.2 ± 1.4 | +4.0 ± 1.2 | +7.7 ± 2.7 |
| aug | allo | 35.7 ± 8.1 | +0.2 ± 0.6 | +0.3 ± 0.8 | +0.8 ± 0.7 | +2.1 ± 3.2 | +4.3 ± 6.4 |

`world` is at chance (~90°) everywhere, so its deltas carry no information.
`world` 在所有條件下都接近隨機（約 90°），其差值不具資訊量。

## 2. Does the frame change roll robustness? / 座標系是否改變側傾穩健性？

Per-fold difference in roll ±20° degradation (negative = first frame more robust).
各折在側傾 ±20° 下退化量的差（負值代表前者較穩健）。

| protocol | comparison | per fold | mean | folds favouring first |
|---|---|---|---|---|
| level | alpha − cam | -11.4, +17.6, +5.1, -4.2, -0.9 | +1.3 | 3/5 |
| level | allo − alpha | +1.1, -0.5, -13.2, -1.4, +0.6 | -2.7 | 3/5 |
| level | allo − cam | -10.3, +17.1, -8.1, -5.6, -0.3 | -1.4 | 4/5 |
| aug | alpha − cam | +2.9, +18.6, +1.6, +6.7, +0.9 | +6.1 | 0/5 |
| aug | allo − alpha | +0.0, -9.1, +0.3, -7.9, -0.5 | -3.4 | 3/5 |
| aug | allo − cam | +2.9, +9.5, +1.9, -1.2, +0.4 | +2.7 | 1/5 |

## 3. Does knowing the extrinsics help? / 已知外參是否有幫助？

Roll ±20° degradation by decoder. / 各解碼方式在側傾 ±20° 的退化量。

| protocol | frame | none | grav | ext | ext_noisy |
|---|---|---|---|---|---|
| level | cam | +19.5 ± 10.1 | — | +19.6 ± 9.9 | +19.6 ± 9.9 |
| level | alpha | +20.7 ± 7.6 | — | +20.8 ± 7.6 | +20.8 ± 7.7 |
| level | allo | +18.1 ± 5.1 | +18.2 ± 5.4 | +17.9 ± 5.3 | +17.9 ± 5.3 |
| aug | cam | +1.6 ± 8.7 | — | +1.8 ± 8.6 | +1.8 ± 8.6 |
| aug | alpha | +7.7 ± 2.7 | — | +7.8 ± 2.6 | +7.8 ± 2.6 |
| aug | allo | +4.3 ± 6.4 | +4.7 ± 6.1 | +4.4 ± 6.3 | +4.4 ± 6.3 |

## 4. Degradation curves / 退化曲線

![degradation](tilt_degradation.png)

Lines: mean over folds; bands: ±1 std over folds; decoder = none.
線：各折平均；色帶：各折 ±1 標準差；解碼方式 = none。

## 5. A1 mechanism check: rotate rolled crops back upright / 機制檢驗：把側傾裁切轉回直立

Level-trained models. `derot` rotates each crop by the TRUE roll before inference (oracle).
水平訓練的模型。`derot` 在推論前依真實側傾把裁切轉回（神諭）。

| frame | level err (°) | roll ±12 plain | roll ±12 derot | roll ±20 plain | roll ±20 derot |
|---|---|---|---|---|---|
| cam | 42.0 | +10.7 ± 5.3 | -0.2 ± 4.2 | +20.5 ± 4.9 | +1.2 ± 7.1 |
| alpha | 36.5 | +8.6 ± 2.7 | +2.0 ± 3.4 | +19.2 ± 5.7 | +3.1 ± 5.1 |
| allo | 42.9 | +11.4 ± 7.2 | +1.0 ± 5.2 | +19.0 ± 6.6 | +0.5 ± 7.2 |

Numbers are extra error vs the same objects with a level camera. A `derot` value near 0 means in-plane rotation explains the roll damage. Rotated crops gain black corners, a small extra handicap.
數值為與同一批物體水平相機相比多出的誤差。`derot` 接近 0 代表影像平面內旋轉可解釋側傾損害。轉正後的裁切角落會出現黑邊，是一個小額外不利因素。