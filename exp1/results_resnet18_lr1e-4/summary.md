# PDS Experiment 1 — Label frame × camera tilt / 標籤座標系 × 攝影機傾斜 (backbone: resnet18, lr 0.0001)

5 folds · 1814 objects · every number below is PAIRED: error at tilt minus the same objects' error with a level camera, mean ± std over folds.
共 5 折、1814 個物體。下表皆為配對差：傾斜下誤差減去同一批物體在水平相機下的誤差（各折平均 ± 標準差）。

## 1. Degradation under tilt, extrinsics unknown / 傾斜造成的退化（外參未知）

| protocol | frame | level err (°) | pitch ±5 | pitch ±15 | roll ±5 | roll ±12 | roll ±20 |
|---|---|---|---|---|---|---|---|
| level | cam | 54.7 ± 9.5 | +0.7 ± 0.8 | +0.9 ± 1.2 | +3.4 ± 2.6 | +9.0 ± 5.0 | +14.6 ± 8.6 |
| level | alpha | 48.9 ± 7.5 | +0.1 ± 0.6 | +0.2 ± 1.1 | +1.9 ± 2.6 | +8.4 ± 5.7 | +13.9 ± 8.2 |
| level | allo | 50.1 ± 9.1 | +0.2 ± 0.4 | +2.3 ± 4.7 | +2.4 ± 3.0 | +6.6 ± 4.9 | +9.2 ± 6.8 |

`world` is at chance (~90°) everywhere, so its deltas carry no information.
`world` 在所有條件下都接近隨機（約 90°），其差值不具資訊量。

## 2. Does the frame change roll robustness? / 座標系是否改變側傾穩健性？

Per-fold difference in roll ±20° degradation (negative = first frame more robust).
各折在側傾 ±20° 下退化量的差（負值代表前者較穩健）。

| protocol | comparison | per fold | mean | folds favouring first |
|---|---|---|---|---|
| level | alpha − cam | -4.1, -4.3, +8.7, -7.0, +3.3 | -0.7 | 3/5 |
| level | allo − alpha | +0.9, -5.2, -1.0, -13.7, -4.4 | -4.7 | 4/5 |
| level | allo − cam | -3.2, -9.4, +7.6, -20.7, -1.0 | -5.4 | 4/5 |

## 3. Does knowing the extrinsics help? / 已知外參是否有幫助？

Roll ±20° degradation by decoder. / 各解碼方式在側傾 ±20° 的退化量。

| protocol | frame | none | grav | ext | ext_noisy |
|---|---|---|---|---|---|
| level | cam | +14.6 ± 8.6 | — | +14.7 ± 8.6 | +14.7 ± 8.6 |
| level | alpha | +13.9 ± 8.2 | — | +14.1 ± 8.2 | +14.1 ± 8.2 |
| level | allo | +9.2 ± 6.8 | +8.9 ± 6.5 | +9.1 ± 7.0 | +9.1 ± 7.0 |

## 4. Degradation curves / 退化曲線

![degradation](tilt_degradation.png)

Lines: mean over folds; bands: ±1 std over folds; decoder = none.
線：各折平均；色帶：各折 ±1 標準差；解碼方式 = none。

## 5. A1 mechanism check: rotate rolled crops back upright / 機制檢驗：把側傾裁切轉回直立

Level-trained models. `derot` rotates each crop by the TRUE roll before inference (oracle).
水平訓練的模型。`derot` 在推論前依真實側傾把裁切轉回（神諭）。

| frame | level err (°) | roll ±12 plain | roll ±12 derot | roll ±20 plain | roll ±20 derot |
|---|---|---|---|---|---|
| cam | 54.7 | +9.0 ± 5.0 | +2.2 ± 6.2 | +14.6 ± 8.6 | +3.6 ± 8.1 |
| alpha | 48.9 | +8.4 ± 5.7 | +2.7 ± 5.0 | +13.9 ± 8.2 | +4.0 ± 6.9 |
| allo | 50.1 | +6.6 ± 4.9 | +0.3 ± 6.9 | +9.2 ± 6.8 | +1.7 ± 8.0 |

Numbers are extra error vs the same objects with a level camera. A `derot` value near 0 means in-plane rotation explains the roll damage. Rotated crops gain black corners, a small extra handicap.
數值為與同一批物體水平相機相比多出的誤差。`derot` 接近 0 代表影像平面內旋轉可解釋側傾損害。轉正後的裁切角落會出現黑邊，是一個小額外不利因素。

## 5b. A1c corner-fill check: black vs mirrored corners / 角落填補檢驗：黑邊 vs 鏡射

Same models and crops as section 5. Rotating a square crop by 20° leaves ~12% of its pixels uncovered; `black` fills them with black (as in section 5), `reflect` with mirrored image content. A real system would de-rotate the whole frame, where such corners do not appear around the object.
與第 5 節相同的模型與裁切。正方形裁切旋轉 20° 會露出約 12% 的像素；`black` 以黑色填補（同第 5 節），`reflect` 以鏡射影像內容填補。實際系統會轉正整張影像，物體周圍不會出現這種角落。

| frame | roll ±20 plain | derot black | derot reflect | per fold, black | per fold, reflect | folds where reflect is closer to level |
|---|---|---|---|---|---|---|
| cam | +14.6 ± 8.6 | +3.6 ± 8.1 | +1.3 ± 1.1 | -1.5, +0.8, -3.7, +6.1, +16.5 | +0.4, -0.0, +1.9, +1.9, +2.5 | 5/5 |
| alpha | +13.9 ± 8.2 | +4.0 ± 6.9 | +0.9 ± 2.0 | -1.8, +0.2, +3.5, +2.5, +15.9 | -1.8, +0.8, +0.3, +1.6, +3.7 | 4/5 |
| allo | +9.2 ± 6.8 | +1.7 ± 8.0 | +0.2 ± 3.1 | -2.3, -2.7, -2.0, -0.1, +15.9 | +0.0, -0.1, -3.7, +0.2, +4.8 | 3/5 |

Extra error at roll ±20° vs the same objects with a level camera (mean ± std over folds).
側傾 ±20° 相對同一批物體水平相機多出的誤差（各折平均 ± 標準差）。

## 6. A1b de-rotation with a noisy roll estimate, black corners / 以含誤差的側傾估計轉正（黑色角落）

Each crop is rotated by (true roll + N(0, σ)); level crops are rotated by the noise alone. Extra error vs the same objects, level camera, NO rotation. Mean ± std over folds.
每個裁切依（真實側傾 + 常態誤差 σ）旋轉；水平裁切只依誤差旋轉。數值為相對於同一批物體、水平相機、不旋轉時多出的誤差（各折平均 ± 標準差）。

| frame | σ=0° level | σ=0° roll ±20 | σ=1° level | σ=1° roll ±20 | σ=2° level | σ=2° roll ±20 | σ=5° level | σ=5° roll ±20 |
|---|---|---|---|---|---|---|---|---|
| cam | +0.0 ± 0.0 | +3.6 ± 8.1 | +1.0 ± 2.1 | +4.0 ± 7.9 | +1.0 ± 1.5 | +4.4 ± 7.7 | +3.0 ± 1.4 | +5.1 ± 7.1 |
| alpha | +0.0 ± 0.0 | +4.0 ± 6.9 | +1.5 ± 1.1 | +4.1 ± 6.8 | +1.4 ± 1.7 | +4.4 ± 6.8 | +3.1 ± 2.2 | +6.1 ± 6.2 |
| allo | +0.0 ± 0.0 | +1.7 ± 8.0 | -0.6 ± 1.2 | +1.4 ± 8.2 | +0.2 ± 2.6 | +1.3 ± 8.2 | +1.6 ± 3.0 | +1.8 ± 7.3 |

## 6b. A1d the same with mirrored corners / 同上，鏡射角落

Each crop is rotated by (true roll + N(0, σ)); level crops are rotated by the noise alone. Extra error vs the same objects, level camera, NO rotation. Mean ± std over folds.
每個裁切依（真實側傾 + 常態誤差 σ）旋轉；水平裁切只依誤差旋轉。數值為相對於同一批物體、水平相機、不旋轉時多出的誤差（各折平均 ± 標準差）。

| frame | σ=0° level | σ=0° roll ±20 | σ=1° level | σ=1° roll ±20 | σ=2° level | σ=2° roll ±20 | σ=5° level | σ=5° roll ±20 |
|---|---|---|---|---|---|---|---|---|
| cam | +0.0 ± 0.0 | +1.3 ± 1.1 | +0.3 ± 1.1 | +1.4 ± 1.8 | +1.7 ± 1.5 | +2.0 ± 1.8 | +4.0 ± 2.6 | +2.9 ± 1.5 |
| alpha | +0.0 ± 0.0 | +0.9 ± 2.0 | +0.7 ± 1.5 | +1.2 ± 2.0 | +2.0 ± 1.3 | +1.6 ± 2.3 | +1.3 ± 2.3 | +2.7 ± 2.4 |
| allo | +0.0 ± 0.0 | +0.2 ± 3.1 | +0.0 ± 0.7 | -0.2 ± 2.5 | +0.8 ± 2.1 | +0.1 ± 2.9 | +2.2 ± 3.2 | +1.1 ± 1.8 |

For reference, roll ±20° WITHOUT any de-rotation costs +9.2 to +14.6° with this backbone (section 5).
對照：此骨幹在完全不轉正時，側傾 ±20° 多出 +9.2° 至 +14.6°（見第 5 節）。