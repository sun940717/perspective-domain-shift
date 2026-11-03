# PDS Experiment 0 — Does the label's frame decide viewpoint generalization?
# PDS 實驗 0 —— 標註座標系是否決定了視角泛化能力？

> Part of [perspective-domain-shift](https://github.com/sun940717/perspective-domain-shift).
> 屬於 perspective-domain-shift 研究專案的第一個可重現實驗。

---

## 1. Why this experiment exists / 為什麼要做這個實驗

PDS claims that a mismatch between **allocentric annotation** (labels stored in a world frame) and **egocentric inference** (a camera seeing the object) hurts perception.
PDS 主張：**他者中心標註**（標籤存在世界座標）與**自我中心推論**（相機看到物體）之間的落差，會損害感知效能。

Until now that claim only exists as text. This experiment turns it into one measurable number.
到目前為止，這個主張只存在於文字。這個實驗把它變成一個可量測的數字。

We use the simplest task where the mismatch can show up: **predicting a car's heading from an image crop**.
我們選擇最簡單、又能顯現這種落差的任務：**從影像裁切預測車輛朝向**。

---

## 2. Hypothesis / 假設

The same label, expressed in three frames:
同一個標籤，用三種座標系表達：

| Target | Meaning | 意義 |
|---|---|---|
| `world` | yaw in the global annotation frame | 原始標註的世界座標 yaw |
| `cam` | yaw in the camera frame | 相機座標 yaw |
| `alpha` | observation angle = `cam_yaw − ray_angle` | 觀測角 = 相機 yaw − 視線角 |

A crop's appearance depends on `alpha`, not on where the car sits in the world.
裁切的外觀取決於 `alpha`，而不是車在世界中的位置。

**H1:** error ranks `alpha < cam < world`.
**H1：** 誤差排序為 `alpha < cam < world`。

**H2:** `cam` error grows with the off-center angle `|ray|`; `alpha` error stays flat.
**H2：** `cam` 的誤差隨偏離中心角 `|ray|` 增加；`alpha` 的誤差保持平坦。

**H3:** when training only on `CAM_FRONT`, `alpha` transfers to the other five cameras better than `cam`.
**H3：** 只用 `CAM_FRONT` 訓練時，`alpha` 轉移到其他五台相機的效果優於 `cam`。

---

## 3. Honest positioning / 誠實的定位

H1 and H2 reproduce a known effect: Mousavian et al., *3D Bounding Box Estimation Using Deep Learning and Geometry* (CVPR 2017), regress the local/observation angle for exactly this reason.
H1 與 H2 是在重現已知效應：Mousavian 等人於 CVPR 2017 的論文，正是因為這個原因改為回歸局部觀測角。

That is intentional. Exp 0 is a **calibration experiment**: it proves the pipeline is correct and gives PDS a quantitative baseline.
這是刻意的。實驗 0 是一個**校準實驗**：證明流程正確，並給 PDS 一個量化基準。

The novel part of PDS must come in Exp 1+, by showing a gap that the `alpha` trick does **not** close (see §8).
PDS 的創新部分必須在實驗 1 之後出現：找出 `alpha` 技巧**無法**消除的落差（見 §8）。

---

## 4. Setup / 實驗設定

| Item | Choice | 說明 |
|---|---|---|
| Data | nuScenes `v1.0-mini` (10 scenes, 6 cameras) | 免費、需註冊下載，約 4 GB |
| Objects | `vehicle.car`, 60–100% visible, ≥5 lidar pts, ≥32 px | 只取可見度高的轎車 |
| Model | ResNet-18 (ImageNet init) → (sin, cos) | 輸出角度的 sin 與 cos |
| Loss | `1 − cos(Δθ)` | 角度誤差的餘弦損失 |
| Split | 5-fold **leave-scenes-out** | 以場景切分，避免同一台車同時出現在訓練與測試 |
| Augment | color jitter, small translate/scale, **no flips** | 不做翻轉，翻轉會改變真實角度 |
| Metric | mean / median angular error, acc@30° | 平均／中位數角度誤差、30° 內準確率 |
| Sanity | constant-prediction baseline per target | 每個目標都附「常數預測」基準 |

---

## 5. How to run / 執行方式

### 5.1 Get the data / 取得資料

1. Register at nuscenes.org, download **v1.0-mini**, and extract it so the folder contains `samples/`, `sweeps/`, `maps/`, `v1.0-mini/`.
1. 到 nuscenes.org 註冊，下載 **v1.0-mini**，解壓後資料夾內應有 `samples/`、`sweeps/`、`maps/`、`v1.0-mini/`。

### 5.2 Install / 安裝（Python 3.10）

```bash
pip install torch torchvision nuscenes-devkit pandas numpy pillow matplotlib
```

### 5.3 Run / 執行

```bash
# Step 1: crop cars from all 6 cameras / 從六台相機裁切車輛
python pds_exp0.py extract --dataroot /path/to/nuscenes --out ./crops

# Step 2: 5 folds x 2 protocols x 3 targets = 30 runs / 共 30 次訓練
python pds_exp0.py run --crops ./crops --out ./results

# Step 3: tables + plot / 產出表格與圖
python pds_exp0.py report --crops ./crops --results ./results
```

Runs are resumable: finished runs are skipped, so a Colab disconnect costs nothing.
訓練可續跑：已完成的會自動略過，Colab 斷線不會白費。

Expected time on a free Colab/Kaggle T4: roughly 1 hour total.
在免費 Colab／Kaggle T4 上預估總共約 1 小時。

---

## 6. Outputs / 產出

| File | Content | 內容 |
|---|---|---|
| `crops/meta.csv` | per-crop geometry (world_yaw, cam_yaw, ray, alpha, depth…) | 每個裁切的幾何資訊 |
| `results/preds/*.csv` | per-run predictions and errors | 每次訓練的預測與誤差 |
| `results/summary.md` | Table 1 overall, Table 2 cross-camera, Table 3 error vs. `|ray|` | 三張結果表 |
| `results/err_vs_ray.png` | error vs. off-center angle, one line per target | 誤差 vs. 偏離角度圖 |

---

## 7. How to read the results / 如何解讀結果

- `world` ≈ constant baseline → the model cannot learn world yaw from a crop at all. This is the PDS mismatch in its purest form.
- `world` ≈ 常數基準 → 模型完全無法從裁切學到世界 yaw。這是 PDS 落差最純粹的形態。
- `cam` line rising with `|ray|` while `alpha` stays flat → H2 confirmed.
- `cam` 曲線隨 `|ray|` 上升而 `alpha` 平坦 → H2 成立。
- If `alpha` still degrades on `CAM_BACK*` in Table 2 → there is a residual gap the frame fix does not explain. **That residual is the seed of Exp 1.**
- 如果表 2 中 `alpha` 在後方相機仍然變差 → 存在座標修正無法解釋的殘餘落差。**這個殘餘就是實驗 1 的種子。**
- A negative or surprising result is still publishable as an honest report.
- 結果不如預期也沒關係，誠實報告本身就有價值。

### Pipeline validation / 流程驗證

The script was smoke-tested on synthetic crops whose appearance depends only on `alpha` (2 folds, 4 epochs, CPU).
程式已用「外觀只取決於 `alpha`」的合成資料做過冒煙測試（2 folds、4 epochs、CPU）。

It recovered the designed behaviour: `alpha` 4.0°, `cam` 18.2°, `world` 88.9° (≈ random), and `cam` error rose from 5.8° to 32.1° as `|ray|` grew.
結果符合設計：`alpha` 4.0°、`cam` 18.2°、`world` 88.9°（≈ 隨機），且 `cam` 誤差隨 `|ray|` 從 5.8° 升到 32.1°。

These are synthetic numbers for checking the code only; they are **not** results.
以上是用來檢查程式的合成數字，**不是**實驗結果。

---

## 8. Roadmap / 後續路線

| Stage | Goal | 目標 |
|---|---|---|
| Exp 0 (this) | reproduce the frame effect on nuScenes-mini | 在 nuScenes-mini 重現座標系效應 |
| Exp 0b | repeat on full nuScenes trainval subset, more categories | 擴大到完整資料子集與更多類別 |
| Exp 1 | isolate the residual gap (camera height, pitch, intrinsics) that `alpha` does not remove | 分離出 `alpha` 無法消除的殘餘落差（相機高度、俯仰、內參） |
| Exp 2 | propose a PDS-specific fix and beat the `alpha` baseline | 提出 PDS 專屬方法並勝過 `alpha` 基準 |
| Write-up | short arXiv preprint with Exp 0–2 | 以實驗 0–2 撰寫 arXiv 短論文 |

---

## 9. Checklist before pushing to GitHub / 推上 GitHub 前的檢查清單

- [ ] Real `summary.md` and `err_vs_ray.png` committed (not the synthetic ones)
- [ ] 提交真實的 `summary.md` 與 `err_vs_ray.png`（不是合成版）
- [ ] `folds.json` committed so anyone can reproduce the split
- [ ] 提交 `folds.json`，讓任何人都能重現資料切分
- [ ] Do **not** commit nuScenes images (license: non-commercial, no redistribution)
- [ ] **不要**提交 nuScenes 影像（授權禁止再散布）
- [ ] Note GPU, torch version, and runtime in this README
- [ ] 在 README 註明 GPU、torch 版本與執行時間
