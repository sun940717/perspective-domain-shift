# Perspective Domain Shift (PDS)
## 視角分布偏移：跨視角表徵轉換統一框架

> **Research Proposal v0.1** | HongChenHe · NKUST CCE  
> Initial commit for timestamping purposes · 初始提交，用於時間戳記錄

---

## Abstract · 摘要

**EN** — This proposal introduces **Perspective Domain Shift (PDS)** as a formally defined machine learning problem class. PDS occurs when a model is trained on data labeled from one spatial perspective (allocentric / third-person / bird's-eye) but must perform inference from a different perspective (egocentric / first-person / camera view). We argue PDS is a distinct, general-purpose problem that has been solved ad hoc across individual domains but never formalized as a unified, transferable framework. This work aims to: (1) formally define PDS, (2) design a modular Perspective Transformation Module (PTM), (3) construct a cross-domain benchmark, and (4) demonstrate applicability across autonomous driving, robotics, and VR/XR training.

**ZH** — 本計劃書正式定義「視角分布偏移（PDS）」為一個機器學習問題類別。PDS 發生於模型以某一空間視角的標注資料訓練（如俯視/第三人稱），但推論時卻需在另一視角下運作（如第一人稱攝影機）時。我們主張 PDS 是一個獨立且通用的問題類別，目前各領域雖有 ad hoc 解法，但從未被正式化為統一可遷移的框架。本研究目標：(1) 正式定義 PDS；(2) 設計通用視角轉換模組（PTM）；(3) 建構跨領域 benchmark；(4) 驗證於自動駕駛、機器人及 VR/XR 訓練場景。

---

## 01 · Motivation · 研究動機

### The Core Observation · 核心觀察

**EN** — Consider Tesla's Autopilot. Human annotators label road scenes from a third-person or map-relative perspective (lane boundaries, bounding boxes, traffic rules). Yet the deployed system must interpret a first-person camera stream in real time. The gap between the labeling perspective and the inference perspective introduces systematic representation mismatch — a phenomenon we call **Perspective Domain Shift**.

**ZH** — 以特斯拉 Autopilot 為例：人類標注者以第三人稱或地圖視角標記道路場景（車道線、物件框、交通規則），但部署系統需即時解讀第一人稱攝影機畫面。標注視角與推論視角之間的落差，引入了系統性的表徵不匹配——這就是我們所定義的「視角分布偏移」。

### Why This Has Not Been Solved · 為何尚未被系統性解決

**EN** — Existing work addresses PDS implicitly and locally. Tesla's BEV Transformer, the E3AD egocentric/allocentric dual-pathway, and multi-agent prediction frameworks all solve perspective mismatch within their own domain pipelines. No prior work has: (a) named PDS as a standalone problem class, (b) proposed a domain-agnostic transformation module, or (c) provided a unified benchmark for measuring PDS severity and mitigation quality.

**ZH** — 現有研究以隱式、局部的方式處理 PDS。特斯拉的 BEV Transformer、E3AD 雙路徑框架、多 agent 軌跡預測框架，都在各自的領域管線中解決視角不匹配，但從未有人：(a) 將 PDS 命名為獨立問題類別；(b) 提出領域無關的轉換模組；(c) 建立衡量 PDS 嚴重性與緩解品質的統一 benchmark。

---

## 02 · Formal Definition · 正式定義

**EN** — Let `D_train = {(x_i, y_i)}` be a training set where `x_i` is observed and `y_i` is labeled from perspective `P_A` (allocentric / third-person). Let `D_test = {x_j}` be a test set where `x_j` is observed from perspective `P_E` (egocentric / first-person). **PDS** is the distribution shift induced by `P_A ≠ P_E`, causing a model trained on `D_train` to suffer systematic generalization loss on `D_test` beyond standard covariate shift.

**ZH** — 設 `D_train = {(x_i, y_i)}` 為訓練集，其中 `x_i` 由視角 `P_A`（allocentric / 第三人稱）觀測並標注。設 `D_test = {x_j}` 為測試集，`x_j` 由視角 `P_E`（egocentric / 第一人稱）觀測。**PDS** 為 `P_A ≠ P_E` 所誘發的分布偏移，使模型在 `D_test` 上產生超出一般協變量偏移的系統性泛化損失。

### Key Dimensions of PDS · PDS 的關鍵維度

| Dimension | EN Description | ZH 描述 |
|-----------|---------------|---------|
| Spatial origin shift | Ego-center vs. environment-center coordinate origin | 自我中心 vs. 環境中心坐標原點 |
| Depth & scale ambiguity | 2D image does not encode absolute depth; BEV does | 2D 影像不編碼絕對深度，BEV 則有 |
| Occlusion asymmetry | First-person view has occlusion patterns absent in top-down | 第一視角有俯視圖中不存在的遮擋模式 |
| Temporal alignment | Label timestamps may not align with egocentric frame capture | 標注時間戳可能與自我視角幀捕獲時間不一致 |

---

## 03 · Proposed Framework — PTM · 提案框架

**EN** — We propose the **Perspective Transformation Module (PTM)**: a plug-in intermediate layer that sits between any backbone encoder and task head. PTM learns a parameterized mapping `f: Z_A → Z_E` from allocentric latent representations `Z_A` to egocentric latent representations `Z_E`, without requiring explicit geometric camera calibration.

**ZH** — 我們提出「視角轉換模組（PTM）」：一個可插拔的中間層，置於任意骨幹編碼器與任務頭之間。PTM 學習參數化映射 `f: Z_A → Z_E`，將 allocentric 潛在表徵 `Z_A` 轉換為 egocentric 潛在表徵 `Z_E`，無需顯式的幾何相機校準。

### Architecture · 架構

| Component | EN | ZH |
|-----------|----|----|
| Input encoder | Shared backbone (ResNet / ViT) extracts `Z_A` | 共享骨幹提取 `Z_A` |
| PTM core | Transformer cross-attention: query = ego pose token, key/value = `Z_A` | 交叉注意力，query = 自我視角姿態 token |
| Pose conditioning | Inject relative pose `Δ(P_A, P_E)` to modulate attention | 注入相對姿態 `Δ(P_A, P_E)` 調製注意力 |
| Output head | Task head operates on `Z_E` | 任務頭在 `Z_E` 上運作 |

### Training Objective · 訓練目標

```
L_total = L_task + λ₁ · L_perspective + λ₂ · L_cycle
```

- `L_task` — downstream task loss · 下游任務損失  
- `L_perspective` — cosine similarity between transformed `Z_E` and ground-truth egocentric features · 視角對齊損失  
- `L_cycle` — invertibility: `f⁻¹(f(Z_A)) ≈ Z_A` · 可逆性約束  

---

## 04 · Research Contributions · 研究貢獻

| # | EN | ZH |
|---|----|----|
| C1 | **Formal Definition** — First formal definition of PDS as a standalone ML problem class, with measurable PDS-score metric | 首次將 PDS 正式定義為獨立 ML 問題類別，提出可量化 PDS-score |
| C2 | **PTM Architecture** — Domain-agnostic, plug-in module attachable to any encoder-decoder without full retraining | 領域無關、可插拔的 PTM，無需重新訓練完整模型 |
| C3 | **Cross-Domain Benchmark** — Spanning autonomous driving (nuScenes), robotics (RLBench), and VR training datasets | 涵蓋自駕、機器人、VR 的統一 benchmark |
| C4 | **Theoretical Analysis** — Generalization bounds relating PDS-score to downstream task error | 推導 PDS-score 與下游任務誤差關係的泛化誤差上界 |

---

## 05 · Application Domains · 應用領域

| Domain | EN | ZH |
|--------|----|----|
| Autonomous Driving | Training labeled from maps/BEV; inference on 8-camera ego stream. PTM replaces ad hoc BEV transformers. | 訓練由地圖/BEV 標注；推論時使用 8 路自我視角攝影機。PTM 取代 ad hoc BEV 轉換器。 |
| Robotics | Demonstrations from overhead cameras; robot acts from wrist-mounted ego camera. PTM bridges the embodiment gap. | 示範由頭頂攝影機錄製；機器人從腕部自我視角攝影機執行。PTM 橋接具身性落差。 |
| VR / XR Training | Scenarios authored in God-view; trainee experiences first-person. PTM improves scene-knowledge transfer. | 場景由上帝視角設計；受訓者在第一人稱體驗。PTM 改善場景知識遷移。 |

---

## 06 · Research Roadmap · 研究路線圖

| Phase | EN Goal | ZH 目標 |
|-------|---------|---------|
| P1 | Literature survey + formal PDS definition + PDS-score metric | 文獻調查 + PDS 正式定義 + PDS-score 指標 |
| P2 | PTM architecture design + synthetic dataset validation | PTM 架構設計 + 合成資料集驗證 |
| P3 | Cross-domain benchmark construction (driving / robotics / VR) | 跨領域 benchmark 建構 |
| P4 | Full experiments + ablation + paper writeup (target: workshop / arXiv) | 完整實驗 + 消融研究 + 論文撰寫 |

---

## 07 · Related Work & Differentiation · 相關研究與差異化

| Work | Scope | PDS-aware? | Portable? |
|------|-------|-----------|-----------|
| Tesla BEV Transformer [Karpathy, AI Day 2021] | Autonomous driving only | Implicit | No |
| E3AD [arXiv 2512.04733, 2025] | VLA driving only | Implicit | No |
| Multi-Agent Trajectory Prediction [Jia et al., CoRL 2022] | Vehicle/pedestrian prediction | Implicit | No |
| **This work — PDS + PTM** | **General / cross-domain** | **Explicit + formal** | **Yes** |

---

## 08 · Open Questions · 開放問題

- Can PTM be pre-trained on synthetic data and zero-shot transferred? · PTM 能否在合成資料上預訓練並做零樣本遷移？
- How does PDS severity scale with perspective angular distance `Δθ`? · PDS 嚴重性如何隨視角角度距離 `Δθ` 縮放？
- Does PTM implicitly learn place-cell-like representations analogous to hippocampal spatial memory? · PTM 是否隱式學習類似海馬迴空間記憶的 place-cell 表徵？
- Extension to temporal PDS: misalignment between label assignment time and ego-observation time. · 擴展至時序 PDS：標注時間與自我視角觀測時間的不對齊問題。

---

## Status · 狀態

```
[v0.1] 2026-03 — Initial proposal + formal definition (this commit)
[v0.2] TBD     — PTM architecture diagram + pseudocode
[v0.3] TBD     — Synthetic experiment results
[v1.0] TBD     — Full paper draft
```

---

*This repository is an early-stage research record for timestamping and IP documentation purposes. All claims are preliminary.*  
*本 repo 為早期研究記錄，用於時間戳與 IP 留存。所有主張均為初步草案。*
