#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS Experiment 0 — Orientation target frame vs. viewpoint generalization
PDS 實驗 0 —— 朝向標註座標系 vs. 視角泛化能力

Question / 問題
    When a model learns object heading from an image crop, does the *frame* the
    label is expressed in (world / camera / observation-angle) change how well it
    generalizes to new positions in the image and to unseen cameras?
    當模型從影像裁切學習物體朝向時，標註所用的座標系（世界／相機／觀測角）
    是否會影響它對影像中新位置、以及未見過相機的泛化能力？

Targets / 預測目標
    world : yaw in the global (allocentric annotation) frame        世界座標 yaw（原始標註座標系）
    cam   : yaw in the camera frame (egocentric)                    相機座標 yaw（自我中心）
    alpha : observation angle = cam_yaw - ray_angle (view-relative) 觀測角 = 相機 yaw − 視線角

Protocols / 實驗協定
    all    : train on all 6 cameras, test on held-out scenes          六台相機訓練，於保留場景測試
    front  : train on CAM_FRONT only, test on all 6 cameras           只用前相機訓練，測全部六台
    allsub : all 6 cameras, subsampled to the SAME size as `front`    六台相機，但抽樣到與 front 相同數量
             (control: separates the viewpoint effect from the data-size effect)
             （對照組：把視角效應與資料量效應分開）

Usage / 用法
    python pds_exp0.py extract --dataroot /path/to/nuscenes --out ./crops
    python pds_exp0.py run     --crops ./crops --out ./results
    python pds_exp0.py report  --crops ./crops --results ./results

Requires Python 3.10, torch, torchvision, nuscenes-devkit, pandas, numpy, pillow, matplotlib.
Windows: run via run_exp0.bat (uses the GPU if CUDA torch is installed).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CAMS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
TARGETS = ["world", "cam", "alpha"]
TARGET_COL = {"world": "world_yaw", "cam": "cam_yaw", "alpha": "alpha"}
PROTOCOLS = ["all", "front", "allsub"]
RAY_BINS_DEG = [0, 5, 10, 15, 20, 25, 30, 40, 90]


# --------------------------------------------------------------------------- #
# Geometry helpers / 幾何工具
# --------------------------------------------------------------------------- #
def wrap(a):
    """Wrap angle(s) to [-pi, pi). 將角度包回 [-pi, pi)。"""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(a: np.ndarray) -> float:
    return float(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def ang_err_deg(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.degrees(np.abs(wrap(pred - gt)))


# --------------------------------------------------------------------------- #
# Step 1: extract crops from nuScenes / 從 nuScenes 擷取物體裁切
# --------------------------------------------------------------------------- #
def cmd_extract(args: argparse.Namespace) -> None:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    for si, sample in enumerate(nusc.sample):
        scene_name = nusc.get("scene", sample["scene_token"])["name"]
        for cam in CAMS:
            sd_token = sample["data"][cam]
            img_path, boxes, K = nusc.get_sample_data(
                sd_token, box_vis_level=BoxVisibility.ALL
            )
            if not boxes:
                continue
            K = np.asarray(K)
            img = None
            for box in boxes:
                if not box.name.startswith(args.category):
                    continue
                ann = nusc.get("sample_annotation", box.token)
                if ann["visibility_token"] not in ("3", "4"):  # 60-100% visible
                    continue
                if ann["num_lidar_pts"] < args.min_lidar_pts:
                    continue

                c = box.center  # camera frame: x right, y down, z forward
                if c[2] <= 0.5:
                    continue

                corners = view_points(box.corners(), K, normalize=True)[:2]
                x0, y0 = corners.min(axis=1)
                x1, y1 = corners.max(axis=1)
                w, h = x1 - x0, y1 - y0
                if max(w, h) < args.min_size:
                    continue

                # Heading vector of the box (box x-axis) in camera frame.
                v = box.orientation.rotation_matrix @ np.array([1.0, 0.0, 0.0])
                cam_yaw = math.atan2(v[0], v[2])
                ray = math.atan2(c[0], c[2])
                alpha = float(wrap(cam_yaw - ray))
                world_yaw = float(nusc.get_box(box.token).orientation.yaw_pitch_roll[0])

                if img is None:
                    img = Image.open(img_path).convert("RGB")
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                s = max(w, h) * 1.15 / 2.0
                crop = img.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s)))
                crop = crop.resize((args.size, args.size), Image.BILINEAR)

                fname = f"{box.token}_{cam}.jpg"
                crop.save(out / "img" / fname, quality=92)
                rows.append(
                    {
                        "file": fname,
                        "ann_token": box.token,
                        "instance_token": ann["instance_token"],
                        "scene": scene_name,
                        "cam": cam,
                        "world_yaw": world_yaw,
                        "cam_yaw": cam_yaw,
                        "ray": ray,
                        "alpha": alpha,
                        "depth": float(c[2]),
                        "u": float(cx),
                        "box_px": float(max(w, h)),
                    }
                )
        if (si + 1) % 50 == 0:
            print(f"  samples {si + 1}/{len(nusc.sample)}  crops={len(rows)}  "
                  f"{time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "meta.csv", index=False)
    print(f"\nSaved {len(df)} crops -> {out / 'meta.csv'}")
    if len(df):
        print("\nCrops per camera / 各相機裁切數:")
        print(df.groupby("cam").size().to_string())
        print("\nCrops per scene / 各場景裁切數:")
        print(df.groupby("scene").size().to_string())


# --------------------------------------------------------------------------- #
# Step 2: train & evaluate / 訓練與評估
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_folds(scenes: list[str], n_folds: int, seed: int = 0) -> list[list[str]]:
    scenes = sorted(scenes)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_folds = min(n_folds, len(scenes))
    return [scenes[i::n_folds] for i in range(n_folds)]


# Module-level so Windows DataLoader workers (spawn) can pickle it.
# 定義在模組層級，Windows 的 DataLoader 多程序（spawn）才能序列化。
class CropDS:
    """Image crop -> (sin, cos) of the chosen target angle.
    No horizontal flips or rotations: they would change the true angle.
    不做翻轉或旋轉：它們會改變真實角度。"""

    def __init__(self, df: pd.DataFrame, root: Path, target: str,
                 train: bool, size: int):
        from torchvision import transforms as T

        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.paths = [root / "img" / f for f in df["file"]]
        self.y = df[TARGET_COL[target]].to_numpy(np.float32)
        if train:
            self.tf = T.Compose([
                T.Resize((size, size)),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                norm,
            ])
        else:
            self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), norm])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        import torch

        x = self.tf(Image.open(self.paths[i]).convert("RGB"))
        a = float(self.y[i])
        y = torch.tensor([math.sin(a), math.cos(a)], dtype=torch.float32)
        return x, y, i


def build_model(pretrained: bool):
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, root: Path,
                      target: str, args: argparse.Namespace, device, seed: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    set_seed(seed)
    model = build_model(args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    dl_tr = DataLoader(
        CropDS(train_df, root, target, True, args.size),
        batch_size=args.bs, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=len(train_df) > args.bs,
        persistent_workers=args.workers > 0,
    )
    dl_te = DataLoader(
        CropDS(test_df, root, target, False, args.size),
        batch_size=args.bs * 2, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            p = F.normalize(model(x), dim=1)
            loss = (1.0 - (p * y).sum(dim=1)).mean()  # 1 - cos(angle error)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        if args.verbose:
            print(f"      epoch {ep + 1}/{args.epochs}  loss={tot / max(n, 1):.4f}")

    model.eval()
    preds = np.zeros(len(test_df), dtype=np.float64)
    with torch.no_grad():
        for x, _, idx in dl_te:
            p = model(x.to(device))
            preds[idx.numpy()] = torch.atan2(p[:, 0], p[:, 1]).cpu().numpy()
    return preds


def cmd_run(args: argparse.Namespace) -> None:
    import torch

    root = Path(args.crops)
    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "meta.csv")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}   crops: {len(df)}")

    folds = make_folds(df["scene"].unique().tolist(), args.folds, seed=0)
    (out / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

    protocols = args.protocols or PROTOCOLS
    targets = args.targets or TARGETS
    for k, held_out in enumerate(folds):
        test_df = df[df["scene"].isin(held_out)].reset_index(drop=True)
        base_train = df[~df["scene"].isin(held_out)]
        n_front = int((base_train["cam"] == "CAM_FRONT").sum())
        for proto in protocols:
            if proto == "all":
                train_df = base_train
            elif proto == "front":
                train_df = base_train[base_train["cam"] == "CAM_FRONT"]
            else:  # allsub: same size as front, all cameras / 與 front 同數量、涵蓋所有相機
                train_df = base_train.sample(n=n_front, random_state=args.seed + k)
            train_df = train_df.reset_index(drop=True)
            for target in targets:
                f = out / "preds" / f"{proto}_{target}_fold{k}.csv"
                if f.exists() and not args.overwrite:
                    print(f"[skip] {f.name} exists")
                    continue
                print(f"[fold {k}] protocol={proto} target={target} "
                      f"train={len(train_df)} test={len(test_df)} held_out={held_out}")
                t0 = time.time()
                pred = train_and_predict(train_df, test_df, root, target, args,
                                         device, seed=args.seed + k)
                gt = test_df[TARGET_COL[target]].to_numpy()
                const = circular_mean(train_df[TARGET_COL[target]].to_numpy())
                res = test_df[["file", "scene", "cam", "ray", "depth"]].copy()
                res["pred"] = pred
                res["gt"] = gt
                res["err_deg"] = ang_err_deg(pred, gt)
                res["const_err_deg"] = ang_err_deg(np.full_like(gt, const), gt)
                res.to_csv(f, index=False)
                print(f"      mean err={res['err_deg'].mean():.1f}°  "
                      f"const baseline={res['const_err_deg'].mean():.1f}°  "
                      f"({time.time() - t0:.0f}s)")
    print("\nDone. Next: python pds_exp0.py report --crops ... --results ...")


# --------------------------------------------------------------------------- #
# Step 3: report / 產出報告
# --------------------------------------------------------------------------- #
def load_preds(results: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((results / "preds").glob("*.csv")):
        proto, target, fold = f.stem.split("_")
        d = pd.read_csv(f)
        d["protocol"], d["target"], d["fold"] = proto, target, int(fold.replace("fold", ""))
        frames.append(d)
    if not frames:
        raise SystemExit("No prediction files found. Run `run` first.")
    return pd.concat(frames, ignore_index=True)


def fmt(m: float, s: float) -> str:
    return f"{m:.1f} ± {s:.1f}" if not np.isnan(s) else f"{m:.1f}"


def cmd_report(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = Path(args.results)
    p = load_preds(results)
    p["abs_ray_deg"] = np.degrees(np.abs(p["ray"]))
    p["acc30"] = (p["err_deg"] <= 30).astype(float)

    lines = ["# PDS Experiment 0 — Results / 結果", ""]
    lines += [f"Predictions: {len(p)} rows, folds: {p['fold'].nunique()}", ""]

    # Table 1: overall per protocol x target (mean ± std across folds)
    lines += ["## 1. Overall heading error / 整體朝向誤差 (deg, mean ± std across folds)", "",
              "| protocol | target | mean err | median err | acc@30° | const baseline |",
              "|---|---|---|---|---|---|"]
    per_fold = (p.groupby(["protocol", "target", "fold"])
                .agg(mean=("err_deg", "mean"), med=("err_deg", "median"),
                     acc=("acc30", "mean"), const=("const_err_deg", "mean"))
                .reset_index())
    for (proto, tgt), g in per_fold.groupby(["protocol", "target"]):
        lines.append(
            f"| {proto} | {tgt} | {fmt(g['mean'].mean(), g['mean'].std())} | "
            f"{fmt(g['med'].mean(), g['med'].std())} | "
            f"{fmt(100 * g['acc'].mean(), 100 * g['acc'].std())}% | "
            f"{g['const'].mean():.1f} |")
    lines.append("")

    # Table 2: cross-camera, size-matched / 跨相機（資料量相同）
    for proto, title in [("front", "trained on CAM_FRONT only / 只用前相機訓練"),
                         ("allsub", "all cameras, same data size as front / 所有相機、與 front 同資料量")]:
        fr = p[p["protocol"] == proto]
        if not len(fr):
            continue
        lines += [f"## 2{'a' if proto == 'front' else 'b'}. Cross-camera: {title}", "",
                  "| camera | " + " | ".join(sorted(fr["target"].unique())) + " |",
                  "|---|" + "---|" * fr["target"].nunique()]
        tab = fr.groupby(["cam", "target"])["err_deg"].mean().unstack()
        for cam in CAMS:
            if cam in tab.index:
                lines.append(f"| {cam} | " + " | ".join(f"{v:.1f}" for v in tab.loc[cam]) + " |")
        lines.append("")
    if {"front", "allsub"} <= set(p["protocol"]):
        lines += ["Viewpoint effect = front − allsub (same data size). Data-size effect = allsub − all.",
                  "視角效應 = front − allsub（資料量相同）。資料量效應 = allsub − all。", ""]

    # Table 3: error vs |ray angle| (all protocol)
    al = p[p["protocol"] == "all"].copy()
    if len(al):
        al["ray_bin"] = pd.cut(al["abs_ray_deg"], RAY_BINS_DEG, right=False)
        tab = al.groupby(["ray_bin", "target"], observed=True)["err_deg"].mean().unstack()
        cnt = al[al["target"] == al["target"].iloc[0]].groupby("ray_bin", observed=True).size()
        lines += ["## 3. Error vs. off-center angle |ray| / 誤差 vs. 偏離中心角度", "",
                  "| abs(ray) bin | n | " + " | ".join(tab.columns) + " |",
                  "|---|---|" + "---|" * len(tab.columns)]
        for b, row in tab.iterrows():
            lines.append(f"| {b} | {int(cnt.get(b, 0))} | " +
                         " | ".join(f"{v:.1f}" for v in row) + " |")
        lines.append("")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        mids = [(b.left + b.right) / 2 for b in tab.index]
        for tgt in tab.columns:
            ax.plot(mids, tab[tgt].values, marker="o", label=tgt)
        ax.set_xlabel("|ray angle| (deg) — distance from image center")
        ax.set_ylabel("mean heading error (deg)")
        ax.set_title("PDS Exp 0: error vs. viewing ray (protocol=all)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results / "err_vs_ray.png", dpi=150)
        plt.close(fig)
        lines += ["![error vs ray](err_vs_ray.png)", ""]

    (results / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {results / 'summary.md'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="PDS Experiment 0")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract object crops from nuScenes")
    e.add_argument("--dataroot", required=True)
    e.add_argument("--version", default="v1.0-mini")
    e.add_argument("--out", default="./crops")
    e.add_argument("--category", default="vehicle.car")
    e.add_argument("--min-size", type=int, default=32, help="min box side in pixels")
    e.add_argument("--min-lidar-pts", type=int, default=5)
    e.add_argument("--size", type=int, default=128, help="saved crop size")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("run", help="train/evaluate all protocols x targets x folds")
    r.add_argument("--crops", default="./crops")
    r.add_argument("--out", default="./results")
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--epochs", type=int, default=12)
    r.add_argument("--bs", type=int, default=64)
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--size", type=int, default=128)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--protocols", nargs="*", choices=PROTOCOLS)
    r.add_argument("--targets", nargs="*", choices=TARGETS)
    r.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="aggregate predictions into summary.md + plot")
    p.add_argument("--crops", default="./crops")
    p.add_argument("--results", default="./results")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
