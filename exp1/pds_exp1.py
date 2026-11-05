#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS Experiment 1 — Does the label frame matter under camera tilt?
PDS 實驗 1 —— 攝影機傾斜時，標籤座標系重要嗎？

Question / 問題
    A heading model is trained on level cameras (nuScenes). At test time the camera
    pitches or rolls and its extrinsics are UNKNOWN. How much heading accuracy does
    each label frame keep?
    朝向模型在水平相機（nuScenes）上訓練。測試時攝影機俯仰或側傾，且外參「未知」。
    每種標籤座標系能保留多少朝向準確度？

Label frames / 標籤座標系
    world : yaw in the global annotation frame                     世界座標 yaw
    cam   : yaw in the (possibly tilted) camera frame              相機座標 yaw
    alpha : cam yaw minus the horizontal angle of the pixel ray     觀測角（像素視線）
    allo  : full 3D rotation relative to the pixel-ray frame (6D)   完整 allocentric 旋轉（6D）

Decoders (how a prediction becomes a heading) / 解碼方式
    none      : extrinsics unknown, camera assumed level            外參未知，假設相機水平
    grav      : allo only — the object's own up-axis estimates gravity  僅 allo：以物體上方向估計重力
    ext       : exact extrinsics known (upper bound, RQ3)            外參精確已知（上限，RQ3）
    ext_noisy : extrinsics known with N(0, 2 deg) error (RQ3)        外參含 2° 高斯誤差（RQ3）

Protocols / 協定
    level : train on level crops only                                只用水平裁切訓練
    aug   : train on level + randomly tilted crops (|pitch|<=15, |roll|<=20 deg)  水平 + 隨機傾斜裁切

Camera tilt is simulated EXACTLY: a pure camera rotation R is the image warp
H = K R K^-1, independent of scene depth; 3D labels rotate with the same R.
攝影機傾斜以精確方式模擬：純旋轉 R 對應影像變形 H = K R K^-1，與場景深度無關；
3D 標籤以同一個 R 旋轉。

Usage / 用法
    python pds_exp1.py selftest
    python pds_exp1.py prepare --dataroot ../exp0/nuscenes --out data
    python pds_exp1.py render  --data data
    python pds_exp1.py run     --data data --out results
    python pds_exp1.py derot   --data data --out results      (A1 mechanism check / 機制檢驗)
    python pds_exp1.py derotnoise --data data --out results   (A1b noisy roll / 含誤差側傾)
    python pds_exp1.py report  --results results
    A2 (second backbone / 第二種骨幹): add --backbone swin_t --out results_swin_t to run/derot/derotnoise
    A2b (LR control / 學習率對照): --backbone resnet18 --lr 1e-4 --out results_resnet18_lr1e-4
    A1c (corner fill / 角落填補): derot --fill reflect (reuses checkpoints; writes derot_reflect/)
    A1d (noisy roll, mirrored corners / 含誤差＋鏡射): derotnoise --fill reflect (writes derot_noise_reflect/)

Python 3.10; needs torch, torchvision, numpy, pandas, pillow, opencv-python, matplotlib
(+ nuscenes-devkit for `prepare`).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FRAMES = ["world", "cam", "alpha", "allo"]
PROTOCOLS = ["level", "aug"]
CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
# nuScenes cameras see only ~39 deg vertically (fy~1266 px, 900 px tall): beyond ~15 deg of
# pitch, cars near the horizon leave the frame. Roll keeps them in view up to 20 deg.
# nuScenes 相機垂直視野僅約 39°：俯仰超過約 15° 時，地平線附近的車會離開畫面；側傾到 20° 仍在畫面內。
PITCH_FULL = [0, 1, 2, 3, 5, 8, 12, 15]
ROLL_FULL = [0, 1, 2, 3, 5, 8, 12, 16, 20]
PITCH_SMALL = [0, 5, 12]
ROLL_SMALL = [0, 5, 20]
AUG_MAX_PITCH, AUG_MAX_ROLL = 15.0, 20.0
EXT_NOISE_DEG = 2.0


# =========================================================================== #
# Geometry (numpy, vectorised) / 幾何工具（numpy 向量化）
# Camera frame: x right, y down, z forward.  Box frame: x forward, y left, z up.
# 相機座標：x 右、y 下、z 前。物體框座標：x 前、y 左、z 上。
# =========================================================================== #
def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def tilt_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rotation applied to camera-frame points when the camera tilts.
    相機傾斜時作用在相機座標點上的旋轉。"""
    return rot_z(math.radians(roll_deg)) @ rot_x(math.radians(pitch_deg))


def tilt_matrices(pitch_deg: np.ndarray, roll_deg: np.ndarray) -> np.ndarray:
    return np.stack([tilt_matrix(p, r) for p, r in zip(pitch_deg, roll_deg)])


def yaw_of(h: np.ndarray) -> np.ndarray:
    """Camera-frame yaw of heading vector(s) (..., 3). 航向向量的相機座標 yaw。"""
    return np.arctan2(h[..., 0], h[..., 2])


def ray_rotation(d: np.ndarray) -> np.ndarray:
    """Minimal rotation(s) taking +z to unit ray(s) d, shape (N,3) -> (N,3,3).
    把 +z 轉到視線 d 的最小旋轉。"""
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    n = d.shape[0]
    v = np.stack([-d[:, 1], d[:, 0], np.zeros(n)], axis=-1)  # e_z x d
    s2 = (v ** 2).sum(-1)
    c = d[:, 2]
    vx = np.zeros((n, 3, 3))
    vx[:, 0, 1], vx[:, 0, 2] = -v[:, 2], v[:, 1]
    vx[:, 1, 0], vx[:, 1, 2] = v[:, 2], -v[:, 0]
    vx[:, 2, 0], vx[:, 2, 1] = -v[:, 1], v[:, 0]
    k = np.where(s2 > 1e-12, (1.0 - c) / np.maximum(s2, 1e-12), 0.5)
    return np.eye(3)[None] + vx + (vx @ vx) * k[:, None, None]


def r6_from_R(R: np.ndarray) -> np.ndarray:
    """First two columns -> 6D. 取前兩欄作為 6D 表示。"""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def R_from_r6(x: np.ndarray) -> np.ndarray:
    """Gram-Schmidt 6D -> rotation (N,3,3). 6D 經 Gram-Schmidt 轉回旋轉矩陣。"""
    a1, a2 = x[:, :3], x[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def gravity_yaw(O: np.ndarray) -> np.ndarray:
    """Yaw measured in a gravity frame estimated from the object's own up-axis.
    以物體自身的上方向估計重力座標系，並在其中量測 yaw。"""
    down = -O[:, :, 2]
    down = down / np.linalg.norm(down, axis=-1, keepdims=True)
    ez = np.array([0.0, 0.0, 1.0])
    zg = ez[None] - (down @ ez)[:, None] * down
    zg = zg / np.linalg.norm(zg, axis=-1, keepdims=True)
    xg = np.cross(down, zg)
    h = O[:, :, 0]
    return np.arctan2((h * xg).sum(-1), (h * zg).sum(-1))


def pixel_rays(u: np.ndarray, v: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def box_corners(c: np.ndarray, O: np.ndarray, wlh: np.ndarray) -> np.ndarray:
    """3x8 corners in camera frame (nuScenes convention). 相機座標下的 8 個角點。"""
    w, l, h = wlh
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return O @ np.vstack([x, y, z]) + c[:, None]


def labels_for(R: np.ndarray, O: np.ndarray, d_pix: np.ndarray) -> dict:
    """Labels of each frame for tilted rotation R@O seen along pixel ray d_pix.
    在傾斜 R 下、沿像素視線 d_pix 觀察時，各座標系的標籤。"""
    Ot = R @ O
    cam = float(yaw_of(Ot[:, 0]))
    ray = float(math.atan2(d_pix[0], d_pix[2]))
    alpha = float(wrap(cam - ray))
    allo = ray_rotation(d_pix[None])[0].T @ Ot
    return {"cam_yaw": cam, "alpha": alpha, **{f"a{i}": float(x) for i, x in enumerate(r6_from_R(allo))}}


# =========================================================================== #
# Decoding predictions to a level-frame heading / 把預測解碼成水平座標系的朝向
# =========================================================================== #
def decode(frame: str, pred: np.ndarray, df: pd.DataFrame, mode: str,
           R_known: np.ndarray | None = None) -> np.ndarray:
    """pred: (N,2) sin/cos for yaw frames or (N,6) for allo. Returns yaw (N,).
    For world: returns world yaw. Others: yaw in the LEVEL camera frame.
    world 回傳世界 yaw；其餘回傳「水平」相機座標系的 yaw。"""
    d = df[["dx", "dy", "dz"]].to_numpy()
    if frame == "world":
        return np.arctan2(pred[:, 0], pred[:, 1])
    if frame in ("cam", "alpha"):
        y = np.arctan2(pred[:, 0], pred[:, 1])
        if frame == "alpha":
            y = wrap(y + np.arctan2(d[:, 0], d[:, 2]))
        if mode == "none":
            return y
        return yaw_from_tilted_yaw(y, R_known)
    # allo
    O = ray_rotation(d) @ R_from_r6(pred)
    if mode == "none":
        return yaw_of(O[:, :, 0])
    if mode == "grav":
        return gravity_yaw(O)
    return yaw_of(np.einsum("nji,njk->nik", R_known, O)[:, :, 0])  # (R^T O) x-axis


def yaw_from_tilted_yaw(y_t: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Level heading whose yaw in the tilted camera equals y_t (exact, extrinsics known).
    A tilted-frame yaw only fixes a plane of directions; intersect it with the ground plane.
    找出在傾斜相機中 yaw 等於 y_t 的水平航向（外參已知時的精確解）。
    傾斜座標的 yaw 只決定一個方向平面；將它與地面相交。"""
    n_t = np.stack([np.cos(y_t), np.zeros_like(y_t), -np.sin(y_t)], axis=-1)  # plane normal, tilted frame
    m = np.einsum("nji,nj->ni", R, n_t)                                       # R^T n_t, level frame
    th = np.arctan2(-m[:, 2], m[:, 0])
    h = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=-1)
    flip = np.abs(wrap(yaw_of(np.einsum("nij,nj->ni", R, h)) - y_t)) > np.pi / 2
    return wrap(th + np.pi * flip)


def decoders_for(frame: str) -> list[str]:
    if frame == "world":
        return ["none"]
    if frame == "allo":
        return ["none", "grav", "ext", "ext_noisy"]
    return ["none", "ext", "ext_noisy"]


# =========================================================================== #
# Stage 1: prepare — nuScenes -> geometry table / 從 nuScenes 建立幾何資料表
# =========================================================================== #
def cmd_prepare(args: argparse.Namespace) -> None:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    root = Path(args.dataroot).resolve()
    rows = []
    for si, sample in enumerate(nusc.sample):
        scene = nusc.get("scene", sample["scene_token"])["name"]
        for cam in CAMS:
            img_path, boxes, K = nusc.get_sample_data(sample["data"][cam], box_vis_level=BoxVisibility.ALL)
            K = np.asarray(K)
            for box in boxes:
                if not box.name.startswith(args.category):
                    continue
                ann = nusc.get("sample_annotation", box.token)
                if ann["visibility_token"] not in ("3", "4") or ann["num_lidar_pts"] < args.min_lidar_pts:
                    continue
                if box.center[2] <= 0.5:
                    continue
                cs = view_points(box.corners(), K, normalize=True)[:2]
                if max(cs[0].ptp(), cs[1].ptp()) < args.min_size:
                    continue
                O = box.orientation.rotation_matrix
                rows.append({
                    "uid": f"{box.token}_{cam}", "scene": scene, "cam": cam,
                    "img": str(Path(img_path).resolve().relative_to(root)).replace("\\", "/"),
                    "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                    "c0": box.center[0], "c1": box.center[1], "c2": box.center[2],
                    **{f"o{i}": float(x) for i, x in enumerate(O.reshape(-1))},
                    "w": box.wlh[0], "l": box.wlh[1], "h": box.wlh[2],
                    "world_yaw": float(nusc.get_box(box.token).orientation.yaw_pitch_roll[0]),
                })
        if (si + 1) % 50 == 0:
            print(f"  samples {si + 1}/{len(nusc.sample)}  objects={len(rows)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(out / "geometry.csv", index=False)
    (out / "dataroot.txt").write_text(str(root), encoding="utf-8")
    print(f"Saved {len(meta)} objects -> {out / 'geometry.csv'}")


# =========================================================================== #
# Stage 2: render — exact tilt warps / 以精確變形產生傾斜裁切
# =========================================================================== #
def conditions(pitch_grid: list[int], roll_grid: list[int]) -> list[tuple[float, float]]:
    """Pitch-only and roll-only sweeps sharing the level point. 俯仰與側傾各自掃描。"""
    c = [(0.0, 0.0)]
    for a in pitch_grid[1:]:
        c += [(float(-a), 0.0), (float(a), 0.0)]
    for a in roll_grid[1:]:
        c += [(0.0, float(-a)), (0.0, float(a))]
    return c


def cond_name(p: float, r: float) -> str:
    return f"p{p:+.0f}_r{r:+.0f}"


def render_one(img: np.ndarray, row: pd.Series, R: np.ndarray, size: int, min_inside: float):
    """Returns (crop, info) or (None, None) if the object leaves the view.
    回傳裁切與資訊；物體離開畫面則回傳 None。"""
    import cv2

    H_img, W_img = img.shape[:2]
    K = np.array([[row.fx, 0, row.cx], [0, row.fy, row.cy], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)
    c = np.array([row.c0, row.c1, row.c2])
    O = np.array([row[f"o{i}"] for i in range(9)]).reshape(3, 3)
    corners = R @ box_corners(c, O, np.array([row.w, row.l, row.h]))
    if corners[2].min() <= 0.3:
        return None, None
    uv = (K @ corners)[:2] / corners[2]
    x0, y0 = uv.min(1)
    x1, y1 = uv.max(1)
    cu, cv_ = (x0 + x1) / 2, (y0 + y1) / 2
    if not (0 <= cu < W_img and 0 <= cv_ < H_img):
        return None, None
    s = max(x1 - x0, y1 - y0) * 1.15 / 2
    g = (np.arange(size) + 0.5) / size * 2 * s - s
    gu, gv = np.meshgrid(cu + g, cv_ + g)
    pts = np.stack([gu.ravel(), gv.ravel(), np.ones(gu.size)])
    src = K @ R.T @ Kinv @ pts                       # tilted pixel -> original pixel
    mx = (src[0] / src[2]).reshape(size, size).astype(np.float32)
    my = (src[1] / src[2]).reshape(size, size).astype(np.float32)
    inside = ((mx >= 0) & (mx < W_img) & (my >= 0) & (my < H_img)).mean()
    if inside < min_inside:
        return None, None
    crop = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    d_pix = pixel_rays(np.array([cu]), np.array([cv_]), row.fx, row.fy, row.cx, row.cy)[0]
    info = {"u": cu, "v": cv_, "dx": d_pix[0], "dy": d_pix[1], "dz": d_pix[2],
            "level_yaw": float(yaw_of(O[:, 0])), **labels_for(R, O, d_pix)}
    return crop, info


def cmd_render(args: argparse.Namespace) -> None:
    import cv2

    data = Path(args.data)
    geo = pd.read_csv(data / "geometry.csv")
    dataroot = Path(args.dataroot or (data / "dataroot.txt").read_text(encoding="utf-8").strip())
    conds = conditions(PITCH_SMALL, ROLL_SMALL) if args.grid == "small" else conditions(PITCH_FULL, ROLL_FULL)
    rng = np.random.default_rng(args.seed)
    (data / "crops").mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    groups = list(geo.groupby("img"))
    for gi, (img_rel, g) in enumerate(groups):
        img = cv2.imread(str(dataroot / img_rel))
        if img is None:
            raise FileNotFoundError(dataroot / img_rel)
        for _, row in g.iterrows():
            jobs = [("test", p, r) for p, r in conds]
            jobs += [("aug", rng.uniform(-AUG_MAX_PITCH, AUG_MAX_PITCH), rng.uniform(-AUG_MAX_ROLL, AUG_MAX_ROLL))
                     for _ in range(args.aug_copies)]
            for k, (split, p, r) in enumerate(jobs):
                crop, info = render_one(img, row, tilt_matrix(p, r), args.size, args.min_inside)
                if crop is None:
                    continue
                name = f"{row.uid}_{split}{k}.jpg"
                cv2.imwrite(str(data / "crops" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append({"uid": row.uid, "scene": row.scene, "cam": row.cam, "split": split,
                             "cond": cond_name(p, r) if split == "test" else "aug",
                             "pitch": p, "roll": r, "file": name, "world_yaw": row.world_yaw, **info})
        if (gi + 1) % 200 == 0:
            print(f"  images {gi + 1}/{len(groups)}  crops={len(rows)}  {time.time() - t0:.0f}s")
    idx = pd.DataFrame(rows)
    test = idx[idx.split == "test"]
    ok = test.groupby("uid").cond.nunique()
    keep = set(ok[ok == len(conds)].index)                  # visible in EVERY condition
    idx["complete"] = idx.uid.isin(keep)
    idx.to_csv(data / "index.csv", index=False)
    (data / "conditions.json").write_text(json.dumps(conds), encoding="utf-8")
    print(f"\nRendered {len(idx)} crops; {len(keep)} objects visible in all {len(conds)} conditions.")


# =========================================================================== #
# Stage 3: train + evaluate / 訓練與評估
# =========================================================================== #
class CropDS:
    """Module-level for Windows DataLoader workers. 定義在模組層級以支援 Windows 多程序。"""

    def __init__(self, df: pd.DataFrame, root: Path, frame: str, train: bool, size: int,
                 rotate_deg: np.ndarray | None = None, fill: str = "black"):
        from torchvision import transforms as T

        # rotate_deg: counter-clockwise image rotation per crop (PIL convention). Rotating by +roll
        # undoes a camera roll (verified on synthetic crops). 每個裁切的逆時針旋轉角；+roll 可抵銷相機側傾。
        self.rot = None if rotate_deg is None else np.asarray(rotate_deg, dtype=np.float64)
        # fill: what the corners uncovered by the rotation show. "black" (PIL default) or "reflect"
        # (mirrored image content, OpenCV BORDER_REFLECT_101). 旋轉後露出的角落：黑色或鏡射內容。
        self.fill = fill

        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.paths = [root / "crops" / f for f in df["file"]]
        if frame == "allo":
            self.y = df[[f"a{i}" for i in range(6)]].to_numpy(np.float32)
        else:
            col = {"world": "world_yaw", "cam": "cam_yaw", "alpha": "alpha"}[frame]
            a = df[col].to_numpy(np.float64)
            self.y = np.stack([np.sin(a), np.cos(a)], axis=-1).astype(np.float32)
        # No flips / rotations / translations: they would change the true label.
        # 不做翻轉、旋轉、平移：它們會改變真實標籤。
        aug = [T.RandomAffine(degrees=0, scale=(0.9, 1.1)), T.ColorJitter(0.2, 0.2, 0.2)] if train else []
        self.tf = T.Compose([T.Resize((size, size)), *aug, T.ToTensor(), norm])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        img = Image.open(self.paths[i]).convert("RGB")
        if self.rot is not None and self.rot[i] != 0:
            img = rotate_crop(img, float(self.rot[i]), self.fill)
        return self.tf(img), torch.from_numpy(self.y[i]), i


def torch_R_from_r6(x):
    import torch
    import torch.nn.functional as F

    b1 = F.normalize(x[:, :3], dim=1)
    a2 = x[:, 3:6] - (b1 * x[:, 3:6]).sum(1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=1)], dim=-1)


def loss_fn(frame: str, out, y):
    import torch.nn.functional as F

    if frame == "allo":
        return ((torch_R_from_r6(out) - torch_R_from_r6(y)) ** 2).sum((1, 2)).mean()
    p = F.normalize(out, dim=1)
    return (1.0 - (p * y).sum(1)).mean()


def make_folds(scenes: list[str], n_folds: int, seed: int = 0) -> list[list[str]]:
    """Identical to Exp 0, so folds match. 與實驗 0 相同，確保切分一致。"""
    scenes = sorted(scenes)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_folds = min(n_folds, len(scenes))
    return [scenes[i::n_folds] for i in range(n_folds)]


BACKBONES = ["resnet18", "swin_t", "convnext_tiny"]


def build_backbone(name: str, out_dim: int, pretrained: bool):
    """ResNet-18 (CNN, default), Swin-T (vision transformer) or ConvNeXt-Tiny (modern CNN).
    ResNet-18（CNN，預設）、Swin-T（視覺 Transformer）或 ConvNeXt-Tiny（新式 CNN）。"""
    import torch.nn as nn
    import torchvision.models as tvm

    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, out_dim)
    elif name == "swin_t":
        m = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        m.head = nn.Linear(m.head.in_features, out_dim)
    elif name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, out_dim)
    else:
        raise ValueError(name)
    return m


def train_model(train_df, root, frame, args, device, seed):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet18_Weights, resnet18

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = build_backbone(getattr(args, "backbone", "resnet18"), 6 if frame == "allo" else 2, args.pretrained)
    m = m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dl = DataLoader(CropDS(train_df, root, frame, True, args.size), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=device.type == "cuda",
                    drop_last=len(train_df) > args.bs, persistent_workers=args.workers > 0)
    for ep in range(args.epochs):
        m.train()
        tot = n = 0
        for x, y, _ in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = loss_fn(frame, m(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x); n += len(x)
        sched.step()
        if args.verbose:
            print(f"      epoch {ep + 1}/{args.epochs} loss={tot / max(n, 1):.4f}")
    return m


def rotate_crop(img, angle: float, fill: str = "black"):
    """Rotate a PIL image counter-clockwise by `angle` degrees about its centre.
    "black" uses PIL (corners become black); "reflect" uses OpenCV with mirrored borders. Both use the
    same centre and direction (checked: interior pixels agree to < 1 grey level).
    逆時針旋轉影像；black 角落為黑色，reflect 以鏡射內容填補。兩者中心與方向一致（已驗證）。"""
    from PIL import Image

    if fill == "black":
        return img.rotate(angle, resample=Image.BILINEAR)
    if fill == "reflect":
        import cv2

        a = np.asarray(img)
        h, w = a.shape[:2]
        M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
        return Image.fromarray(cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REFLECT_101))
    raise ValueError(fill)


def predict(m, df, root, frame, args, device, rotate_deg=None, fill="black") -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(CropDS(df, root, frame, False, args.size, rotate_deg, fill), batch_size=args.bs * 2,
                    shuffle=False, num_workers=args.workers)
    out = np.zeros((len(df), 6 if frame == "allo" else 2))
    m.eval()
    with torch.no_grad():
        for x, _, i in dl:
            out[i.numpy()] = m(x.to(device)).float().cpu().numpy()
    return out


def evaluate(frame, pred, test_df, rng) -> pd.DataFrame:
    R = tilt_matrices(test_df.pitch.to_numpy(), test_df.roll.to_numpy())
    noise = np.radians(rng.normal(0, EXT_NOISE_DEG, (len(test_df), 2)))
    R_noisy = tilt_matrices(test_df.pitch.to_numpy() + np.degrees(noise[:, 0]),
                            test_df.roll.to_numpy() + np.degrees(noise[:, 1]))
    gt = test_df.world_yaw.to_numpy() if frame == "world" else test_df.level_yaw.to_numpy()
    res = []
    for mode in decoders_for(frame):
        Rk = {"ext": R, "ext_noisy": R_noisy}.get(mode)
        yaw = decode(frame, pred, test_df, "ext" if mode == "ext_noisy" else mode, Rk)
        err = np.degrees(np.abs(wrap(yaw - gt)))
        res.append(pd.DataFrame({"uid": test_df.uid.values, "cond": test_df.cond.values,
                                 "pitch": test_df.pitch.values, "roll": test_df.roll.values,
                                 "decoder": mode, "err_deg": err.round(3)}))
    return pd.concat(res, ignore_index=True)


def cmd_run(args: argparse.Namespace) -> None:
    import torch

    root, out = Path(args.data), Path(args.out)
    (out / "errors").mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(root / "index.csv")
    idx = idx[idx.complete].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}   backbone: {args.backbone}   objects: {idx.uid.nunique()}   crops: {len(idx)}")
    (out / "backbone.txt").write_text(f"{args.backbone}, lr {args.lr:g}", encoding="utf-8")
    folds = make_folds(idx.scene.unique().tolist(), args.folds)
    if args.max_folds:
        folds = folds[:args.max_folds]
    (out / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    for k, held in enumerate(folds):
        tr_all = idx[~idx.scene.isin(held)]
        test_df = idx[idx.scene.isin(held) & (idx.split == "test")].reset_index(drop=True)
        for proto in args.protocols or PROTOCOLS:
            tr = tr_all[tr_all.cond == "p+0_r+0"] if proto == "level" else \
                tr_all[(tr_all.cond == "p+0_r+0") | (tr_all.split == "aug")]
            tr = tr.reset_index(drop=True)
            for frame in args.frames or FRAMES:
                f = out / "errors" / f"{proto}_{frame}_fold{k}.csv.gz"
                if f.exists() and not args.overwrite:
                    print(f"[skip] {f.name}")
                    continue
                t0 = time.time()
                print(f"[fold {k}] {proto:5s} {frame:5s} train={len(tr)} test={len(test_df)}")
                m = train_model(tr, root, frame, args, device, args.seed + k)
                (out / "ckpt").mkdir(exist_ok=True)
                torch.save(m.state_dict(), out / "ckpt" / f"{proto}_{frame}_fold{k}.pt")
                pred = predict(m, test_df, root, frame, args, device)
                ev = evaluate(frame, pred, test_df, np.random.default_rng(args.seed + k))
                ev.insert(0, "fold", k); ev.insert(0, "frame", frame); ev.insert(0, "protocol", proto)
                ev.to_csv(f, index=False)
                lvl = ev[(ev.cond == "p+0_r+0") & (ev.decoder == "none")].err_deg.mean()
                worst = ev[ev.decoder == "none"].groupby("cond").err_deg.mean().max()
                print(f"      level err={lvl:.1f}°  worst-tilt err={worst:.1f}°  ({time.time() - t0:.0f}s)")
    print("\nDone. Next: python pds_exp1.py report --results", out)


def load_or_train(path: Path, tr, root, frame, args, device, seed):
    """Reuse a saved checkpoint, else train and save it. 有檢查點就重用，否則訓練並儲存。"""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    if path.exists():
        m = build_backbone(getattr(args, "backbone", "resnet18"), 6 if frame == "allo" else 2, False)
        m.load_state_dict(torch.load(path, map_location=device))
        return m.to(device)
    m = train_model(tr, root, frame, args, device, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), path)
    return m


def cmd_derot(args: argparse.Namespace) -> None:
    """A1 mechanism check: level-trained models on roll-tilted crops, with and without rotating
    each crop back upright by the TRUE roll (an oracle). If de-rotation restores the level error,
    in-plane image rotation is the cause of the roll damage.
    A1 機制檢驗：水平訓練的模型在側傾裁切上推論，比較「原樣」與「依真實側傾轉回直立」（神諭）。
    若轉正後誤差回到水平值，即證實側傾損害來自影像平面內旋轉。"""
    import torch

    root, out = Path(args.data), Path(args.out)
    fill = getattr(args, "fill", "black")
    ddir = out / ("derot" if fill == "black" else f"derot_{fill}")
    ddir.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(root / "index.csv")
    idx = idx[idx.complete].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    folds = make_folds(idx.scene.unique().tolist(), args.folds)
    if args.max_folds:
        folds = folds[:args.max_folds]
    for k, held in enumerate(folds):
        tr = idx[~idx.scene.isin(held) & (idx.cond == "p+0_r+0")].reset_index(drop=True)
        te = idx[idx.scene.isin(held) & (idx.split == "test") & (idx.pitch == 0)].reset_index(drop=True)
        for frame in args.frames or ["cam", "alpha", "allo"]:
            f = ddir / f"{frame}_fold{k}.csv.gz"
            if f.exists() and not args.overwrite:
                print(f"[skip] {f.name}")
                continue
            t0 = time.time()
            m = load_or_train(out / "ckpt" / f"level_{frame}_fold{k}.pt", tr, root, frame, args, device, args.seed + k)
            gt = te.level_yaw.to_numpy()
            parts = []
            for mode, rot in [("plain", None), ("derot", te.roll.to_numpy())]:
                pred = predict(m, te, root, frame, args, device, rot, fill)
                if mode == "derot":
                    # The upright crop now matches the level labels, so decode against the level ray.
                    # 轉正後的裁切對應水平標籤，因此以水平視線解碼。
                    lv = idx[idx.scene.isin(held) & (idx.cond == "p+0_r+0")].set_index("uid")
                    dfd = te[["uid"]].join(lv[["dx", "dy", "dz"]], on="uid")
                else:
                    dfd = te
                yaw = decode(frame, pred, dfd, "none")
                parts.append(pd.DataFrame({"uid": te.uid, "roll": te.roll, "mode": mode,
                                           "err_deg": np.degrees(np.abs(wrap(yaw - gt))).round(3)}))
            r = pd.concat(parts, ignore_index=True)
            r.insert(0, "fold", k); r.insert(0, "frame", frame)
            r.to_csv(f, index=False)
            g = r.groupby(["mode", "roll"]).err_deg.mean()
            print(f"[fold {k}] {frame:5s} fill={fill}  level={g[('plain', 0.0)]:.1f}  roll±20 plain="
                  f"{(g[('plain', 20.0)] + g[('plain', -20.0)]) / 2:.1f}  derot="
                  f"{(g[('derot', 20.0)] + g[('derot', -20.0)]) / 2:.1f}  ({time.time() - t0:.0f}s)")
    print("\nDone. Next: python pds_exp1.py report --results", out)


def cmd_derotnoise(args: argparse.Namespace) -> None:
    """A1b: de-rotate with a NOISY roll estimate (roll + N(0, sigma)), as an IMU or horizon
    estimate would give. Level crops (roll 0) are also rotated by the noise alone, since a
    real system would apply its (noisy) correction to every frame.
    A1b：以含誤差的側傾估計（真實側傾 + 常態誤差）轉正，模擬 IMU 或地平線估計。水平裁切也只依誤差旋轉，
    因為實際系統會對每一張影像套用（含誤差的）修正。"""
    import torch

    root, out = Path(args.data), Path(args.out)
    fill = getattr(args, "fill", "black")
    ndir = out / ("derot_noise" if fill == "black" else f"derot_noise_{fill}")
    ndir.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(root / "index.csv")
    idx = idx[idx.complete].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    folds = make_folds(idx.scene.unique().tolist(), args.folds)
    if args.max_folds:
        folds = folds[:args.max_folds]
    for k, held in enumerate(folds):
        tr = idx[~idx.scene.isin(held) & (idx.cond == "p+0_r+0")].reset_index(drop=True)
        te = idx[idx.scene.isin(held) & (idx.split == "test") & (idx.pitch == 0)].reset_index(drop=True)
        lv = idx[idx.scene.isin(held) & (idx.cond == "p+0_r+0")].set_index("uid")
        dfd = te[["uid"]].join(lv[["dx", "dy", "dz"]], on="uid")
        gt = te.level_yaw.to_numpy()
        for frame in args.frames or ["cam", "alpha", "allo"]:
            f = ndir / f"{frame}_fold{k}.csv.gz"
            if f.exists() and not args.overwrite:
                print(f"[skip] {f.name}")
                continue
            t0 = time.time()
            m = load_or_train(out / "ckpt" / f"level_{frame}_fold{k}.pt", tr, root, frame, args, device, args.seed + k)
            parts = []
            for sigma in args.noise:
                rng = np.random.default_rng(1000 * k + int(10 * sigma))   # same noise for every frame
                rot = te.roll.to_numpy() + rng.normal(0.0, sigma, len(te))
                pred = predict(m, te, root, frame, args, device, rot, fill)
                yaw = decode(frame, pred, dfd, "none")
                parts.append(pd.DataFrame({"uid": te.uid, "roll": te.roll, "sigma": sigma,
                                           "err_deg": np.degrees(np.abs(wrap(yaw - gt))).round(3)}))
            r = pd.concat(parts, ignore_index=True)
            r.insert(0, "fold", k); r.insert(0, "frame", frame)
            r.to_csv(f, index=False)
            g = r.groupby(["sigma", "roll"]).err_deg.mean()
            msg = "  ".join(f"s={sg:g}: lvl={g[(sg, 0.0)]:.1f} r20={(g[(sg, 20.0)] + g[(sg, -20.0)]) / 2:.1f}"
                            for sg in args.noise)
            print(f"[fold {k}] {frame:5s} fill={fill}  {msg}  ({time.time() - t0:.0f}s)")
    print("\nDone. Next: python pds_exp1.py report --results", out)


# =========================================================================== #
# Stage 4: report / 報告
# =========================================================================== #
FRAME_STYLE = {  # validated categorical slots 1-3 + distinct markers (colour is never the only cue)
    "cam": ("#2a78d6", "o"), "alpha": ("#eb6834", "s"), "allo": ("#1baf7a", "^")}


def paired_delta(e: pd.DataFrame, conds: list[tuple[float, float]]) -> pd.Series:
    """Per-fold mean error at `conds` minus the SAME objects' level error.
    每折：同一批物體在 conds 的平均誤差減去水平誤差。"""
    f = e.groupby(["fold", "pitch", "roll"]).err_deg.mean()
    out = {}
    for k in sorted(e.fold.unique()):
        keys = [(k, float(p), float(r)) for p, r in conds]
        if (k, 0.0, 0.0) not in f.index or any(key not in f.index for key in keys):
            continue                                   # angle not rendered on this grid / 此格點未產生該角度
        out[k] = np.mean([f[key] for key in keys]) - f[(k, 0.0, 0.0)]
    return pd.Series(out, dtype=float)


def ms(x: pd.Series) -> str:
    if len(x) == 0:
        return "—"
    return f"{x.mean():+.1f} ± {x.std():.1f}" if len(x) > 1 else f"{x.mean():+.1f}"


def cmd_report(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = Path(args.results)
    files = sorted((res / "errors").glob("*.csv.gz"))
    if not files:
        raise SystemExit("No results yet. Run `run` first.")
    e = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    nf = e.fold.nunique()
    sym = lambda a, axis: [(a, 0.0), (-a, 0.0)] if axis == "pitch" else [(0.0, a), (0.0, -a)]
    bb = (res / "backbone.txt").read_text(encoding="utf-8").strip() if (res / "backbone.txt").exists() else "resnet18"
    L = [f"# PDS Experiment 1 — Label frame × camera tilt / 標籤座標系 × 攝影機傾斜 (backbone: {bb})", "",
         f"{nf} folds · {e.uid.nunique()} objects · every number below is PAIRED: error at tilt minus the "
         "same objects' error with a level camera, mean ± std over folds.",
         f"共 {nf} 折、{e.uid.nunique()} 個物體。下表皆為配對差：傾斜下誤差減去同一批物體在水平相機下的誤差（各折平均 ± 標準差）。", ""]

    # Table 1: degradation per frame (decoder = none)
    L += ["## 1. Degradation under tilt, extrinsics unknown / 傾斜造成的退化（外參未知）", "",
          "| protocol | frame | level err (°) | pitch ±5 | pitch ±15 | roll ±5 | roll ±12 | roll ±20 |",
          "|---|---|---|---|---|---|---|---|"]
    none = e[e.decoder == "none"]
    for proto in [p for p in PROTOCOLS if p in set(e.protocol)]:
        for fr in [f for f in FRAMES if f in set(e.frame)]:
            g = none[(none.protocol == proto) & (none.frame == fr)]
            lvl = g[(g.pitch == 0) & (g.roll == 0)].groupby("fold").err_deg.mean()
            cells = [ms(paired_delta(g, sym(a, ax))) for ax, a in
                     [("pitch", 5), ("pitch", 15), ("roll", 5), ("roll", 12), ("roll", 20)]]
            spread = f" ± {lvl.std():.1f}" if len(lvl) > 1 else ""
            L.append(f"| {proto} | {fr} | {lvl.mean():.1f}{spread} | " + " | ".join(cells) + " |")
    L += ["", "`world` is at chance (~90°) everywhere, so its deltas carry no information.",
          "`world` 在所有條件下都接近隨機（約 90°），其差值不具資訊量。", ""]

    # Table 2: paired frame differences at roll ±20, per fold
    L += ["## 2. Does the frame change roll robustness? / 座標系是否改變側傾穩健性？", "",
          "Per-fold difference in roll ±20° degradation (negative = first frame more robust).",
          "各折在側傾 ±20° 下退化量的差（負值代表前者較穩健）。", "",
          "| protocol | comparison | per fold | mean | folds favouring first |", "|---|---|---|---|---|"]
    for proto in [p for p in PROTOCOLS if p in set(e.protocol)]:
        d = {fr: paired_delta(none[(none.protocol == proto) & (none.frame == fr)], sym(20, "roll"))
             for fr in ("cam", "alpha", "allo") if fr in set(e.frame)}
        for a, b in [("alpha", "cam"), ("allo", "alpha"), ("allo", "cam")]:
            if a in d and b in d:
                diff = d[a] - d[b]
                L.append(f"| {proto} | {a} − {b} | " + ", ".join(f"{x:+.1f}" for x in diff) +
                         f" | {diff.mean():+.1f} | {(diff < 0).sum()}/{len(diff)} |")
    L.append("")

    # Table 3: decoders at roll ±20
    L += ["## 3. Does knowing the extrinsics help? / 已知外參是否有幫助？", "",
          "Roll ±20° degradation by decoder. / 各解碼方式在側傾 ±20° 的退化量。", "",
          "| protocol | frame | none | grav | ext | ext_noisy |", "|---|---|---|---|---|---|"]
    for proto in [p for p in PROTOCOLS if p in set(e.protocol)]:
        for fr in ("cam", "alpha", "allo"):
            row = []
            for dec in ("none", "grav", "ext", "ext_noisy"):
                g = e[(e.protocol == proto) & (e.frame == fr) & (e.decoder == dec)]
                row.append(ms(paired_delta(g, sym(20, "roll"))) if len(g) else "—")
            L.append(f"| {proto} | {fr} | " + " | ".join(row) + " |")
    L.append("")

    # Figure: degradation curves, rows = protocol, cols = pitch / roll
    protos = [p for p in PROTOCOLS if p in set(e.protocol)]
    fig, axes = plt.subplots(len(protos), 2, figsize=(10, 3.6 * len(protos)), sharey=True, squeeze=False)
    for i, proto in enumerate(protos):
        for j, axis in enumerate(["pitch", "roll"]):
            ax = axes[i][j]
            ends = []
            for fr, (col, mk) in FRAME_STYLE.items():
                g = none[(none.protocol == proto) & (none.frame == fr)]
                if not len(g):
                    continue
                other = "roll" if axis == "pitch" else "pitch"
                angles = sorted(g[g[other] == 0][axis].unique())
                mu, sd = [], []
                for a in angles:
                    d = paired_delta(g, [(a, 0.0)] if axis == "pitch" else [(0.0, a)])
                    mu.append(d.mean()); sd.append(d.std() if len(d) > 1 else 0.0)
                mu, sd = np.array(mu), np.array(sd)
                ax.fill_between(angles, mu - sd, mu + sd, color=col, alpha=0.12, linewidth=0)
                ax.plot(angles, mu, color=col, marker=mk, markersize=4, linewidth=2, label=fr)
                ends.append([mu[-1], fr, angles[-1]])
            ends.sort()                        # spread end labels so they never overlap / 錯開尾端標籤
            for n in range(1, len(ends)):
                ends[n][0] = max(ends[n][0], ends[n - 1][0] + 3.0)
            for y, fr, x in ends:
                ax.annotate(fr, (x, y), xytext=(5, 0), textcoords="offset points",
                            fontsize=9, color="#52514e", va="center")
            ax.axhline(0, color="#bdbbb5", linewidth=1)
            ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel(f"camera {axis} (deg)")
            ax.set_title(f"{proto} training · {axis}", fontsize=10, loc="left", color="#0b0b0b")
        axes[i][0].set_ylabel("extra heading error vs level (deg)")
    axes[0][1].legend(frameon=False, fontsize=9, loc="upper center")
    fig.suptitle("Roll, not pitch, degrades heading, and every label frame degrades alike",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(res / "tilt_degradation.png", dpi=160)
    plt.close(fig)
    L += ["## 4. Degradation curves / 退化曲線", "", "![degradation](tilt_degradation.png)", "",
          "Lines: mean over folds; bands: ±1 std over folds; decoder = none.",
          "線：各折平均；色帶：各折 ±1 標準差；解碼方式 = none。"]
    dfiles = sorted((res / "derot").glob("*.csv.gz")) if (res / "derot").exists() else []
    if dfiles:
        d = pd.concat([pd.read_csv(f) for f in dfiles], ignore_index=True)
        L += ["", "## 5. A1 mechanism check: rotate rolled crops back upright / 機制檢驗：把側傾裁切轉回直立", "",
              "Level-trained models. `derot` rotates each crop by the TRUE roll before inference (oracle).",
              "水平訓練的模型。`derot` 在推論前依真實側傾把裁切轉回（神諭）。", "",
              "| frame | level err (°) | roll ±12 plain | roll ±12 derot | roll ±20 plain | roll ±20 derot |",
              "|---|---|---|---|---|---|"]
        for fr in ("cam", "alpha", "allo"):
            g = d[d.frame == fr]
            if not len(g):
                continue
            lvl = g[(g["mode"] == "plain") & (g.roll == 0)].groupby("fold").err_deg.mean()
            cells = []
            for a in (12, 20):
                for mode in ("plain", "derot"):
                    h = g[g["mode"] == mode].rename(columns={})
                    h = h.assign(pitch=0.0)
                    lv0 = g[(g["mode"] == "plain") & (g.roll == 0)].assign(pitch=0.0)
                    hh = pd.concat([h[h.roll != 0], lv0])
                    cells.append(ms(paired_delta(hh, [(0.0, float(a)), (0.0, float(-a))])))
            L.append(f"| {fr} | {lvl.mean():.1f} | " + " | ".join(cells) + " |")
        L += ["", "Numbers are extra error vs the same objects with a level camera. A `derot` value near 0 means "
              "in-plane rotation explains the roll damage. Rotated crops gain black corners, a small extra handicap.",
              "數值為與同一批物體水平相機相比多出的誤差。`derot` 接近 0 代表影像平面內旋轉可解釋側傾損害。"
              "轉正後的裁切角落會出現黑邊，是一個小額外不利因素。"]
    rfiles = sorted((res / "derot_reflect").glob("*.csv.gz")) if (res / "derot_reflect").exists() else []
    if rfiles and dfiles:
        db = pd.concat([pd.read_csv(f) for f in dfiles], ignore_index=True)
        dr_ = pd.concat([pd.read_csv(f) for f in rfiles], ignore_index=True)

        def r20(g, mode):
            # Per-fold roll ±20 error minus the same objects' level (plain, roll 0) error.
            # 各折側傾 ±20 誤差減去同一批物體水平誤差。
            g = g.assign(pitch=0.0)
            h = pd.concat([g[(g["mode"] == mode) & (g.roll != 0)], g[(g["mode"] == "plain") & (g.roll == 0)]])
            return paired_delta(h, [(0.0, 20.0), (0.0, -20.0)])

        L += ["", "## 5b. A1c corner-fill check: black vs mirrored corners / 角落填補檢驗：黑邊 vs 鏡射", "",
              "Same models and crops as section 5. Rotating a square crop by 20° leaves ~12% of its pixels "
              "uncovered; `black` fills them with black (as in section 5), `reflect` with mirrored image content. "
              "A real system would de-rotate the whole frame, where such corners do not appear around the object.",
              "與第 5 節相同的模型與裁切。正方形裁切旋轉 20° 會露出約 12% 的像素；`black` 以黑色填補（同第 5 節），"
              "`reflect` 以鏡射影像內容填補。實際系統會轉正整張影像，物體周圍不會出現這種角落。", "",
              "| frame | roll ±20 plain | derot black | derot reflect | per fold, black | per fold, reflect | folds where reflect is closer to level |",
              "|---|---|---|---|---|---|---|"]
        for fr in ("cam", "alpha", "allo"):
            gb, gr = db[db.frame == fr], dr_[dr_.frame == fr]
            if not len(gb) or not len(gr):
                continue
            pl, bk, rf = r20(gb, "plain"), r20(gb, "derot"), r20(gr, "derot")
            common = bk.index.intersection(rf.index)
            better = int((rf[common].abs() < bk[common].abs()).sum())
            L.append(f"| {fr} | {ms(pl)} | {ms(bk)} | {ms(rf)} | "
                     + ", ".join(f"{v:+.1f}" for v in bk[common]) + " | "
                     + ", ".join(f"{v:+.1f}" for v in rf[common]) + f" | {better}/{len(common)} |")
        L += ["", "Extra error at roll ±20° vs the same objects with a level camera (mean ± std over folds).",
              "側傾 ±20° 相對同一批物體水平相機多出的誤差（各折平均 ± 標準差）。"]
    any_noise = False
    for ndir_name, heading in [
            ("derot_noise", "## 6. A1b de-rotation with a noisy roll estimate, black corners / 以含誤差的側傾估計轉正（黑色角落）"),
            ("derot_noise_reflect", "## 6b. A1d the same with mirrored corners / 同上，鏡射角落")]:
        nfiles = sorted((res / ndir_name).glob("*.csv.gz")) if (res / ndir_name).exists() else []
        if not nfiles:
            continue
        any_noise = True
        d = pd.concat([pd.read_csv(f) for f in nfiles], ignore_index=True)
        sigmas = sorted(d.sigma.unique())
        L += ["", heading, "",
              "Each crop is rotated by (true roll + N(0, σ)); level crops are rotated by the noise alone. "
              "Extra error vs the same objects, level camera, NO rotation. Mean ± std over folds.",
              "每個裁切依（真實側傾 + 常態誤差 σ）旋轉；水平裁切只依誤差旋轉。數值為相對於同一批物體、"
              "水平相機、不旋轉時多出的誤差（各折平均 ± 標準差）。", "",
              "| frame | " + " | ".join(f"σ={s:g}° level | σ={s:g}° roll ±20" for s in sigmas) + " |",
              "|---|" + "---|---|" * len(sigmas)]
        for fr in ("cam", "alpha", "allo"):
            g = d[d.frame == fr]
            if not len(g):
                continue
            base = g[(g.sigma == 0) & (g.roll == 0)].groupby("fold").err_deg.mean()
            cells = []
            for sg in sigmas:
                h = g[g.sigma == sg].groupby(["fold", "roll"]).err_deg.mean()
                lvl = pd.Series({k: h[(k, 0.0)] - base[k] for k in base.index})
                r20 = pd.Series({k: (h[(k, 20.0)] + h[(k, -20.0)]) / 2 - base[k] for k in base.index})
                cells += [ms(lvl), ms(r20)]
            L.append(f"| {fr} | " + " | ".join(cells) + " |")
    if any_noise:
        ref = ""
        if dfiles:
            dd = pd.concat([pd.read_csv(f) for f in dfiles], ignore_index=True)
            vals = []
            for fr in ("cam", "alpha", "allo"):
                g = dd[(dd.frame == fr) & (dd["mode"] == "plain")].assign(pitch=0.0)
                if len(g):
                    vals.append(paired_delta(g, [(0.0, 20.0), (0.0, -20.0)]).mean())
            if vals:
                lo, hi = min(vals), max(vals)
                ref = "yes"
        if ref:
            L += ["", f"For reference, roll ±20° WITHOUT any de-rotation costs {lo:+.1f} to {hi:+.1f}° with this backbone (section 5).",
                  f"對照：此骨幹在完全不轉正時，側傾 ±20° 多出 {lo:+.1f}° 至 {hi:+.1f}°（見第 5 節）。"]
    (res / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# =========================================================================== #
# Self-test: geometry must be exact before any training / 訓練前先驗證幾何
# =========================================================================== #
def cmd_selftest(_args: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    K = np.array([[1266.0, 0, 816], [0, 1266.0, 491], [0, 0, 1]])
    n_ok = 0

    def check(name, cond):
        nonlocal n_ok
        if not cond:
            raise AssertionError(f"FAILED: {name}")
        n_ok += 1
        print(f"  ok  {name}")

    # 1. Homography = rotating the 3D points. 單應變形 = 旋轉 3D 點。
    P = rng.uniform([-5, -1, 8], [5, 2, 40], (50, 3)).T
    R = tilt_matrix(7, -11)
    uv_rot = (K @ (R @ P))[:2] / (K @ (R @ P))[2]
    h = K @ R @ np.linalg.inv(K) @ np.vstack([(K @ P)[:2] / (K @ P)[2], np.ones(50)])
    check("homography matches 3D rotation", np.allclose(h[:2] / h[2], uv_rot, atol=1e-6))

    # 2. 6D round trip and ray rotation. 6D 往返與視線旋轉。
    Q = np.linalg.qr(rng.normal(size=(20, 3, 3)))[0]
    Q[np.linalg.det(Q) < 0, :, 2] *= -1
    check("6D -> R round trip", np.allclose(R_from_r6(r6_from_R(Q)), Q, atol=1e-9))
    d = pixel_rays(rng.uniform(0, 1600, 20), rng.uniform(0, 900, 20), 1266, 1266, 816, 491)
    Rr = ray_rotation(d)
    check("ray rotation maps +z to ray", np.allclose(Rr[:, :, 2], d, atol=1e-9))
    check("ray rotation is orthonormal", np.allclose(Rr @ Rr.transpose(0, 2, 1), np.eye(3), atol=1e-9))

    # 3. Perfect predictions decode back to the level heading. 完美預測可解碼回水平朝向。
    n = 200
    yaw = rng.uniform(-np.pi, np.pi, n)
    O = np.stack([rot_y_box(y) for y in yaw])           # upright objects at level
    pitch, roll = rng.uniform(-20, 20, n), rng.uniform(-20, 20, n)
    Rt = tilt_matrices(pitch, roll)
    dpx = pixel_rays(rng.uniform(100, 1500, n), rng.uniform(100, 800, n), 1266, 1266, 816, 491)
    labs = pd.DataFrame([labels_for(Rt[i], O[i], dpx[i]) for i in range(n)])
    df = pd.DataFrame({"dx": dpx[:, 0], "dy": dpx[:, 1], "dz": dpx[:, 2]})
    level = yaw_of(O[:, :, 0])
    check("level yaw = box yaw", np.allclose(wrap(level - yaw), 0, atol=1e-9))
    sc = lambda a: np.stack([np.sin(a), np.cos(a)], -1)
    six = labs[[f"a{i}" for i in range(6)]].to_numpy()
    for fr, p in [("cam", sc(labs.cam_yaw.to_numpy())), ("alpha", sc(labs.alpha.to_numpy())), ("allo", six)]:
        y = decode(fr, p, df, "ext", Rt)
        check(f"{fr}: exact extrinsics recover level yaw", np.abs(wrap(y - level)).max() < 1e-6)
    # Pure pitch and pure roll: object up-axis recovers gravity exactly.
    for name, pp, rr in [("pitch", pitch, np.zeros(n)), ("roll", np.zeros(n), roll)]:
        Rs = tilt_matrices(pp, rr)
        ls = pd.DataFrame([labels_for(Rs[i], O[i], dpx[i]) for i in range(n)])
        yg = decode("allo", ls[[f"a{i}" for i in range(6)]].to_numpy(), df, "grav")
        check(f"allo/grav recovers level yaw under pure {name}", np.abs(wrap(yg - level)).max() < 1e-6)
    # At level, every decoder agrees. 水平時所有解碼一致。
    l0 = pd.DataFrame([labels_for(np.eye(3), O[i], dpx[i]) for i in range(n)])
    for fr, p in [("cam", sc(l0.cam_yaw.to_numpy())), ("alpha", sc(l0.alpha.to_numpy())),
                  ("allo", l0[[f"a{i}" for i in range(6)]].to_numpy())]:
        check(f"{fr}: level decode without extrinsics is exact",
              np.abs(wrap(decode(fr, p, df, "none") - level)).max() < 1e-6)
    # Without extrinsics, tilt must actually hurt a yaw-only frame (sanity). 未知外參時傾斜確實造成誤差。
    y_none = decode("cam", sc(labs.cam_yaw.to_numpy()), df, "none")
    check("tilt without extrinsics causes error for cam", np.degrees(np.abs(wrap(y_none - level))).mean() > 0.5)
    try:  # A1c: both corner fills rotate the same way about the same centre
        import cv2  # noqa: F401
        from PIL import Image

        img = np.asarray(Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).resize((128, 128)))
        pb = np.asarray(rotate_crop(Image.fromarray(img), 20.0, "black")).astype(float)
        pr = np.asarray(rotate_crop(Image.fromarray(img), 20.0, "reflect")).astype(float)
        check("black and reflect rotations agree inside the crop", np.abs(pb - pr)[40:88, 40:88].mean() < 2.0)
    except ImportError:
        print("  --  opencv not installed, skipping rotation-fill check")
    print(f"\nAll {n_ok} geometry checks passed. 全部 {n_ok} 項幾何檢查通過。")


def rot_y_box(yaw: float) -> np.ndarray:
    """Upright box with heading `yaw` in camera frame (y down). 相機座標中直立、朝向為 yaw 的物體。"""
    h = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    up = np.array([0.0, -1.0, 0.0])
    left = np.cross(up, h)
    return np.stack([h, left, up], axis=1)


# =========================================================================== #
# CLI
# =========================================================================== #
def main() -> None:
    # Windows: when output is redirected, Python falls back to cp950 and crashes on symbols like "−".
    # Windows 重新導向輸出時會改用 cp950，遇到「−」等符號會當掉；強制使用 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="PDS Experiment 1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest"); s.set_defaults(func=cmd_selftest)

    p = sub.add_parser("prepare")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--out", default="data")
    p.add_argument("--category", default="vehicle.car")
    p.add_argument("--min-size", type=int, default=32)
    p.add_argument("--min-lidar-pts", type=int, default=5)
    p.set_defaults(func=cmd_prepare)

    r = sub.add_parser("render")
    r.add_argument("--data", default="data")
    r.add_argument("--dataroot", default=None)
    r.add_argument("--grid", choices=["full", "small"], default="full")
    r.add_argument("--aug-copies", type=int, default=3)
    r.add_argument("--size", type=int, default=128)
    r.add_argument("--min-inside", type=float, default=0.7)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_render)

    u = sub.add_parser("run")
    u.add_argument("--data", default="data")
    u.add_argument("--out", default="results")
    u.add_argument("--folds", type=int, default=5)
    u.add_argument("--max-folds", type=int, default=0)
    u.add_argument("--epochs", type=int, default=12)
    u.add_argument("--bs", type=int, default=64)
    u.add_argument("--lr", type=float, default=3e-4)
    u.add_argument("--size", type=int, default=128)
    u.add_argument("--workers", type=int, default=2)
    u.add_argument("--seed", type=int, default=0)
    u.add_argument("--protocols", nargs="*", choices=PROTOCOLS)
    u.add_argument("--frames", nargs="*", choices=FRAMES)
    u.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    u.add_argument("--cpu", action="store_true")
    u.add_argument("--overwrite", action="store_true")
    u.add_argument("--verbose", action="store_true")
    u.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    u.set_defaults(func=cmd_run)

    dr = sub.add_parser("derot", help="A1 mechanism check: de-rotate rolled crops (oracle)")
    dr.add_argument("--data", default="data")
    dr.add_argument("--out", default="results")
    dr.add_argument("--folds", type=int, default=5)
    dr.add_argument("--max-folds", type=int, default=0)
    dr.add_argument("--epochs", type=int, default=12)
    dr.add_argument("--bs", type=int, default=64)
    dr.add_argument("--lr", type=float, default=3e-4)
    dr.add_argument("--size", type=int, default=128)
    dr.add_argument("--workers", type=int, default=2)
    dr.add_argument("--seed", type=int, default=0)
    dr.add_argument("--frames", nargs="*", choices=FRAMES)
    dr.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    dr.add_argument("--cpu", action="store_true")
    dr.add_argument("--overwrite", action="store_true")
    dr.add_argument("--verbose", action="store_true")
    dr.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    dr.add_argument("--fill", choices=["black", "reflect"], default="black",
                    help="corner fill after rotation; reflect writes to derot_reflect/ (A1c)")
    dr.set_defaults(func=cmd_derot)

    dn = sub.add_parser("derotnoise", help="A1b: de-rotation with a noisy roll estimate")
    for a, kw in [("--data", dict(default="data")), ("--out", dict(default="results")),
                  ("--folds", dict(type=int, default=5)), ("--max-folds", dict(type=int, default=0)),
                  ("--epochs", dict(type=int, default=12)), ("--bs", dict(type=int, default=64)),
                  ("--lr", dict(type=float, default=3e-4)), ("--size", dict(type=int, default=128)),
                  ("--workers", dict(type=int, default=2)), ("--seed", dict(type=int, default=0)),
                  ("--noise", dict(type=float, nargs="*", default=[0.0, 1.0, 2.0, 5.0]))]:
        dn.add_argument(a, **kw)
    dn.add_argument("--frames", nargs="*", choices=FRAMES)
    dn.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    dn.add_argument("--cpu", action="store_true")
    dn.add_argument("--overwrite", action="store_true")
    dn.add_argument("--verbose", action="store_true")
    dn.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    dn.add_argument("--fill", choices=["black", "reflect"], default="black",
                    help="corner fill after rotation; reflect writes to derot_noise_reflect/ (A1d)")
    dn.set_defaults(func=cmd_derotnoise)

    q = sub.add_parser("report")
    q.add_argument("--results", default="results")
    q.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
