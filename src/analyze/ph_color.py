"""由培养基颜色反推 pH（酚红比色法）。

思路和实验室里"对着比色卡看"是同一件事，只是把眼睛换成了相机：

1. 同框白平衡 —— 画面内放一块标准白卡，用它把光照漂移消掉。
   没有这一步，昼夜光照变化会被误读成 pH 变化，这是比色法最容易翻车的地方。
2. 取培养基区域的中位色（中位数比均值抗气泡与反光干扰）。
3. 在标定色表中做最近邻 + 反距离加权插值。

标定色表来自 scripts/calibrate_ph.py：用已知 pH 的缓冲液拍一组图，
把每张图的实测颜色存下来。**没做标定之前，本模块只能给出演示级结果。**
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import Config, resolve, resolve_patch


def white_balance(img: np.ndarray, patch, target: float = 240.0) -> np.ndarray:
    """用画面内标准白卡做逐通道增益校正。

    白卡区域越界时**直接抛错**，不静默返回原图 ——
    否则会得到一张没校正过的图，pH 结果偏了却毫无提示。
    （本项目已经在 cv2.imwrite 上吃过一次静默失败的亏。）
    """
    x, y, w, h = resolve_patch(patch, img.shape)
    H, W = img.shape[:2]
    if not (0 <= x and 0 <= y and w > 0 and h > 0 and x + w <= W and y + h <= H):
        raise ValueError(
            f"白卡区域越界: patch=({x}, {y}, {w}, {h})，图像为 {W}×{H}。"
            f"请在 config.yaml 中改用相对坐标（四个值都 ≤ 1.0）或按实际分辨率重新测量。"
        )

    ref = img[y : y + h, x : x + w].reshape(-1, 3).astype(np.float32)
    measured = np.median(ref, axis=0)
    measured = np.where(measured < 1e-3, 1e-3, measured)

    gains = target / measured
    out = img.astype(np.float32) * gains[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def medium_color(img: np.ndarray, roi: tuple[float, float, float, float],
                 medium_percentile: float = 70.0) -> np.ndarray:
    """取培养基区域的中位 RGB。roi 为相对比例 [x, y, w, h]。

    只在**较亮的那部分像素**上取中位色 —— 细胞比培养基暗，
    如果直接对整个区域取中位，高汇合度时中位色会落到细胞上，
    pH 会被读偏（实测偏高 0.4）。取最亮的 (100-p) 分位即可稳定锁定培养基。
    """
    H, W = img.shape[:2]
    rx, ry, rw, rh = roi
    x0, y0 = int(rx * W), int(ry * H)
    x1, y1 = int((rx + rw) * W), int((ry + rh) * H)
    region = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)

    lum = region @ np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR 亮度
    cut = np.percentile(lum, medium_percentile)
    medium = region[lum >= cut]
    if medium.shape[0] < 16:          # 高汇合度下亮像素可能太少
        medium = region
    return np.median(medium, axis=0)  # BGR


def load_calibration(cfg: Config) -> tuple[list[tuple[float, np.ndarray]], str]:
    """返回 (标定点, 来源)。来源为 'measured'（实测标定）或 'demo'（演示色表）。"""
    path = resolve(cfg.get("analyze.ph.calibration_file", "data/processed/ph_calibration.json"))
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = [(float(p["ph"]), np.array(p["rgb"], dtype=np.float32)) for p in data["points"]]
        if pts:
            return pts, "measured"

    demo = cfg.get("analyze.ph.demo_palette") or {}
    pts = [(float(k), np.array(v, dtype=np.float32)) for k, v in demo.items()]
    pts.sort(key=lambda t: t[0])
    return pts, "demo"


def ph_from_color(bgr: np.ndarray, points: list[tuple[float, np.ndarray]]) -> float | None:
    """最近邻 + 反距离加权插值。"""
    if not points:
        return None
    rgb = bgr[::-1].astype(np.float32)  # BGR -> RGB，与色表一致
    dists = np.array([np.linalg.norm(rgb - c) for _, c in points], dtype=np.float32)

    order = np.argsort(dists)
    if dists[order[0]] < 1e-6 or len(points) == 1:
        return float(points[order[0]][0])

    k = order[: min(2, len(points))]
    w = 1.0 / dists[k]
    w = w / w.sum()
    return float(sum(wi * points[i][0] for wi, i in zip(w, k)))


def estimate_ph(img: np.ndarray, cfg: Config) -> float | None:
    """估计 pH。返回 None 表示本次无法给出结果。"""
    patch = tuple(cfg.get("analyze.ph.reference_patch", [1620, 40, 120, 120]))
    roi = tuple(cfg.get("analyze.ph.medium_roi", [0.30, 0.30, 0.40, 0.40]))

    balanced = white_balance(img, patch)
    color = medium_color(balanced, roi)
    points, _ = load_calibration(cfg)
    return ph_from_color(color, points)


def calibration_source(cfg: Config) -> str:
    """'measured' 表示已用真实缓冲液标定；'demo' 表示仍是色表演示。"""
    return load_calibration(cfg)[1]


def apply_white_balance_file(img: np.ndarray, cfg: Config) -> np.ndarray:
    patch = tuple(cfg.get("analyze.ph.reference_patch", [1620, 40, 120, 120]))
    return white_balance(img, patch)
