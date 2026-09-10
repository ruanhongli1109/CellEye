"""汇合度（confluence）估计 —— 细胞铺满视野的比例。

第一阶段用 Otsu 阈值法。它在真实图像上一定不够准（细胞团块、焦外、
光照不均都会骗它），但那正是要如实记录下来的过程：
先用最朴素的方法跑通链路、建立基线，再换成分割模型去对比改进幅度。

====================== 走过的弯路（详见 docs/log）======================
最初加了"平场校正"想做光照不均补偿，先后试过中值滤波和大核形态学闭运算。
两种都失败了，原因值得记下来：

- 中值滤波核尺寸和细胞直径相当 → 65% 汇合度时背景本身被细胞带偏，测成 85%。
- 形态学闭运算（rolling ball 思路）依赖"暗目标比结构元小"，
  而细胞长成连片团块后就不再比结构元小，高汇合度直接失效。

真正的问题其实在图像模型：当时把细胞做成了只比培养基暗 17%，
而暗角造成的灰度跨度相当，信噪比根本不够。
把细胞对比度调到明场该有的水平、暗角按实际背光板收到 10% 之后，
**朴素的全局 Otsu 反而最稳**（0.10~0.65 区间误差 ≤ 0.06）。

结论：先把物理模型做对，再谈算法。装饰性的"高级处理"往往是掩盖错误的。
=======================================================================

method=model 时读取 ONNX/TFLite 权重。当前为**未实现**的占位，
故意让它显式报错而不是悄悄回退 —— 免得把阈值法的结果误当成模型效果。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import Config

# 双峰性守卫：两类的均值差小于这个值，就判定"视野里没有细胞"。
# 实测空白视野的类间差约 8，真实细胞可达 50，取 20 能干净分开。
# 不加这道守卫，Otsu 一定会把暗角渐变切成两类，空白视野也能报出 40% 汇合度。
_MIN_BIMODAL_CONTRAST = 20.0


def _center_roi(img: np.ndarray, margin: float = 0.06) -> tuple[int, int, int, int]:
    H, W = img.shape[:2]
    m = int(margin * min(H, W))
    return m, m, W - 2 * m, H - 2 * m


def confluence_threshold(
    img: np.ndarray,
    exclude: tuple[int, int, int, int] | None = None,
) -> float:
    """Otsu 阈值法估计汇合度。返回 0~1。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    x, y, w, h = _center_roi(img)
    roi = gray[y : y + h, x : x + w].astype(np.float32)

    # 挖掉标准白卡等人工区域，用中位数填上，免得被算成前景
    if exclude is not None:
        ex, ey, ew, eh = (int(v) for v in exclude)
        rx0, ry0 = max(0, ex - x), max(0, ey - y)
        rx1, ry1 = min(w, ex + ew - x), min(h, ey + eh - y)
        if rx1 > rx0 and ry1 > ry0:
            roi[ry0:ry1, rx0:rx1] = np.nan

    nan_mask = np.isnan(roi)
    if nan_mask.all():
        return 0.0
    if nan_mask.any():
        roi = np.where(nan_mask, np.nanmedian(roi), roi)

    norm = np.clip(roi, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lo, hi = norm[binary == 0], norm[binary == 255]
    if lo.size == 0 or hi.size == 0:
        return 0.0
    # 双峰性守卫：分不开就说明视野里没有真正的前景
    if abs(float(hi.mean()) - float(lo.mean())) < _MIN_BIMODAL_CONTRAST:
        return 0.0

    # 细胞比背景暗：取均值更低的那一类
    cells = (binary == 0) if lo.mean() < hi.mean() else (binary == 255)
    m = (cells.astype(np.uint8)) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    return float((m > 0).mean())


def confluence_model(img: np.ndarray, cfg: Config) -> float:
    model_path = cfg.get("analyze.confluence.model_path")
    if not model_path:
        raise NotImplementedError(
            "analyze.confluence.model_path 未配置：分割模型尚未训练。"
            "当前请使用 method=threshold。"
        )
    raise NotImplementedError(
        "语义分割推理尚未接入。计划：MobileNetV3-UNet 轻量版，"
        "导出为 ONNX 后用 onnxruntime 推理（树莓派上也跑得动）。"
    )


def confluence_fraction(img: np.ndarray, cfg: Config) -> float:
    patch = tuple(cfg.get("analyze.ph.reference_patch", [1620, 40, 120, 120]))
    method = str(cfg.get("analyze.confluence.method", "threshold")).lower()
    if method == "threshold":
        return confluence_threshold(img, exclude=patch)
    return confluence_model(img, cfg)
