"""浑浊度估计 —— 用于污染的早期预警。

为什么不做"识别杂菌"：杂菌在显微镜下没什么可辨认的形态，
真正可靠的早期信号是**培养基整体开始变浑**。

因此这里测的是画面的"清晰度"：
清晰度下降 → 浊度上升。用拉普拉斯方差，它对细节丢失非常敏感，
且不需要任何标定，非常适合做第一道防线。

告警用的是**相对基线**的倍数，而不是绝对值 —— 每台设备、每次装液的光学
条件都不同，只有和"自己最初清澈时的样子"比才有意义。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import Config


def clarity_score(img: np.ndarray, exclude: tuple[int, int, int, int] | None = None) -> float:
    """清晰度得分 = 归一化动态范围 (p95 - p5) / (p95 + p5)。值越大越清澈。

    ==================== 为什么是这个量 ====================
    这是本项目迭代了四轮才定下来的指标，三次翻车都值得记下来：

    第 1 轮：拉普拉斯方差。**方向反了** —— 培养基变浑时噪声同时变大，
        而该指标对噪声极敏感，于是"更浑浊"被读成"更清晰"。

    第 2 轮：灰度标准差（对比度）。方向对了，但有个致命耦合：
        培养过程中细胞越长越多，对比度本来就在上升。
        生长与污染两个效应互相抵消，20 小时的污染情景只测出 1.03 倍。

    第 3 轮：p95 - p5（动态范围）。物理上浑浊会同时**抬高暗部**（散射把光
        填进阴影）和**压暗亮部**（雾化），动态范围因而收缩；而汇合度只改变
        "有多少像素处于这两个水平"，几乎不改变水平本身。
        实测：浑浊 42% / 汇合 1% —— 汇合度耦合基本消除。
        但 pH 漂移会把培养基整体变亮变暗，又带来 34% 的干扰。

    第 4 轮（当前）：再除以 (p95 + p5) 做归一化。
        归一化后，培养基整体亮度的变化被约掉，只留下"相对对比度"的塌缩。
        三维网格（汇合度 × pH × 浑浊度）实测：
            range       浑浊 42% / 汇合 1% / pH 34%   信噪比 1.19
            norm_range  浑浊 55% / 汇合 7% / pH 15%   信噪比 2.60
        且在所有 60 个格点上对浑浊度严格单调。
    =======================================================
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if exclude is not None:
        x, y, w, h = (int(v) for v in exclude)
        pad = 8
        gray[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad] = np.nan

    # 只取中央区域，避开暗角
    H, W = gray.shape
    m = int(0.06 * min(H, W))
    roi = gray[m : H - m, m : W - m]

    # NaN 不能直接参与模糊，先用中位数填上
    nan_mask = np.isnan(roi)
    if nan_mask.all():
        return 0.0
    if nan_mask.any():
        roi = np.where(nan_mask, np.nanmedian(roi), roi)

    # 先模糊压掉像素噪声，再取分位数 —— 分位数本身抗离群值，不用取极值
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=2.0)
    p5, p95 = np.percentile(blurred, [5, 95])
    mid = float(p95 + p5)
    if mid < 1e-6:
        return 0.0
    return float((p95 - p5) / mid)


class TurbidityTracker:
    """用最初 N 帧建立"清澈基线"，之后每帧给出相对浑浊度倍数。"""

    def __init__(self, cfg: Config):
        self.baseline_frames = int(cfg.get("analyze.turbidity.baseline_frames", 5))
        self.warn_ratio = float(cfg.get("analyze.turbidity.warn_ratio", 1.30))
        self.alert_ratio = float(cfg.get("analyze.turbidity.alert_ratio", 1.60))
        self._samples: list[float] = []
        self._baseline: float | None = None

    @property
    def baseline(self) -> float | None:
        return self._baseline

    def update(self, clarity: float) -> float | None:
        """返回浑浊度倍数。基线尚未建立时返回 None。"""
        if self._baseline is None:
            self._samples.append(clarity)
            if len(self._samples) >= self.baseline_frames:
                self._baseline = float(np.median(self._samples))
            return None

        if clarity < 1e-6:
            return float("inf")
        return self._baseline / clarity

    def level(self, ratio: float | None) -> str:
        if ratio is None:
            return "unknown"
        if ratio >= self.alert_ratio:
            return "alert"
        if ratio >= self.warn_ratio:
            return "warn"
        return "ok"
