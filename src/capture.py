"""图像采集。

两种相机实现共用同一个接口：

- MockCamera  合成图像。不需要任何硬件，让整条分析链路在笔记本上就能跑通、
  可复现、可写单元测试。第一阶段的大部分算法调试都在这上面完成。
- PiCamera    树莓派 + picamera2（或任意 USB 摄像头）。

刻意保留合成图像并非权宜之计：真实培养实验每周只能拿到有限几帧，
而合成图像可以按需生成任意 pH / 浑浊度 / 汇合度组合，
是标定算法和回归测试的必需品。
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import Config, data_path, resolve_patch
from .imgio import imwrite

# 酚红指示剂在培养基中的近似显色（pH -> RGB）
PHENOL_RED = [
    (6.40, (235, 190, 70)),   # 偏酸：黄
    (6.80, (228, 150, 75)),
    (7.00, (220, 120, 85)),
    (7.20, (215, 100, 95)),   # 正常：橙红
    (7.40, (205, 90, 110)),
    (7.80, (185, 80, 145)),   # 偏碱：紫红
]


def phenol_red_rgb(ph: float) -> tuple[int, int, int]:
    """按 pH 线性插值出培养基的近似 RGB 颜色。"""
    if ph <= PHENOL_RED[0][0]:
        return PHENOL_RED[0][1]
    if ph >= PHENOL_RED[-1][0]:
        return PHENOL_RED[-1][1]
    for (p0, c0), (p1, c1) in zip(PHENOL_RED, PHENOL_RED[1:]):
        if p0 <= ph <= p1:
            t = (ph - p0) / (p1 - p0)
            return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))
    return PHENOL_RED[-1][1]


def synth_frame(
    confluence: float = 0.30,
    turbidity: float = 0.05,
    ph: float = 7.20,
    size: tuple[int, int] = (1080, 1920),
    reference_patch: tuple[int, int, int, int] = (1620, 40, 120, 120),
    illumination: float = 1.0,
    seed: int | None = 0,
    noise_seed: int | None = None,
) -> np.ndarray:
    """生成一帧合成的培养图像（BGR）。

    参数
    ----
    confluence    : 细胞覆盖面积占比 0~1
    turbidity     : 浑浊度 0~1，越大画面越"雾"、噪声越多、对比度越低
    ph            : 培养基 pH，决定底色（酚红显色）
    illumination  : 全局亮度漂移，用来验证白平衡校正是否真的起作用
    seed          : **细胞布局**的随机种子。连续帧请传同一个值 ——
                    真实培养里菌落位置是固定的，只有数量在变。
                    每帧换种子会让汇合度剧烈抖动，把趋势分析彻底带偏。
    noise_seed    : 像素噪声的种子，应当逐帧变化。
    """
    h, w = size
    rng = np.random.default_rng(seed)
    nrng = np.random.default_rng(noise_seed if noise_seed is not None else seed)

    # ---- 1. 培养基底色 ----
    r, g, b = phenol_red_rgb(ph)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    # 加一点空间不均匀性，模拟真实光照。
    # 幅度取 10% —— 和 BOM 里那块透射背光板的实际均匀度接近；
    # 早先取 18% 时，暗角造成的灰度跨度已经和细胞对比度相当，
    # 任何全局阈值法都会被暗角带偏（详见 docs/log 里的复盘）。
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vignette = 1.0 - 0.10 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    img *= vignette[:, :, None]

    # ---- 2. 铺细胞 ----
    mask = np.zeros((h, w), dtype=np.uint8)
    target = float(np.clip(confluence, 0, 0.95)) * h * w
    radius = 11
    covered = 0
    placed = 0
    # 上限给足：圆盘是**允许重叠**的随机投放，越接近铺满、每个新圆贡献的有效面积越小，
    # 所以需要的个数远多于 target/(πr²)。上限给太紧会让高汇合度帧提前停止，
    # 表现为"0.65 和 0.85 测出来一样"——那是生成器撞顶，不是测量失准。
    max_cells = int(8 * target / (math.pi * radius**2)) + 256
    # 注意：mask.sum() 是 O(h*w)，不能每画一个圆就调一次，
    # 否则铺满高汇合度需要几千次全图求和，慢到不可用。改成每 16 个圆核一次。
    while covered < target and placed < max_cells:
        cy = int(rng.integers(radius, h - radius))
        cx = int(rng.integers(radius, w - radius))
        cv2.circle(mask, (cx, cy), radius, 1, -1)
        placed += 1
        if placed % 16 == 0:
            covered = int(mask.sum())
    covered = int(mask.sum())

    if placed:
        # 明场下细胞比培养基暗得多（不是"略暗"）。
        # 早先只做到暗 17%，比暗角的影响还小，等于把信噪比做没了。
        cell_color = np.array([b * 0.50 + 10, g * 0.46 + 10, r * 0.42 + 10], dtype=np.float32)
        texture = rng.normal(1.0, 0.10, size=(h, w, 1)).astype(np.float32)
        cells = np.clip(cell_color[None, None, :] * texture, 0, 255)
        # 细胞边缘做一点羽化，避免锯齿
        soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)[:, :, None]
        img = img * (1 - soft) + cells * soft

    # ---- 3. 浑浊度：雾化 + 噪声 + 对比度下降 ----
    t = float(np.clip(turbidity, 0, 1))
    if t > 0:
        haze = np.full_like(img, 225.0)
        img = img * (1 - 0.45 * t) + haze * (0.45 * t)
        img = (img - img.mean()) * (1 - 0.35 * t) + img.mean()
        img += nrng.normal(0, 14 * t, img.shape).astype(np.float32)

    # ---- 4. 全局亮度漂移 ----
    img *= illumination

    # ---- 5. 标准白块（用于同框白平衡校正）----
    # 支持相对/像素两种写法，与 config.resolve_patch 保持同一套约定
    x, y, pw, ph_ = resolve_patch(reference_patch, (h, w))
    if x + pw <= w and y + ph_ <= h:
        img[y : y + ph_, x : x + pw] = 242.0  # 标准白：本身不受培养基颜色影响

    return np.clip(img, 0, 255).astype(np.uint8)


class MockCamera:
    """按预设情景逐帧漂移的合成相机。

    细胞布局用固定种子 —— 真实培养箱里菌落不会每 20 分钟重新洗一次牌，
    只有覆盖面积在变。布局固定后，汇合度序列才是连续的，趋势判定才有意义。
    """

    LAYOUT_SEED = 20260910

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.size = tuple(cfg.get("capture.resolution", [1080, 1920]))
        self.ref = tuple(cfg.get("analyze.ph.reference_patch", [1620, 40, 120, 120]))
        self.scenario = str(cfg.get("capture.mock_scenario", "normal"))
        self.interval_min = float(cfg.get("capture.interval_minutes", 20))
        self.n = 0

    def grab(self) -> np.ndarray:
        """返回下一帧。按帧序号推进情景参数，模拟培养箱内随时间发生的变化。"""
        i = self.n
        self.n += 1
        hours = i * self.interval_min / 60.0

        confluence, turbidity, ph = 0.06, 0.03, 7.20

        if self.scenario == "normal":
            # 逻辑斯蒂式生长，24h 左右进入平台期
            confluence = 0.06 + 0.72 / (1 + math.exp(-(hours - 10) / 3.5))
            turbidity = 0.03 + 0.02 * math.sin(hours / 6)
            ph = 7.20 - 0.03 * math.sin(hours / 9)
        elif self.scenario == "growth":
            confluence = min(0.06 + 0.03 * hours, 0.95)
            ph = 7.20 - 0.01 * hours
        elif self.scenario == "contamination":
            confluence = 0.05 + 0.55 / (1 + math.exp(-(hours - 8) / 3.0))
            # 关键：第 8h 起杂菌进入对数期，浑浊度指数抬升（真实污染就是这个形态）
            turbidity = 0.03 + (0.0 if hours < 8 else 0.72 * (1 - math.exp(-(hours - 8) / 3.0)))
            ph = 7.20 - 0.06 * max(0, hours - 8)
        elif self.scenario == "ph_drift":
            confluence = 0.05 + 0.45 / (1 + math.exp(-(hours - 9) / 3.5))
            turbidity = 0.03
            ph = max(6.40, 7.20 - 0.045 * hours)

        # 每帧给一点光照抖动，逼着白平衡模块必须真的工作
        illum = 1.0 + 0.05 * math.sin(i / 3.0)

        return synth_frame(
            confluence=confluence,
            turbidity=turbidity,
            ph=ph,
            size=self.size,
            reference_patch=self.ref,
            illumination=illum,
            seed=self.LAYOUT_SEED,   # 布局固定
            noise_seed=i,            # 噪声逐帧变化
        )


class PiCamera:
    """树莓派相机（picamera2）。需要真实硬件，import 延迟到实例化时。"""

    def __init__(self, cfg: Config):
        from picamera2 import Picamera2  # 仅在真实硬件上才需要

        self.size = tuple(cfg.get("capture.resolution", [1080, 1920]))
        self.cam = Picamera2()
        self.cam.configure(
            self.cam.create_still_configuration(
                main={"size": (self.size[1], self.size[0])}
            )
        )
        self.cam.start()

    def grab(self) -> np.ndarray:
        rgb = self.cam.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_camera(cfg: Config):
    if bool(cfg.get("capture.mock", True)):
        return MockCamera(cfg)
    return PiCamera(cfg)


def save_frame(
    img: np.ndarray,
    out_dir: Path,
    ts: datetime | None = None,
    quality: int = 92,
) -> Path:
    """把一帧写成带时间戳的 JPEG，返回路径。

    走 imgio.imwrite 而不是 cv2.imwrite —— 后者在中文路径下会静默失败，
    流水线会报"成功"但磁盘上什么都没有（详见 src/imgio.py 的说明）。
    """
    ts = ts or datetime.now()
    out_dir.mkdir(parents=True, exist_ok=True)
    # 精确到毫秒：真机是 20 分钟一帧不冲突，但演示/回放模式会连续出帧，
    # 只用秒级时间戳会让同一秒内的帧互相覆盖，静默丢数据。
    path = out_dir / f"{ts:%Y%m%d_%H%M%S}_{ts.microsecond // 1000:03d}.jpg"
    ok = imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise OSError(f"图像写入失败: {path}")
    return path


def frames_dir(cfg: Config) -> Path:
    return data_path(cfg, "capture.output_dir", "data/raw/frames")
