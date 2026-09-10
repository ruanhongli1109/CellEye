"""路径安全图像 IO 的回归测试。

这条测试守的是一个**真实踩过的坑**：项目放在 `F:\\桌面\\CellEye` 下，
而 `cv2.imwrite` 在 Windows 上遇到中文路径会静默返回 False 且不抛异常，
导致流水线一路报"成功"、磁盘上却一张图都没有。

用中文目录名做往返测试，防止这个坑再回来。
"""

from __future__ import annotations

import numpy as np

from src.imgio import imread, imwrite


def test_roundtrip_ascii(tmp_path):
    img = (np.random.default_rng(0).random((40, 60, 3)) * 255).astype(np.uint8)
    p = tmp_path / "frame.jpg"
    assert imwrite(p, img) is True
    back = imread(p)
    assert back is not None
    assert back.shape == img.shape


def _smooth_image(h=40, w=60, seed=1):
    """平滑图像。测的是 IO 路径，不该拿纯随机噪声去测 ——
    噪声是 JPEG 的最坏情况，失真是压缩算法的性质，不是我们这层的问题。"""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (x / w * 200 + y / h * 40)
    img = np.stack([base, base * 0.7 + 30, base * 0.5 + 60], axis=-1)
    img += np.random.default_rng(seed).normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_roundtrip_non_ascii_path(tmp_path):
    """中文路径必须能存能读 —— 这是本项目的实际部署路径。"""
    img = _smooth_image()
    d = tmp_path / "桌面" / "实验数据" / "20260910"
    p = d / "图像_001.jpg"

    assert imwrite(p, img) is True, "中文路径写入失败"
    assert p.exists(), "imwrite 返回 True 但文件不存在"

    back = imread(p)
    assert back is not None, "中文路径读取失败"
    assert back.shape == img.shape
    # JPEG 有损，不能断言逐像素相同；用 PNG 单独验证无损往返
    assert np.abs(back.astype(int) - img.astype(int)).mean() < 3.0, "往返失真过大"


def test_roundtrip_lossless_non_ascii(tmp_path):
    """PNG 路径下必须是完全无损的往返。"""
    img = (np.random.default_rng(2).random((40, 60, 3)) * 255).astype(np.uint8)
    p = tmp_path / "桌面" / "无损.png"
    assert imwrite(p, img) is True
    back = imread(p)
    assert back is not None
    assert np.array_equal(back, img), "PNG 往返应当逐像素一致"


def test_imwrite_creates_parent_dirs(tmp_path):
    img = np.zeros((10, 10, 3), np.uint8)
    p = tmp_path / "a" / "b" / "c.jpg"
    assert imwrite(p, img) is True
    assert p.exists()


def test_imread_missing_file_returns_none(tmp_path):
    assert imread(tmp_path / "不存在.jpg") is None
