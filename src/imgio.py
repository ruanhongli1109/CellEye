"""路径安全的图像读写。

============================ 为什么需要这个模块 ============================
OpenCV 的 `cv2.imread` / `cv2.imwrite` 在 Windows 上**不支持非 ASCII 路径**。
本项目放在 `F:\\桌面\\CellEye` 下，中文目录直接命中这个缺陷：
imread 返回 None，imwrite 静默返回 False 且不抛异常。

后果非常隐蔽 —— 流水线一路打印"成功"，磁盘上却一张图都没有。
这是实验数据丢失级别的坑，所以统一在这里封掉。

绕过办法是先用 np.fromfile / ndarray.tofile 处理字节流，
再交给 cv2.imdecode / cv2.imencode。

Linux / macOS 上这些函数同样可用（只是多一次内存拷贝），
所以全项目统一走这里，不做平台分支 —— 少一个分支就少一处只在别人机器上炸的代码。
==========================================================================
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读图。路径含中文也安全。读不出来返回 None。"""
    p = Path(path)
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> bool:
    """写图。路径含中文也安全。返回是否真的写成功。

    **调用方应当检查返回值** —— 磁盘满、权限不足、目录不存在都会返回 False。
    """
    p = Path(path)
    ext = p.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(p))
    return True
