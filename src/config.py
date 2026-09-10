"""配置加载。

config.yaml 里的路径统一写相对路径（相对仓库根目录），
由 here() / data_path() 负责解析成绝对路径，避免依赖调用者的工作目录。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "src" / "config.yaml"


class Config:
    """薄薄一层包装，支持 cfg.get("capture.interval_minutes", 20) 这种点号取值。"""

    def __init__(self, data: dict[str, Any]):
        self._d = data

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._d
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def raw(self) -> dict[str, Any]:
        return self._d


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f) or {})


def data_path(cfg: Config, key: str, default: str) -> Path:
    """把配置里的相对目录解析成绝对路径（相对仓库根），并确保目录存在。"""
    p = ROOT / str(cfg.get(key, default))
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve(path_like: str | Path) -> Path:
    """相对路径按仓库根解析；绝对路径原样返回。"""
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def resolve_patch(patch, shape) -> tuple[int, int, int, int]:
    """把白卡坐标统一成像素 (x, y, w, h)。

    四个值**全部 ≤ 1.0** 时按相对比例解释，否则按像素解释。
    相对坐标的好处是换分辨率不用重标（演示用 540p、真机用 1080p 是常态）；
    像素坐标则适合真机上对着截图量出来的情况。
    """
    H, W = shape[:2]
    vals = [float(v) for v in patch]
    if max(vals) <= 1.0:
        x, y, w, h = vals
        return int(x * W), int(y * H), int(w * W), int(h * H)
    return (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]))
