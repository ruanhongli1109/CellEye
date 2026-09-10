"""图像分析模块。

三个指标各自独立、互不依赖，任何一个算不出来都不影响另外两个：
- ph_color    颜色 → pH
- turbidity   清晰度 → 浑浊度 → 污染风险
- confluence  二值分割 → 细胞汇合度
"""

from .confluence import confluence_fraction, confluence_threshold
from .ph_color import calibration_source, estimate_ph
from .turbidity import TurbidityTracker, clarity_score

__all__ = [
    "confluence_fraction",
    "confluence_threshold",
    "estimate_ph",
    "calibration_source",
    "TurbidityTracker",
    "clarity_score",
]
