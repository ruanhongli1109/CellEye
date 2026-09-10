"""CellEye 的核心数据结构。

整个项目围绕两个对象展开：
- SensorReading   —— 某一时刻培养箱的环境读数
- AnalysisResult  —— 某一时刻对一帧图像的判定结果 + 综合告警

两者用同一个 timestamp 对齐，最终落成一行 CSV，就是项目的原始数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class SensorReading:
    """培养箱环境读数。任一字段可为 None —— 传感器缺失不应阻断整条链路。"""

    timestamp: datetime
    temperature_c: float | None = None
    humidity_pct: float | None = None
    co2_ppm: float | None = None
    pressure_hpa: float | None = None

    def to_row(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat(timespec="seconds")
        return d


@dataclass
class AnalysisResult:
    """单帧图像的分析结果。

    ph / turbidity_ratio / confluence 为 None 表示该指标本次未能计算
    （例如缺少标定文件），而不是"计算失败"—— 下游必须容忍 None。
    """

    timestamp: datetime
    image_path: str
    ph: float | None = None
    turbidity_ratio: float | None = None
    confluence: float | None = None
    alerts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        """综合风险等级：ok / warn / alert。"""
        if any(a.startswith("[高]") for a in self.alerts):
            return "alert"
        if self.alerts:
            return "warn"
        return "ok"

    def to_row(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat(timespec="seconds")
        d["alerts"] = "; ".join(self.alerts)
        d["notes"] = "; ".join(self.notes)
        d["level"] = self.level
        return d


@dataclass
class Record:
    """一条完整记录 = 环境读数 + 图像判定，用于写入 CSV。"""

    reading: SensorReading
    result: AnalysisResult

    def to_row(self) -> dict:
        row = self.result.to_row()
        r = self.reading.to_row()
        row.update(
            {
                "temperature_c": r["temperature_c"],
                "humidity_pct": r["humidity_pct"],
                "co2_ppm": r["co2_ppm"],
                "pressure_hpa": r["pressure_hpa"],
            }
        )
        return row
