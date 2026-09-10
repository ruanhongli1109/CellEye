"""主流水线：采集 → 分析 → 判定 → 告警 → 落盘。

一条记录的产生过程就是本项目的最小闭环。
所有判定阈值都写在 config.yaml 里，不在代码里写死 ——
阈值本身是要靠实验数据反复调整的，属于实验参数而非程序常量。
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from .alert import Alerter
from .analyze import TurbidityTracker, calibration_source, clarity_score, confluence_fraction, estimate_ph
from .capture import frames_dir, make_camera, save_frame
from .config import Config, data_path, load_config
from .schema import AnalysisResult, Record, SensorReading
from .sensors import make_sensor_hub

# 判定用的固定阈值（超出这些范围本身就是异常，不属于可调实验参数）
PH_NORMAL = 7.20
PH_WARN_DELTA = 0.30
PH_ALERT_DELTA = 0.50
GROWTH_STALL_WINDOW = 6      # 只看最近 N 帧判断生长是否停滞
GROWTH_STALL_MIN_CONFLUENCE = 0.60   # 汇合度已很高时生长自然变慢，不算异常
GROWTH_STALL_DROP = 0.04     # 前后两半中位数需下降超过这个幅度才算真跌


class Monitor:
    def __init__(self, cfg: Config, camera=None, sensor_hub=None, alerter=None,
                 save_frames: bool = True):
        self.cfg = cfg
        self.camera = camera if camera is not None else make_camera(cfg)
        self.sensor_hub = sensor_hub if sensor_hub is not None else make_sensor_hub(cfg)
        self.alerter = alerter if alerter is not None else Alerter(cfg)
        self.save_frames = save_frames

        self.turbidity = TurbidityTracker(cfg)
        self._confluence_hist: deque[float] = deque(maxlen=GROWTH_STALL_WINDOW)
        self._patch = tuple(cfg.get("analyze.ph.reference_patch", [1620, 40, 120, 120]))
        self._confluence_warn = float(cfg.get("analyze.confluence.warn_confluence", 0.85))
        self._demo_note_emitted = False

    def analyze(self, img: np.ndarray, ts: datetime, image_path: str) -> AnalysisResult:
        res = AnalysisResult(timestamp=ts, image_path=image_path)
        cfg = self.cfg

        # ---- pH ----
        try:
            ph = estimate_ph(img, cfg)
            res.ph = None if ph is None else round(ph, 2)
            if calibration_source(cfg) == "demo" and not self._demo_note_emitted:
                res.notes.append("pH 使用演示色表，尚未用缓冲液实测标定")
                self._demo_note_emitted = True
        except Exception as exc:
            res.notes.append(f"pH 计算异常: {exc}")

        # ---- 浑浊度 ----
        try:
            ratio = self.turbidity.update(clarity_score(img, exclude=self._patch))
            res.turbidity_ratio = None if ratio is None else round(ratio, 3)
            lv = self.turbidity.level(ratio)
            if lv == "alert":
                res.alerts.append(
                    f"[高] 培养基浑浊度达基线 {ratio:.2f} 倍，疑似污染，建议立即开箱核查"
                )
            elif lv == "warn":
                res.alerts.append(f"[中] 浑浊度升至基线 {ratio:.2f} 倍，需加密观察")
        except Exception as exc:
            res.notes.append(f"浑浊度计算异常: {exc}")

        # ---- 汇合度 ----
        try:
            conf = confluence_fraction(img, cfg)
            res.confluence = round(conf, 3)

            if conf >= self._confluence_warn:
                res.notes.append(f"汇合度 {conf:.0%} 已达传代参考线")

            self._confluence_hist.append(conf)
            if len(self._confluence_hist) >= GROWTH_STALL_WINDOW:
                hist = list(self._confluence_hist)
                # 用前后两半的中位数比，而不是首尾两帧比 ——
                # 单帧的估计抖动是本方法的固有噪声，拿首尾比会天天误报。
                half = len(hist) // 2
                first = float(np.median(hist[:half]))
                second = float(np.median(hist[half:]))
                if (hist[-1] < GROWTH_STALL_MIN_CONFLUENCE
                        and second < first - GROWTH_STALL_DROP):
                    res.alerts.append(
                        f"[中] 汇合度趋势转跌（{first:.0%} → {second:.0%}），生长可能停滞"
                    )
        except Exception as exc:
            res.notes.append(f"汇合度计算异常: {exc}")

        # ---- pH 判读 ----
        if res.ph is not None:
            delta = abs(res.ph - PH_NORMAL)
            if delta >= PH_ALERT_DELTA:
                res.alerts.append(f"[高] pH {res.ph:.2f} 偏离正常值 {PH_NORMAL:.2f} 达 {delta:.2f}，培养基可能需更换")
            elif delta >= PH_WARN_DELTA:
                res.alerts.append(f"[中] pH {res.ph:.2f} 偏离 {delta:.2f}，建议关注")

        # ---- 环境参数 ----
        return res

    def step(self) -> Record:
        ts = datetime.now()
        img = self.camera.grab()
        if self.save_frames:
            path = save_frame(img, frames_dir(self.cfg), ts,
                              quality=int(self.cfg.get("capture.jpeg_quality", 92)))
        else:
            path = Path("<memory>")

        reading = self.sensor_hub.read() if self.sensor_hub is not None else SensorReading(timestamp=ts)
        reading.timestamp = ts

        result = self.analyze(img, ts, str(path))

        if result.level in ("warn", "alert"):
            summary = "；".join(result.alerts)
            self.alerter.send(result.level, "培养箱状态异常", summary, key=summary[:40])

        return Record(reading=reading, result=result)

    def run(self, loops: int = 1, interval_s: float | None = None, progress: bool = True):
        if interval_s is None:
            # 合成模式下时间是我们自己造的，没有理由真的等 ——
            # 否则 `-n 10` 会按 20 分钟的采集间隔睡上三个小时。
            if bool(self.cfg.get("capture.mock", True)):
                interval_s = 0.0
            else:
                interval_s = float(self.cfg.get("capture.interval_minutes", 20)) * 60
        records: list[Record] = []
        for i in range(loops):
            rec = self.step()
            records.append(rec)
            if progress:
                r = rec.result
                print(
                    f"[{i + 1}/{loops}] {r.timestamp:%H:%M:%S}  level={r.level:<5} "
                    f"pH={_fmt(r.ph)}  浊度×{_fmt(r.turbidity_ratio)}  汇合度={_pct(r.confluence)}"
                    + (f"  | {r.alerts[0]}" if r.alerts else "")
                )
            if i < loops - 1:
                time.sleep(interval_s)
        return records


def _fmt(v, nd=2):
    return "  --  " if v is None else f"{v:.{nd}f}"


def _pct(v):
    return "  --  " if v is None else f"{v * 100:5.1f}%"


def write_records(records: list[Record], cfg: Config) -> Path:
    """把记录追加到 CSV。首行写表头。"""
    out_dir = data_path(cfg, "analyze.output_dir", "data/processed")
    path = out_dir / "records.csv"
    rows = [r.to_row() for r in records]
    if not rows:
        return path

    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    return path


def resolve_setting(cfg: Config, key: str, value):
    """把命令行覆盖项写回配置（点号路径）。"""
    node = cfg.raw
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CellEye 培养箱智能守护 —— 主流水线")
    ap.add_argument("-c", "--config", default=None, help="配置文件路径")
    ap.add_argument("-n", "--loops", type=int, default=1, help="采集帧数")
    ap.add_argument("--interval", type=float, default=None,
                    help="帧间隔（分钟），覆盖配置。默认：合成模式 0，真机模式取 capture.interval_minutes")
    ap.add_argument("--scenario", default=None,
                    help="mock 情景：normal | growth | contamination | ph_drift")
    ap.add_argument("--no-save", action="store_true", help="不写 CSV")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.scenario:
        resolve_setting(cfg, "capture.mock_scenario", args.scenario)
    interval_s = None if args.interval is None else args.interval * 60

    mon = Monitor(cfg)
    records = mon.run(loops=args.loops, interval_s=interval_s)

    if not args.no_save:
        path = write_records(records, cfg)
        print(f"\n记录已写入: {path}")

    bad = [r for r in records if r.result.level != "ok"]
    print(f"完成：{len(records)} 帧，其中 {len(bad)} 帧异常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
