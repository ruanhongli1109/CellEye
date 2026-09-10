"""用已知 pH 的缓冲液标定比色色表。

流程（就是实验里"配标准、拍照片、建曲线"那套）：

1. 配一组已知 pH 的缓冲液（建议 6.40 / 6.80 / 7.00 / 7.20 / 7.40 / 7.80），
   补足培养基成分使酚红浓度与真实培养一致 —— 否则标出来的色表用不上。
2. 在**实际使用的光路和光照条件**下，每档拍 3~5 张。
3. 文件命名带上 pH，例如 `ph_7.20_01.jpg`。
4. 运行本脚本，得到 data/processed/ph_calibration.json。

标定必须在最终光路下做 —— 换了光源或相机位置就要重新标。

用法：
    python scripts/calibrate_ph.py --dir data/raw/calibration
    python scripts/calibrate_ph.py --dir ... --report   # 输出留一交叉验证误差
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyze.ph_color import medium_color, ph_from_color, white_balance  # noqa: E402
from src.config import ROOT, load_config, resolve  # noqa: E402
from src.imgio import imread  # noqa: E402

PH_IN_NAME = re.compile(r"ph[_-]?(\d+(?:\.\d+)?)", re.IGNORECASE)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_ph(path: Path) -> float | None:
    m = PH_IN_NAME.search(path.stem)
    return float(m.group(1)) if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="酚红比色 pH 标定")
    ap.add_argument("--dir", required=True, help="标定图像目录")
    ap.add_argument("--out", default=None, help="输出 json 路径（默认按配置）")
    ap.add_argument("--report", action="store_true", help="打印留一交叉验证误差")
    args = ap.parse_args(argv)

    cfg = load_config()
    src_dir = resolve(args.dir)
    if not src_dir.exists():
        print(f"目录不存在: {src_dir}")
        return 1

    patch = tuple(cfg.get("analyze.ph.reference_patch", [1620, 40, 120, 120]))
    roi = tuple(cfg.get("analyze.ph.medium_roi", [0.30, 0.30, 0.40, 0.40]))

    # 按 pH 归组，同一 pH 的多张取中位色，压掉单张的偶然误差
    groups: dict[float, list[np.ndarray]] = {}
    skipped = 0
    for p in sorted(src_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        ph = parse_ph(p)
        if ph is None:
            skipped += 1
            continue
        img = imread(p)
        if img is None:
            skipped += 1
            continue
        groups.setdefault(ph, []).append(medium_color(white_balance(img, patch), roi))

    if not groups:
        print("没有解析到任何带 pH 的图像。文件名需形如 ph_7.20_01.jpg")
        return 1

    points = []
    for ph in sorted(groups):
        colors = np.vstack(groups[ph])
        med = np.median(colors, axis=0)
        points.append({"ph": ph, "rgb": [float(v) for v in med[::-1]], "n": len(colors)})
        print(f"pH {ph:.2f}  样本 {len(colors):>2} 张  RGB=({med[2]:.1f}, {med[1]:.1f}, {med[0]:.1f})")

    if skipped:
        print(f"（跳过 {skipped} 个文件名不含 pH 的文件）")

    out_path = Path(resolve(args.out)) if args.out else resolve(
        cfg.get("analyze.ph.calibration_file", "data/processed/ph_calibration.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created": datetime.now().isoformat(timespec="seconds"),
                "reference_patch": list(patch),
                "medium_roi": list(roi),
                "points": points,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n标定文件已写入: {out_path}")

    if args.report:
        pts = [(p["ph"], np.array(p["rgb"], dtype=np.float32)) for p in points]
        preds, errs = [], []
        for i in range(len(pts)):
            rest = pts[:i] + pts[i + 1 :]
            pred = ph_from_color(np.array(pts[i][1], dtype=np.float32)[::-1], rest)
            preds.append(pred)
            errs.append(None if pred is None else abs(pred - pts[i][0]))

        print("\n留一交叉验证（用其余点推当前点）：")
        for (ph, _), pred, err in zip(pts, preds, errs):
            flag = ""
            if pred is None:
                flag = "  ← 无法外推"
            elif ph in (pts[0][0], pts[-1][0]):
                flag = "  ← 端点，插值法无法外推"
            print(f"  真实 {ph:.2f}  预测 {'--' if pred is None else f'{pred:.2f}'}"
                  f"  误差 {'--' if err is None else f'{err:.3f}'}{flag}")

        # 端点必然外推失败，把它算进 MAE 只会得到一个虚假的差评。
        # 真正有意义的是**内点**误差 —— 那才是实际使用区间。
        interior = [e for (ph, _), e in zip(pts, errs)
                    if e is not None and ph not in (pts[0][0], pts[-1][0])]
        if interior:
            print(f"\n内点平均绝对误差 MAE = {np.mean(interior):.3f} pH（目标 ≤ 0.10）")
            print(f"内点最大误差       = {max(interior):.3f} pH")
        print("提示：端点处的留一误差是插值法无法外推造成的假象，不代表真实精度。")
        print("      要验证端点，请在该 pH 附近补一个标定点。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
