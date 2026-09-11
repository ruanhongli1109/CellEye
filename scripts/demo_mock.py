"""端到端演示：不需要任何硬件，直接跑通「采集 → 分析 → 判定 → 告警」。

这是仓库建立第一周就能拿出的东西 —— 硬件还在路上，但整条算法链路已经可以
被任何人 clone 下来复现。

运行：
    python scripts/demo_mock.py

产出：
    docs/figures/demo_scenarios.png   四种情景的指标曲线对比
    （stdout 上的对照表与告警日志）
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, load_config  # noqa: E402
from src.pipeline import Monitor, resolve_setting  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SCENARIOS = {
    "normal": "正常生长",
    "growth": "快速生长（需传代）",
    "contamination": "污染发生",
    "ph_drift": "pH 漂移",
}
FRAMES = 48
INTERVAL_MIN = 30          # 48 帧 × 30 min = 24 h
DEMO_RESOLUTION = [540, 960]   # 演示用小图，跑得快


def run_scenario(name: str):
    cfg = load_config()
    resolve_setting(cfg, "capture.mock_scenario", name)
    resolve_setting(cfg, "capture.resolution", DEMO_RESOLUTION)
    resolve_setting(cfg, "capture.interval_minutes", INTERVAL_MIN)
    resolve_setting(cfg, "alert.enabled", False)

    mon = Monitor(cfg, save_frames=False)
    return mon.run(loops=FRAMES, interval_s=0, progress=False), cfg


def main() -> int:
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 8.6), dpi=200, sharex=True)
    # 藏蓝 + 香槟金为主，红只留给"真出事"的那条曲线
    NAVY, TEAL, RED, GOLD = "#1F2A44", "#26655F", "#B23A2E", "#B8891F"
    colors = [NAVY, TEAL, RED, GOLD]

    print("=" * 78)
    print("CellEye 端到端演示（合成图像，无硬件）")
    print("=" * 78)
    print(f"{'情景':<20}{'终点pH':>8}{'终点浊度×':>12}{'终点汇合度':>12}{'告警':>8}")
    print("-" * 78)

    for (name, label), color in zip(SCENARIOS.items(), colors):
        records, _ = run_scenario(name)
        hours = [i * INTERVAL_MIN / 60 for i in range(len(records))]

        ph = [r.result.ph for r in records]
        turb = [r.result.turbidity_ratio for r in records]
        conf = [r.result.confluence for r in records]
        n_alert = sum(1 for r in records if r.result.level != "ok")

        axes[0].plot(hours, conf, color=color, lw=2.0, label=label,
                     solid_capstyle="round")
        axes[1].plot(hours, turb, color=color, lw=2.0, label=label,
                     solid_capstyle="round")
        axes[2].plot(hours, ph, color=color, lw=2.0, label=label,
                     solid_capstyle="round")

        last = lambda v: "  --  " if v[-1] is None else f"{v[-1]:.2f}"  # noqa: E731
        print(f"{label:<20}{last(ph):>8}{last(turb):>12}"
              f"{'  --  ' if conf[-1] is None else f'{conf[-1] * 100:.1f}%':>12}{n_alert:>8}")

    print("-" * 78)

    axes[0].set_ylabel("汇合度", fontsize=10.5, color=NAVY)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.85, ls=(0, (5, 3)), lw=1.1, color="#B9C1CE", zorder=0)
    axes[0].text(5.9, 0.87, "传代参考线 85%", fontsize=8.6, color="#8B95A5",
                 ha="right", va="top")

    axes[1].set_ylabel("浑浊度（相对基线倍数）", fontsize=10.5, color=NAVY)
    axes[1].set_ylim(0.7, 2.35)
    axes[1].axhline(1.30, ls=(0, (5, 3)), lw=1.1, color="#D8B25E", zorder=0)
    axes[1].axhline(1.60, ls=(0, (5, 3)), lw=1.1, color="#C97A6E", zorder=0)
    axes[1].text(0.12, 1.63, "高等级告警 1.60×", fontsize=8.6, color="#B23A2E")
    axes[1].text(0.12, 1.33, "提醒 1.30×", fontsize=8.6, color="#A8842E")

    axes[2].set_ylabel("pH", fontsize=10.5, color=NAVY)
    axes[2].axhspan(6.90, 7.50, color=TEAL, alpha=0.07, zorder=0)
    axes[2].text(0.12, 7.42, "正常区间 6.90–7.50", fontsize=8.6, color=TEAL)
    axes[2].set_xlabel("培养时间（小时）", fontsize=10.5, color=NAVY)

    for ax in axes:
        ax.grid(axis="y", color="#EDF1F6", lw=0.9, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#C7CFDB")
        ax.tick_params(colors="#5A6472", labelsize=9.5, length=0)

    axes[0].legend(loc="upper left", ncol=4, fontsize=9.6, frameon=False,
                   handlelength=1.6, columnspacing=1.6)
    fig.suptitle("四种合成情景 · 24 小时指标走势", fontsize=15,
                 fontweight="bold", color=NAVY, x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.945,
             "合成图像演示，不含真实实验数据。注意中图：只有「污染发生」的浑浊度冲到基线 2 倍，"
             "其余三种始终平在 1.0 附近 —— 这正是告警要抓的东西。",
             fontsize=9.2, color="#707C90", ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.925))

    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "demo_scenarios.png", facecolor="white")
    print(f"图已保存: {out / 'demo_scenarios.png'}")
    print("\n注：pH 目前使用演示色表，尚未用缓冲液实测标定（见 README「当前状态」）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
