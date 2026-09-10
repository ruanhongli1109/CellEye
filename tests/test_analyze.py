"""分析链路的回归测试。

这些测试全部基于合成图像，所以不需要任何硬件、几秒钟跑完，
可以放心挂到 CI 上。它们守住的是一条底线：
**算法改动之后，浑浊度还能反映浑浊、汇合度还能反映覆盖、颜色还能反推出 pH。**
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analyze import TurbidityTracker, clarity_score, confluence_threshold, estimate_ph
from src.analyze.ph_color import ph_from_color
from src.capture import phenol_red_rgb, synth_frame
from src.config import load_config

PATCH = (480, 20, 80, 80)
SIZE = (400, 600)
NO_CALIB = "data/processed/__no_such_calibration__.json"


@pytest.fixture
def cfg():
    c = load_config()
    ph = c.raw["analyze"]["ph"]
    ph["reference_patch"] = list(PATCH)
    ph["medium_roi"] = [0.28, 0.28, 0.44, 0.44]
    ph["calibration_file"] = NO_CALIB  # 强制回落到演示色表，避免依赖本地标定产物
    return c


def frame(**kw):
    kw.setdefault("size", SIZE)
    kw.setdefault("reference_patch", PATCH)
    kw.setdefault("illumination", 1.0)
    return synth_frame(**kw)


# ---------------- 色表本身 ----------------

def test_phenol_red_direction():
    """酚红：偏酸偏黄（绿通道高），偏碱偏紫（蓝通道高）。"""
    acid = phenol_red_rgb(6.4)
    alkaline = phenol_red_rgb(7.8)
    assert acid[1] > alkaline[1], "偏酸应当更黄（绿通道更高）"
    assert alkaline[2] > acid[2], "偏碱应当更紫（蓝通道更高）"


def test_ph_from_color_exact():
    """颜色恰好等于色表点时，应精确还原 pH。"""
    for ph, rgb in [(6.4, (235, 190, 70)), (7.2, (215, 100, 95)), (7.8, (185, 80, 145))]:
        bgr = np.array(rgb[::-1], dtype=np.float32)
        pts = [(6.4, np.array([235, 190, 70], np.float32)),
               (7.2, np.array([215, 100, 95], np.float32)),
               (7.8, np.array([185, 80, 145], np.float32))]
        assert ph_from_color(bgr, pts) == pytest.approx(ph, abs=0.01)


# ---------------- 汇合度 ----------------

def test_confluence_tracks_coverage(cfg):
    """汇合度估计应随真实覆盖面积单调上升。"""
    ests = [confluence_threshold(frame(confluence=c, turbidity=0.0), exclude=PATCH)
            for c in (0.10, 0.35, 0.65)]
    assert ests[0] < ests[1] < ests[2], f"应单调递增，实际 {ests}"
    for target, est in zip((0.10, 0.35, 0.65), ests):
        assert est == pytest.approx(target, abs=0.15), f"目标 {target}，估计 {est}"


def test_confluence_ignores_reference_patch(cfg):
    """标准白卡不能被算成细胞。"""
    with_patch = confluence_threshold(frame(confluence=0.0, turbidity=0.0), exclude=PATCH)
    without = confluence_threshold(frame(confluence=0.0, turbidity=0.0), exclude=None)
    assert with_patch < 0.05, "空白视野的汇合度应接近 0"
    assert without > with_patch, "不挖掉白卡时会被误算成前景"


# ---------------- 浑浊度 ----------------

def test_clarity_drops_with_turbidity(cfg):
    """清晰度必须随浑浊度**严格单调下降**。

    这里特意不写"下降 2 倍"之类的魔数断言 —— 真实比值约 1.75，
    写死一个拍脑袋的倍数是自欺欺人。单调性才是这个指标真正的契约。
    """
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
    scores = [clarity_score(frame(confluence=0.35, turbidity=t), exclude=PATCH) for t in levels]
    for a, b, ta, tb in zip(scores, scores[1:], levels, levels[1:]):
        assert a > b, f"浑浊度 {ta}→{tb} 时清晰度应下降，实际 {a:.2f} → {b:.2f}"
    assert scores[0] > scores[-1] * 1.4, f"端点落差过小：{scores[0]:.2f} → {scores[-1]:.2f}"


def test_confluence_robust_to_turbidity(cfg):
    """浑浊不该被误读成细胞覆盖 —— 两个指标必须解耦。"""
    ests = [confluence_threshold(frame(confluence=0.35, turbidity=t), exclude=PATCH)
            for t in (0.0, 0.3, 0.6)]
    assert max(ests) - min(ests) < 0.08, f"浑浊度不应显著影响汇合度估计，实际 {ests}"


def test_turbidity_tracker_baseline_and_ratio(cfg):
    tr = TurbidityTracker(cfg)
    for _ in range(tr.baseline_frames):
        assert tr.update(100.0) is None, "基线建立期不应给出倍数"
    assert tr.baseline == pytest.approx(100.0)
    assert tr.update(50.0) == pytest.approx(2.0)
    assert tr.update(100.0) == pytest.approx(1.0)


def test_turbidity_levels(cfg):
    tr = TurbidityTracker(cfg)
    for _ in range(tr.baseline_frames):
        tr.update(100.0)
    assert tr.level(1.00) == "ok"
    assert tr.level(cfg.get("analyze.turbidity.warn_ratio")) == "warn"
    assert tr.level(cfg.get("analyze.turbidity.alert_ratio")) == "alert"
    assert tr.level(None) == "unknown"


def test_contamination_scenario_triggers_alert(cfg):
    """端到端的意义所在：污染情景必须真的报出来，正常情景不许乱报。"""
    from src.pipeline import Monitor
    from src.pipeline import resolve_setting

    def run(scenario):
        c = load_config()
        resolve_setting(c, "capture.mock_scenario", scenario)
        resolve_setting(c, "capture.resolution", [360, 640])
        resolve_setting(c, "alert.enabled", False)
        mon = Monitor(c, save_frames=False)
        # 30 分钟 × 40 帧 = 20 小时，污染情景设计的就是这个时间尺度
        resolve_setting(c, "capture.interval_minutes", 30)
        return mon.run(loops=40, interval_s=0, progress=False)

    polluted = [r.result.turbidity_ratio for r in run("contamination")]
    polluted = [v for v in polluted if v is not None]
    assert max(polluted) > 1.6, f"污染情景浊度应显著超过基线，实际最大 {max(polluted):.2f}"

    normal = [r.result.level for r in run("normal")]
    assert "alert" not in normal, "正常情景不应产生高等级告警"


# ---------------- pH ----------------

@pytest.mark.parametrize("ph_true", [6.4, 7.0, 7.4, 7.8])
def test_estimate_ph_pipeline(cfg, ph_true):
    """整条链路（白平衡 → 取样 → 查表）应能大致还原 pH。"""
    img = frame(confluence=0.0, turbidity=0.0, ph=ph_true)
    est = estimate_ph(img, cfg)
    assert est is not None
    assert est == pytest.approx(ph_true, abs=0.35), f"真实 {ph_true}，估计 {est}"


@pytest.mark.parametrize("illum", [0.80, 1.25])
def test_white_balance_cancels_illumination_drift(cfg, illum):
    """光照漂移 ±25% 不应被误读成 pH 变化。"""
    img = frame(confluence=0.0, turbidity=0.0, ph=7.20, illumination=illum)
    est = estimate_ph(img, cfg)
    assert est == pytest.approx(7.20, abs=0.35), f"光照 {illum} 下估计为 {est}"


def test_estimate_ph_returns_none_without_points(cfg):
    assert ph_from_color(np.array([1.0, 2.0, 3.0]), []) is None
