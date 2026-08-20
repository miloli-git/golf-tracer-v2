from __future__ import annotations

import numpy as np

from golftracer.avoffset import departure_time, measure_av_offset, summary_lines
from golftracer.config import Config


def _signal(fps: float, depart_index: int, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(n) / fps
    signal = np.full(n, 180.0)
    signal[depart_index:] = 60.0
    return signal, timestamps


def test_departure_time_finds_first_persistently_absent_frame() -> None:
    signal, timestamps = _signal(60.0, 70)
    provisional = timestamps[66]  # 4 frames early, like issue #5
    t_dep, contrast, reason = departure_time(signal, timestamps, provisional)
    assert reason is None
    assert contrast == 120.0
    assert t_dep == timestamps[70]


def test_departure_time_ignores_one_frame_occlusion() -> None:
    signal, timestamps = _signal(60.0, 70)
    signal[62] = 60.0  # club shadow crossing the tee for one frame
    t_dep, _, reason = departure_time(signal, timestamps, timestamps[66])
    assert reason is None
    assert t_dep == timestamps[70]


def test_departure_time_rejects_low_contrast_and_missing_departure() -> None:
    timestamps = np.arange(120) / 60.0
    flat = np.full(120, 100.0)
    assert departure_time(flat, timestamps, timestamps[60])[2] == "no_contrast"
    signal, timestamps = _signal(60.0, 95)  # departs after the search window but inside the post span
    assert departure_time(signal, timestamps, timestamps[60])[2] == "no_departure_in_window"


def test_measure_av_offset_reports_lag_corrected_estimate(monkeypatch) -> None:
    from golftracer import avoffset

    def fake_measure(video, t_audio, config, roi=None):
        return avoffset.ImpactOffset(t_audio, t_audio - 0.19, (1.0, 2.0), t_audio - 0.10, 0.10, 90.0, None)

    monkeypatch.setattr(avoffset, "measure_impact_offset", fake_measure)
    summary = measure_av_offset("clip.mp4", [10.0, 20.0], Config())
    assert summary["measured"] == 2
    assert summary["median_offset_s"] == 0.1
    assert summary["impact_offset_estimate_s"] == round(0.1 + Config().av_departure_lag_s, 4)
    assert any("--av-offset" in line for line in summary_lines(summary))


def test_apply_measured_av_offset_rewrites_t_video(monkeypatch) -> None:
    from golftracer import avoffset

    def fake_measure(video, t_audio, config, roi=None):
        return avoffset.ImpactOffset(t_audio, t_audio - 0.19, (1.0, 2.0), t_audio - 0.10, 0.10, 90.0, None)

    monkeypatch.setattr(avoffset, "measure_impact_offset", fake_measure)
    rows = [{"t_audio": 10.0, "t_video": 9.81}, {"t_audio": 20.0, "t_video": 19.81}]
    updated, config, summary = avoffset.apply_measured_av_offset("clip.mp4", rows, Config())
    assert summary["applied"] is True
    estimate = round(0.10 + Config().av_departure_lag_s, 4)
    assert config.av_offset_s == estimate
    assert updated[0]["t_video"] == round(10.0 - estimate, 6)
    assert updated[0]["av_offset_applied_s"] == estimate
    assert rows[0]["t_video"] == 9.81  # input untouched


def test_cli_accepts_auto_av_offset() -> None:
    from golftracer.cli import build_parser

    args = build_parser().parse_args(["track", "clip.mp4", "--out", "x", "--av-offset", "auto"])
    assert args.av_offset == "auto"
    args = build_parser().parse_args(["track", "clip.mp4", "--out", "x", "--av-offset", "0.12"])
    assert args.av_offset == 0.12
