from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from golftracer.config import Config
from golftracer.label.schema import (
    Label, LabelDocument, convert_v1_label_documents, load_labels, save_labels,
)
from golftracer.phases.ball import estimate_tee_tophat_frames, gate_track
from golftracer.phases.base import fit_label_constrained, monotone_viterbi
from golftracer.label.app import phase_window, selected_indices
from golftracer.tracking import _club_handoff_impact_t


def test_phase_split_refit_preserves_shared_top_cusp() -> None:
    config = Config()
    back = [
        Label(0, 0.0, 0.0, 10.0, "backswing"),
        Label(1, 0.1, 5.0, 3.0, "backswing"),
        Label(2, 0.2, 10.0, 0.0, "backswing"),
    ]
    down = [
        Label(2, 0.2, 10.0, 0.0, "downswing"),
        Label(3, 0.3, 5.0, 4.0, "downswing"),
        Label(4, 0.4, 0.0, 10.0, "downswing"),
    ]
    first = fit_label_constrained("backswing", [], back, config, observation_weight=0.65)
    second = fit_label_constrained("downswing", [], down, config, observation_weight=2.0)
    assert first is not None and second is not None
    np.testing.assert_allclose(first.xy_at_arc(first.length), second.xy_at_arc(0.0), atol=1e-6)
    left_tangent = first.xy_at_arc(first.length) - first.xy_at_arc(first.length - 0.1)
    right_tangent = second.xy_at_arc(0.1) - second.xy_at_arc(0.0)
    assert float(np.dot(left_tangent, right_tangent)) < 0.0


def test_dense_retiming_is_monotone() -> None:
    emissions = np.full((8, 14), -4.0, np.float32)
    wanted = [0, 1, 3, 3, 6, 8, 11, 13]
    for frame, arc in enumerate(wanted):
        emissions[frame, arc] = 5.0
    path = monotone_viterbi(emissions, max_step=4)
    assert np.all(np.diff(path) >= 0)
    assert path[0] == 0
    assert path[-1] == 13


def test_zero_smoothing_curve_passes_through_human_constraints() -> None:
    labels = [
        Label(0, 1.0, 10.0, 90.0, "backswing"),
        Label(4, 1.1, 45.0, 15.0, "backswing"),
        Label(8, 1.2, 90.0, 80.0, "backswing"),
    ]
    curve = fit_label_constrained(
        "backswing", [], labels, Config(), observation_weight=0.65,
        label_smoothing_px=0.0,
    )
    assert curve is not None
    for label in labels:
        np.testing.assert_allclose(
            curve.xy_at_time(label.t), (label.x, label.y), atol=1e-6,
        )


def test_full_labeller_selects_every_count_indexed_phase_frame() -> None:
    assert selected_indices(22, full=True) == list(range(22))
    start, duration = phase_window("downswing", 7.51, Config())
    assert start == pytest.approx(7.16)
    assert duration == pytest.approx(Config().downswing_pre_s)


def _synthetic_track(steps: list[float]) -> list[dict[str, float]]:
    vertical = 1200.0
    points = []
    for frame, step in enumerate(steps, 1):
        vertical -= step
        points.append({"rel_frame": frame, "u": 500.0, "v": vertical, "area": 8.0})
    return points


def test_ball_gate_rejects_280px_walker_and_accepts_600px_ball() -> None:
    config = Config()
    walker = _synthetic_track([50, 45, 40, 35, 30, 25, 20, 15, 12, 8])
    ball = _synthetic_track([105, 95, 85, 75, 65, 55, 45, 35, 25, 15])
    assert gate_track(walker, (500.0, 1300.0), config)[0] is False
    ok, reason, metrics = gate_track(ball, (500.0, 1300.0), config)
    assert ok is True, reason
    assert metrics is not None and metrics["rise"] >= 480.0


def test_tee_tophat_present_then_absent_synthetic() -> None:
    frames = np.full((48, 140, 180, 3), 50, np.uint8)
    timestamps = np.arange(len(frames), dtype=float) / 30.0
    impact = 0.70
    for index, timestamp in enumerate(timestamps):
        if timestamp < impact - 0.10:
            cv2.circle(frames[index], (90, 96), 10, (230, 230, 230), -1)
    config = Config().with_overrides(tee_roi=(0, 140, 0, 180))
    point = estimate_tee_tophat_frames(frames, timestamps, impact, config)
    assert point is not None
    assert point == pytest.approx((90.0, 96.0), abs=1.0)


def test_label_schema_round_trip(tmp_path) -> None:
    path = tmp_path / "labels.json"
    expected = LabelDocument("synthetic.mp4", 2.0, 60.0, "backswing", [
        Label(3, 2.05, 12.5, 19.5, "backswing", "corrected")
    ], time_on_task_s=7.25)
    save_labels(path, expected)
    assert load_labels(path) == expected


def test_v1_label_converter_inline_and_human_collision_wins() -> None:
    base = {
        "schema_version": 1,
        "fps": 60.0,
        "video_path": "fixture.mp4",
        "samples": [{
            "swing_id": 1, "rel_frame": 4, "frame_time_s": 10.1,
            "phase": "top", "completed": True, "skipped": False,
            "clicked": {"u": 100.0, "v": 200.0},
        }],
    }
    later = json.loads(json.dumps(base))
    later["samples"][0]["clicked"] = {"u": 101.0, "v": 201.0}
    converted = convert_v1_label_documents([base, later], swing_id=1)
    assert len(converted.labels) == 1
    assert converted.labels[0] == Label(4, 10.1, 101.0, 201.0, "backswing")


def test_precise_downswing_impact_beats_rounded_follow_window() -> None:
    precise = Label(21, 7.5134798115, 680.0, 1428.0, "downswing")
    follow = LabelDocument("fixture.mp4", 7.513, 60.0, "followthrough")
    assert _club_handoff_impact_t(7.480088667, [precise], follow, 60.0) == precise.t

    preimpact = Label(20, 76.786, 680.0, 1428.0, "downswing")
    follow = LabelDocument("fixture.mp4", 76.802, 60.0, "followthrough")
    assert _club_handoff_impact_t(76.79, [preimpact], follow, 60.0) == 76.802
