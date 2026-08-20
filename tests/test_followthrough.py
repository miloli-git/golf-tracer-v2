from __future__ import annotations

import numpy as np
import pytest

from golftracer.config import Config
from golftracer.label.schema import Label, LabelDocument, load_labels, save_labels
from golftracer.phases.followthrough import FollowthroughPhase, detect_finish
from golftracer.phases.base import dp_sweep
from golftracer.session import Observation, Swing


def _pose() -> np.ndarray:
    points = np.zeros((33, 2), np.float32)
    points[11], points[12] = (40, 20), (60, 20)
    points[15], points[16] = (42, 45), (58, 45)
    points[23], points[24] = (40, 80), (60, 80)
    return points


def test_mirrored_tracker_walks_synthetic_arc_backwards_then_returns_time_order(monkeypatch) -> None:
    frames = np.stack([np.full((120, 160, 3), index, np.uint8) for index in range(14)])
    monkeypatch.setattr(
        "golftracer.phases.followthrough._pose_series",
        lambda values, config: [_pose() for _ in values],
    )

    def endpoint(gray, background, wrist, scale, hint):
        index = int(gray[0, 0])
        return float(20 + 4 * index), float(90 - 0.4 * index * index), float(index)

    monkeypatch.setattr("golftracer.phases.followthrough._club_endpoint", endpoint)
    config = Config().with_overrides(
        follow_finish_velocity_floor_px_s=0.0,
        follow_exit_margin_px=0.0,
    )
    swing = Swing(1, 0.0, 14 / 60, 4 / 60)
    phase = FollowthroughPhase(config, impact_point=(36.0, 83.6))
    observations = phase.track(frames, swing, config)
    assert [item.frame_index for item in observations] == list(range(10))
    assert observations[0].x == pytest.approx(36.0)
    assert observations[-1].x == pytest.approx(72.0)
    assert all(a.t < b.t for a, b in zip(observations, observations[1:]))


def test_finish_detection_velocity_floor_and_frame_exit() -> None:
    moving = [Observation(i, i / 60, float(i * 8), 60.0, source="observed") for i in range(22)]
    stationary = [
        Observation(i, i / 60, 176.0 + 0.1 * (i - 22), 60.0, source="observed")
        for i in range(22, 31)
    ]
    config = Config().with_overrides(
        follow_finish_min_time_s=0.0,
        follow_finish_stationary_frames=8,
        follow_finish_velocity_floor_px_s=45.0,
    )
    finish = detect_finish(
        [*moving, *stationary], frame_count=40, frame_shape=(200, 300), fps=60.0,
        config=config,
    )
    assert finish is not None
    assert finish.reason == "velocity_floor"
    assert finish.frame_index == 30

    exit_track = [Observation(i, i / 60, 100.0 + i, 100.0, source="observed") for i in range(12)]
    finish = detect_finish(
        exit_track, frame_count=20, frame_shape=(200, 300), fps=60.0,
        config=config,
    )
    assert finish is not None
    assert finish.reason == "frame_exit"
    assert finish.frame_index == 11


def test_followthrough_dp_can_inherit_fast_impact_velocity() -> None:
    emissions = np.zeros((4, 30), np.float32)
    inherited = dp_sweep(
        emissions, 0, 12, 0.0002, initial_step=10,
        initial_step_weight=0.1, initial_step_frames=3,
    )
    default = dp_sweep(emissions, 0, 12, 0.0002)
    assert inherited[1] >= 8
    assert np.all(np.diff(inherited)[:3] >= 8)
    assert default[1] == 0


def test_followthrough_fit_starts_at_exact_downswing_impact() -> None:
    config = Config()
    impact = (80.0, 150.0)
    phase = FollowthroughPhase(config, impact_point=impact)
    phase.impact_t = 2.0
    phase.finish_t = 2.4
    labels = [
        Label(12, 2.2, 55.0, 70.0, "followthrough"),
        Label(24, 2.4, 95.0, 35.0, "followthrough"),
    ]
    curve = phase.fit([], labels)
    assert curve is not None
    np.testing.assert_allclose(curve.xy_at_arc(0.0), impact, atol=1e-6)


def test_unlabelled_followthrough_can_fit_anchor_plus_tracker_evidence() -> None:
    phase = FollowthroughPhase(Config(), impact_point=(80.0, 150.0))
    phase.impact_t = 2.0
    phase.finish_t = 2.2
    observations = [
        Observation(1, 2.0 + 1 / 60, 70.0, 120.0, source="observed"),
        Observation(6, 2.1, 45.0, 60.0, source="observed"),
        Observation(12, 2.2, 90.0, 30.0, source="observed"),
    ]
    curve = phase.fit(observations, [])
    assert curve is not None
    np.testing.assert_allclose(curve.xy_at_arc(0.0), (80.0, 150.0), atol=1e-6)


def test_followthrough_label_round_trip(tmp_path) -> None:
    path = tmp_path / "4.followthrough.json"
    expected = LabelDocument("fixture.mp4", 3.25, 60.0, "followthrough", [
        Label(0, 3.25, 20.0, 80.0, "followthrough"),
        Label(10, 3.25 + 10 / 60, 55.0, 30.0, "followthrough"),
    ])
    save_labels(path, expected)
    assert load_labels(path) == expected
