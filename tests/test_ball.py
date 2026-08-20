from __future__ import annotations

import numpy as np
import pytest

from golftracer.config import Config
from golftracer.label.schema import Label, LabelDocument, save_labels
from golftracer.phases import ball
from golftracer.render.compositor import fit_track
from golftracer.session import Observation, Session, Swing
from golftracer.tracking import _ball_track


def _flight_points(vertical_steps: list[float], *, angle_deg: float = 0.0) -> list[dict]:
    tee = np.asarray((640.0, 1400.0))
    vertical = 100.0 + np.r_[0.0, np.cumsum(vertical_steps)]
    tangent = np.tan(np.radians(angle_deg))
    return [
        {
            "rel_frame": index + 1,
            "u": float(tee[0] + rise * tangent),
            "v": float(tee[1] - rise),
            "t_s": index / 60.0,
            "area": 12.0,
        }
        for index, rise in enumerate(vertical)
    ]


def test_vote_winner_on_synthetic_ray() -> None:
    config = Config()
    tee = (640.0, 1400.0)
    true_points = _flight_points([80, 75, 70, 65, 60, 55, 50, 45, 40], angle_deg=10.0)
    observations = np.asarray([
        row
        for point in true_points
        for row in (
            (point["rel_frame"], point["u"], point["v"], point["area"]),
            (point["rel_frame"], 640.0 - (1400.0 - point["v"]) * 0.30, point["v"] + 7.0, 160.0),
        )
    ], dtype=float)
    debug: dict = {}
    selected, reason, metrics = ball.launch_vote(
        observations, tee, 10.0, config, debug=debug,
    )
    assert reason is None
    assert selected is not None and metrics is not None
    assert [item["rel_frame"] for item in selected] == list(range(1, 11))
    assert all(item["u"] > tee[0] for item in selected)
    # Equal-scoring windows retain the first fixed-grid centre that contains
    # the ray: 9.0 is the first 1.25-degree window to include 10 degrees.
    assert debug["winner"]["centre_deg"] == 9.0
    assert debug["winner"]["half_width_deg"] == 1.25


def test_gates_reject_walker_but_accept_ball() -> None:
    config = Config()
    tee = (640.0, 1400.0)
    struck_ball = _flight_points([80, 75, 70, 65, 60, 55, 50, 45, 40])
    walker = _flight_points([40, 38, 36, 34, 32, 30, 28, 26, 24])
    ok, reason, _ = ball.gate_track(struck_ball, tee, config)
    assert ok and reason is None
    ok, reason, _ = ball.gate_track(walker, tee, config)
    assert not ok and reason == "rise_too_small"


def test_longest_rising_preserves_gapped_subsequence() -> None:
    frames = np.asarray([1, 2, 7, 9, 22])
    vertical = np.asarray([100.0, 90.0, 80.0, 70.0, 60.0])
    assert ball._longest_rising(frames, vertical, max_gap=5).tolist() == [0, 1, 2, 3]


def test_session_tee_table_remeasures_outlier_and_fills_neighbour(monkeypatch) -> None:
    impacts = [1.0, 2.0, 3.0, 4.0]
    first = {
        1.0: (100.0, 100.0),
        2.0: None,
        3.0: (110.0, 90.0),
        4.0: (500.0, 500.0),
    }

    def fake_read(*_args, **_kwargs):
        return np.zeros((20, 8, 8, 3), np.uint8), np.arange(20, dtype=float)

    def fake_estimate(_frames, _timestamps, impact, _config, *, roi, prior_xy=None):
        del roi
        if prior_xy is not None and impact == 4.0:
            return (120.0, 95.0)
        return first[impact]

    monkeypatch.setattr(ball, "read_window_pts", fake_read)
    monkeypatch.setattr(ball, "estimate_tee_frames", fake_estimate)
    table, provenance = ball.estimate_session_tees(
        "fixture.mp4", impacts, Config().with_overrides(tee_method="v1"),
        roi=(0, 8, 0, 8), return_provenance=True,
    )
    assert table[4.0] == (120.0, 95.0)
    assert provenance[4.0]["source"] == "remeasured_with_prior"
    assert table[2.0] == (105.0, 95.0)
    assert provenance[2.0]["source"] == "neighbour_fill"
    assert provenance[2.0]["neighbours"] == [1.0, 3.0]


def test_tee_estimator_selected_by_config(monkeypatch) -> None:
    frames = np.zeros((20, 8, 8, 3), np.uint8)
    timestamps = np.arange(20, dtype=float)
    monkeypatch.setattr(ball, "estimate_tee_v1_frames", lambda *_args, **_kwargs: (1.0, 2.0))
    monkeypatch.setattr(ball, "estimate_tee_tophat_frames", lambda *_args, **_kwargs: (3.0, 4.0))
    assert ball.estimate_tee_frames(
        frames, timestamps, 10.0, Config().with_overrides(tee_method="v1"),
    ) == (1.0, 2.0)
    assert ball.estimate_tee_frames(
        frames, timestamps, 10.0, Config().with_overrides(tee_method="tophat"),
    ) == (3.0, 4.0)


def test_derived_tee_roi_keeps_the_hitting_side() -> None:
    frames = np.zeros((20, 100, 100, 3), np.uint8)
    # A golfer-sized moving component is left of the ball at impact.
    frames[15, 35:90, 15:35] = 255
    roi = ball.derive_tee_roi(frames, 10, Config())
    assert roi[3] >= 85
    assert roi[3] - roi[2] >= 70


def test_ball_phase_impact_source_is_swing_not_club_handoff(monkeypatch) -> None:
    seen: list[float] = []

    def fake_track_video(self, _video, swing, _config):
        seen.append(swing.impact_t)
        return []

    monkeypatch.setattr(ball.BallPhase, "track_video", fake_track_video)
    swing = Swing(1, 0.0, 2.0, 1.25)
    session = Session("fixture.mp4", 100, 100, 60.0, 2.0, 0, swings=[swing])
    _ball_track(session, swing, Config(), None, tee_xy=(50.0, 80.0))
    assert seen == [1.25]


def test_ball_labels_are_exact_fit_and_render_constraints(tmp_path, monkeypatch) -> None:
    raw = [
        Observation(frame, 10.0 + frame / 60.0, 400.0 + frame * 2.0,
                    1300.0 - frame * 24.0, source="observed")
        for frame in (1, 3, 5, 7, 9, 11)
    ]
    labels = [
        Label(7, 10.0 + 7 / 60.0, 455.0, 1090.0, "ball"),
        Label(11, 10.0 + 11 / 60.0, 510.0, 940.0, "ball"),
    ]
    save_labels(
        tmp_path / "1.ball.json",
        LabelDocument("fixture.mp4", 10.0, 60.0, "ball", labels),
    )

    def fake_track_video(self, _video, _swing, _config):
        self.abstained = False
        self.reason = None
        return raw

    monkeypatch.setattr(ball.BallPhase, "track_video", fake_track_video)
    swing = Swing(1, 9.0, 12.0, 10.0)
    session = Session("fixture.mp4", 1080, 1920, 60.0, 12.0, 0, swings=[swing])
    track = _ball_track(
        session, swing, Config(), None, tee_xy=(400.0, 1400.0),
        labels_root=tmp_path,
    )

    assert track.metadata["label_constraint_mode"] == "exact_residual_correction"
    assert track.audit.passed
    assert track.audit.metrics["max_label_residual_px"] <= 1e-9
    fitted = fit_track(track, Config())
    assert fitted is not None
    for label in labels:
        point = fitted.pieces[0].xy(label.t)
        assert point == pytest.approx((label.x, label.y), abs=1e-3)
    assert next(item for item in track.observations if item.frame_index == 7).y != raw[3].y


def test_ball_track_without_labels_is_unchanged(monkeypatch) -> None:
    raw = [
        Observation(1, 2.01, 100.0, 500.0, source="observed"),
        Observation(4, 2.04, 105.0, 420.0, source="observed"),
        Observation(8, 2.08, 112.0, 330.0, source="observed"),
    ]

    def fake_track_video(self, _video, _swing, _config):
        self.abstained = False
        self.reason = None
        self.metrics = {"n_observed": 3}
        return raw

    monkeypatch.setattr(ball.BallPhase, "track_video", fake_track_video)
    swing = Swing(1, 1.0, 3.0, 2.0)
    session = Session("fixture.mp4", 100, 100, 60.0, 3.0, 0, swings=[swing])
    track = _ball_track(session, swing, Config(), None, tee_xy=(50.0, 80.0))

    assert track.observations == raw
    assert track.metadata == {
        "tee_xy": (50.0, 80.0), "n_observed": 3,
        "shaft_rule_fired": False,
    }
    assert track.audit.metrics == {"frames": 3.0, "abstained": 0.0}
    assert "label_constrained" not in track.metadata


def test_drop_static_repeats_keeps_pairs_drops_dwells() -> None:
    from golftracer.config import Config
    from golftracer.phases.ball import _drop_static_repeats

    def point(frame: int, u: float, v: float) -> dict[str, float]:
        return {"rel_frame": frame, "t_s": frame / 60.0, "u": u, "v": v}

    config = Config()
    # a genuine near-apex pair (2 repeats) survives
    pair = [point(1, 10.0, 100.0), point(2, 10.0, 100.0), point(3, 20.0, 90.0)]
    kept, dropped = _drop_static_repeats(pair, config)
    assert dropped == 0 and len(kept) == 3
    # a 4-frame dwell keeps the first two and drops the rest (gappy-track dwell loop case)
    dwell = [
        point(15, 549.4, 685.7), point(17, 549.4, 685.7),
        point(18, 549.4, 685.7), point(27, 549.4, 685.7),
        point(31, 436.7, 84.0),
    ]
    kept, dropped = _drop_static_repeats(dwell, config)
    assert dropped == 2
    assert [item["rel_frame"] for item in kept] == [15, 17, 31]
    # disabled by config
    kept, dropped = _drop_static_repeats(dwell, config.with_overrides(ball_max_coord_repeats=0))
    assert dropped == 0 and len(kept) == 5
