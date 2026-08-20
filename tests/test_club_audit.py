from __future__ import annotations

from golftracer.config import Config
from golftracer.session import Observation
from golftracer.tracking import _club_sanity_abstention


def _fit(*points: tuple[float, float]) -> list[Observation]:
    return [
        Observation(index, index / 60.0, x, y, source="retimed")
        for index, (x, y) in enumerate(points)
    ]


def test_club_audit_rejects_off_frame_impact_with_measurements() -> None:
    track = _club_sanity_abstention(
        _fit((100.0, 200.0), (160.0, 240.0), (-17.0, 280.0)),
        impact_t=1.0, frame_width=1080, frame_height=1920, config=Config(),
    )

    assert track is not None
    assert track.abstained is True
    assert track.reason == "impact_out_of_frame"
    assert track.audit.failures == ["phase abstained: impact_out_of_frame"]
    assert track.audit.metrics["impact_x_px"] == -17.0
    assert track.audit.metrics["impact_frame_margin_px"] == 16.0
    assert track.metadata["impact_xy"] == (-17.0, 280.0)


def test_club_audit_rejects_near_zero_impact_speed_with_measurements() -> None:
    track = _club_sanity_abstention(
        _fit(
            (500.0, 900.0), (501.0, 900.0), (502.0, 900.0),
            (503.0, 900.0), (504.0, 900.0),
        ),
        impact_t=1.0, frame_width=1080, frame_height=1920, config=Config(),
    )

    assert track is not None
    assert track.abstained is True
    assert track.reason == "impact_speed_too_low"
    assert track.audit.failures == ["phase abstained: impact_speed_too_low"]
    assert track.audit.metrics["impact_speed_px_per_frame"] == 1.0
    assert track.audit.metrics["min_impact_speed_px_per_frame"] == 30.0
    assert track.metadata["impact_speed_px_per_frame"] == 1.0


def test_club_audit_preserves_healthy_fit() -> None:
    track = _club_sanity_abstention(
        _fit(
            (300.0, 600.0), (340.0, 640.0), (390.0, 690.0),
            (450.0, 750.0), (520.0, 820.0),
        ),
        impact_t=1.0, frame_width=1080, frame_height=1920, config=Config(),
    )

    assert track is None
