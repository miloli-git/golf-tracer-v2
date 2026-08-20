from __future__ import annotations

from pathlib import Path

from golftracer.config import Config
from golftracer.decode import probe
from golftracer.session import Session, Swing
from golftracer.tracking import (
    _SwingWorker, _WorkerTask, _apply_worker_swing, _record_swing_failure,
    _session_shell, track_session,
)


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.mp4"


def _session() -> Session:
    meta = probe(FIXTURE)
    return Session(
        str(FIXTURE.resolve()), meta.width, meta.height, meta.fps,
        meta.duration, meta.rotation,
        swings=[Swing(1, 0.0, 0.9, 0.5), Swing(2, 0.0, 0.9, 0.7)],
    )


def _failure_track(swing: Swing):
    return next(track for track in swing.tracks if track.phase == "swing")


def test_unexpected_swing_exception_abstains_and_session_continues(tmp_path: Path) -> None:
    session = _session()
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "1.backswing.json").write_text("{not-json", encoding="utf-8")

    track_session(
        session,
        Config().with_overrides(pose_enabled=False, track_swing_timeout_s=30.0),
        phase="backswing",
        labels_root=labels,
    )

    failed, survived = session.swings
    assert failed.metadata["tracking_failure_reason"] == "unexpected_exception"
    assert failed.metadata["tracking_failure_stage"] == "club"
    assert _failure_track(failed).reason == "unexpected_exception"
    assert _failure_track(failed).audit.failures == [
        "phase abstained: unexpected_exception"
    ]
    assert all(track.phase != "swing" for track in survived.tracks)

    path = session.to_json(tmp_path / "session.json")
    restored = Session.from_json(path)
    assert [swing.id for swing in restored.swings] == [1, 2]
    assert _failure_track(restored.swings[0]).abstained is True
    assert all(track.phase != "swing" for track in restored.swings[1].tracks)


def test_timeout_terminates_worker_then_next_swing_processes_and_serializes(
    tmp_path: Path,
) -> None:
    session = _session()
    config = Config().with_overrides(pose_enabled=False)
    shell = _session_shell(session)
    worker = _SwingWorker()
    try:
        timed_out = worker.run(_WorkerTask(
            "club", shell, config, swing=session.swings[0],
            selected=("backswing",),
        ), timeout_s=0.0)
        assert timed_out.failure is not None
        assert timed_out.failure.reason == "timeout"
        _record_swing_failure(session.swings[0], timed_out.failure)

        survived = worker.run(_WorkerTask(
            "club", shell, config, swing=session.swings[1],
            selected=("backswing",),
        ), timeout_s=30.0)
        assert survived.failure is None
        assert isinstance(survived.value, Swing)
        _apply_worker_swing(session.swings[1], survived.value)
    finally:
        worker.close()

    path = session.to_json(tmp_path / "session.json")
    restored = Session.from_json(path)
    assert _failure_track(restored.swings[0]).reason == "timeout"
    assert restored.swings[0].metadata["tracking_failure_stage"] == "club"
    assert all(track.phase != "swing" for track in restored.swings[1].tracks)
