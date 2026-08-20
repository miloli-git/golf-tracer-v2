from __future__ import annotations

import json
from pathlib import Path

from golftracer.config import Config
from golftracer.label.queue import (
    QueueSwing, format_status, label_command, load_queue, queue_status,
    run_queue, swing_status,
)
from golftracer.label.schema import Label, LabelDocument, save_labels


def _swing(tmp_path: Path, total: int = 4) -> QueueSwing:
    return QueueSwing(
        "session:1", "Session 1", tmp_path / "video.mp4", 1, 2.0,
        tmp_path / "session", ("followthrough",), {"followthrough": total},
    )


def test_queue_status_counts_explicit_coverage_and_correction_metrics(tmp_path: Path) -> None:
    swing = _swing(tmp_path)
    save_labels(swing.label_path("followthrough"), LabelDocument(
        str(swing.video), 2.0, 60.0, "followthrough",
        [
            Label(0, 2.0, 10.0, 20.0, "followthrough", "accepted", delta_px=0.0),
            Label(1, 2.0 + 1 / 60, 12.0, 18.0, "followthrough", "corrected", delta_px=5.0),
        ],
        missing_frames=[2], skipped_frames=[3], time_on_task_s=90.0,
    ))
    status = swing_status(swing, Config())
    assert status["done"] == 4
    assert status["remaining"] == 0
    assert status["complete"] is True
    assert status["accepted_pct"] == 50.0
    assert status["median_correction_px"] == 2.5
    assert status["minutes"] == 1.5


def test_queue_manifest_paths_commands_and_completed_skip(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    manifest = tmp_path / "queue.json"
    manifest.write_text(json.dumps({
        "schema": 1,
        "weights": "weights.pt",
        "swings": [{
            "key": "session:1", "name": "Session 1", "video": "video.mp4",
            "swing_id": 1, "impact_t": 2.0, "labels_dir": "session",
            "phases": ["followthrough"],
            "frame_counts": {"followthrough": 1},
        }],
    }), encoding="utf-8")
    queue = load_queue(manifest)
    swing = queue.swings[0]
    command = label_command(queue, swing, "followthrough")
    assert "--full" in command
    assert command[command.index("--propose") + 1] == "detector"
    save_labels(swing.label_path("followthrough"), LabelDocument(
        str(swing.video), 2.0, 60.0, "followthrough",
        [Label(0, 2.0, 1.0, 2.0, "followthrough", "accepted", delta_px=0.0)],
    ))
    launched = []
    assert run_queue(
        queue, Config(), launcher=lambda *args: launched.append(args) or 0,
    ) == 0
    assert launched == []
    status = queue_status(queue, Config())
    assert status["complete"] is True
    assert "TOTAL 1/1 frames" in format_status(status)


def test_queue_stops_for_resume_when_labeller_returns_incomplete(tmp_path: Path) -> None:
    swing = _swing(tmp_path, total=2)
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps({
        "schema": 1, "weights": "weights.pt",
        "swings": [{
            "key": swing.key, "name": swing.name, "video": str(swing.video),
            "swing_id": 1, "impact_t": 2.0, "labels_dir": str(swing.labels_dir),
            "phases": ["followthrough"], "frame_counts": {"followthrough": 2},
        }],
    }), encoding="utf-8")
    (tmp_path / "weights.pt").write_bytes(b"weights")
    queue = load_queue(queue_path)
    launches = []

    def launch(command, log_path, cwd):
        launches.append((command, log_path, cwd))
        return 0

    assert run_queue(queue, Config(), launcher=launch) == 0
    assert len(launches) == 1
