from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2

from golftracer.config import Config
from golftracer.label.app import LabelApp, Proposal, _append_time_log, load_proposals
from golftracer.label.schema import Label, LabelDocument, load_labels, save_labels
from golftracer.phases.base import fit_label_constrained


def test_tracker_session_proposals_are_rebased_and_phase_filtered(tmp_path: Path) -> None:
    session = {
        "swings": [{
            "id": 2,
            "impact_t": 2.0,
            "tracks": [{
                "phase": "club",
                "metadata": {"top_t": 1.8},
                "observations": [
                    {"frame_index": 4, "t": 1.7, "x": 10.0, "y": 20.0},
                    {"frame_index": 5, "t": 1.8, "x": 11.0, "y": 19.0},
                    {"frame_index": 6, "t": 1.9, "x": 12.0, "y": 18.0},
                ],
            }],
        }],
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session), encoding="utf-8")
    proposals = load_proposals(path, "downswing", 1.5, 60.0, swing_id=2)
    assert sorted(proposals) == [18, 24]
    assert proposals[24].x == 12.0


def test_correction_records_source_delta_and_human_collision(tmp_path: Path) -> None:
    output = tmp_path / "1.downswing.json"
    document = LabelDocument("fixture.mp4", 1.0, 60.0, "downswing", correction_mode=True)
    frames = np.zeros((2, 20, 20, 3), np.uint8)
    timestamps = np.asarray([1.0, 1.0 + 1 / 60])
    proposals = {
        0: Proposal(0, 5.0, 6.0, 0.9),
        1: Proposal(1, 10.0, 10.0, 0.8),
    }
    app = LabelApp(frames, timestamps, document, output, [0, 1], proposals)
    app._record(0, 5.0, 6.0, "accepted")
    app._record(1, 13.0, 14.0, "corrected")
    loaded = load_labels(output)
    assert loaded.labels[0].source == "accepted"
    assert loaded.labels[0].delta_px == 0.0
    assert loaded.labels[1].source == "corrected"
    assert loaded.labels[1].delta_px == 5.0

    loaded.labels.append(Label(0, 1.0, 4.0, 4.0, "downswing", "human"))
    save_labels(output, loaded)
    assert load_labels(output).merged_labels()[0].source == "human"


def test_time_log_is_jsonl_next_to_labels(tmp_path: Path) -> None:
    output = tmp_path / "labels.json"
    document = LabelDocument("fixture.mp4", 1.0, 60.0, "backswing")
    path = _append_time_log(output, document, 90.0)
    assert path == tmp_path / "labels.json.time.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["minutes"] == 1.5


def test_unchanged_accepted_proposals_reproduce_label_fit() -> None:
    human = [
        Label(0, 1.0, 10.0, 30.0, "downswing"),
        Label(1, 1.1, 20.0, 10.0, "downswing"),
        Label(2, 1.2, 40.0, 30.0, "downswing"),
    ]
    accepted = [
        Label(item.frame_index, item.t, item.x, item.y, item.phase, "accepted", delta_px=0.0)
        for item in human
    ]
    first = fit_label_constrained("downswing", [], human, Config(), observation_weight=2.0)
    second = fit_label_constrained("downswing", [], accepted, Config(), observation_weight=2.0)
    assert first is not None and second is not None
    first_arc = np.linspace(0.0, first.length, 100)
    second_arc = np.linspace(0.0, second.length, 100)
    np.testing.assert_allclose(
        first.xy_at_arc(first_arc), second.xy_at_arc(second_arc), atol=0.0, rtol=0.0,
    )


def test_scripted_correction_keys_marker_skip_finish_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "1.followthrough.json"
    document = LabelDocument(
        "fixture.mp4", 1.0, 60.0, "followthrough", correction_mode=True,
    )
    frames = np.zeros((4, 100, 100, 3), np.uint8)
    timestamps = 1.0 + np.arange(4) / 60.0
    proposals = {0: Proposal(0, 50.0, 85.0, 0.9)}
    app = LabelApp(frames, timestamps, document, output, range(4), proposals)
    view = app._view()
    assert np.any(np.all(view == np.asarray([255, 0, 255]), axis=2))
    app._handle_key(ord("l"))
    app._handle_key(ord("a"))
    assert document.labels[0].source == "corrected"
    assert document.labels[0].delta_px == 1.0
    app._handle_key(ord("s"))
    app._handle_key(ord("u"))
    assert document.skipped_frames == []
    app._handle_key(ord("s"))
    app._handle_key(ord("m"))
    app._handle_key(ord("f"))
    assert document.skipped_frames == [1, 3]
    assert document.missing_frames == [2]
    resumed = LabelApp(frames, timestamps, load_labels(output), output, range(4), proposals)
    assert resumed.indices == []


def test_scripted_click_moves_detector_proposal(tmp_path: Path) -> None:
    output = tmp_path / "1.followthrough.json"
    document = LabelDocument(
        "fixture.mp4", 1.0, 60.0, "followthrough", correction_mode=True,
    )
    app = LabelApp(
        np.zeros((1, 100, 100, 3), np.uint8), np.asarray([1.0]),
        document, output, [0], {0: Proposal(0, 20.0, 20.0, 0.9)},
    )
    app._mouse(cv2.EVENT_LBUTTONDOWN, 23, 24, 0, None)
    assert document.labels[0].source == "corrected"
    assert document.labels[0].delta_px == 5.0
