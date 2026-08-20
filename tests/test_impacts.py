from __future__ import annotations

from pathlib import Path

import cv2

from golftracer.config import Config
from golftracer.impacts import detect_impacts, read_candidate_times, write_calibration_strips


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.mp4"


def test_synthetic_click_train_exact_to_one_frame() -> None:
    config = Config().with_overrides(
        av_offset_s=0.0, impact_min_gap_s=0.4, pose_enabled=False,
        motion_floor_percentile=80.0,
    )
    impacts = detect_impacts(FIXTURE, config)
    actual = [item["t_video"] for item in impacts]
    expected = [0.5, 1.5, 2.5]
    assert len(actual) == len(expected)
    assert all(abs(found - wanted) <= 1 / 30 for found, wanted in zip(actual, expected))


def test_zero_impacts_and_calibration_strip(tmp_path: Path) -> None:
    quiet_config = Config().with_overrides(impact_min_onset=1.0, pose_enabled=False)
    assert detect_impacts(FIXTURE, quiet_config) == []

    config = Config().with_overrides(
        av_offset_s=0.0, impact_min_gap_s=0.4, pose_enabled=False,
        motion_floor_percentile=80.0,
    )
    impacts = detect_impacts(FIXTURE, config)
    paths = write_calibration_strips(FIXTURE, impacts, tmp_path, config, count=1)
    strip = cv2.imread(str(paths[0]))
    assert strip is not None
    assert strip.shape[1] == 5 * 480


def test_explicit_candidate_file_filters_rejected_rows(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        '[{"t_video": 1.0}, {"t_impact_s": 2.0, "accepted": false}, '
        '{"time": 3.0}]',
        encoding="utf-8",
    )
    assert read_candidate_times(path) == [1.0, 3.0]
