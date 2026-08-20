from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from golftracer.config import Config
from golftracer.detect import clubhead
from golftracer.detect.dataset import _augmentations, collect_samples
from golftracer.label.schema import Label, LabelDocument, save_labels
from golftracer.phases.base import fit_label_constrained
from golftracer.session import Observation


def test_collect_samples_deduplicates_phase_collision(tmp_path: Path) -> None:
    root = tmp_path / "session" / "labels"
    save_labels(root / "1.backswing.json", LabelDocument(
        "fixture.mp4", 1.0, 60.0, "backswing",
        [Label(0, 1.0, 10.0, 20.0, "backswing")],
    ))
    save_labels(root / "1.downswing.json", LabelDocument(
        "fixture.mp4", 1.0, 60.0, "downswing",
        [Label(0, 1.0, 10.0, 20.0, "downswing")],
    ))
    samples, counts = collect_samples([root])
    assert counts["raw_labels"] == 2
    assert counts["unique_frames"] == 1
    assert samples[0].phases == {"backswing", "downswing"}


def test_offline_augmentation_has_blur_streak_brightness_but_no_flip() -> None:
    image = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
    variants = dict(_augmentations(image))
    assert set(variants) == {"base", "blur", "streak", "dark", "bright"}
    np.testing.assert_array_equal(variants["base"], image)
    assert not any("flip" in name for name in variants)


def test_detector_proposal_maps_roi_box_to_source_pixels(tmp_path: Path, monkeypatch) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"test")

    class FakeModel:
        def predict(self, *, source, **_kwargs):
            assert source[0].shape[:2] == (90, 200)
            class FakeTensor:
                def __init__(self, value):
                    self.value = np.asarray(value, dtype=np.float32)

                def detach(self):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return self.value

                def __getitem__(self, index):
                    return FakeTensor(self.value[index])

                def __len__(self):
                    return len(self.value)

            class FakeBoxes:
                conf = FakeTensor([0.8])
                xyxy = FakeTensor([[40.0, 20.0, 60.0, 40.0]])

                def __len__(self):
                    return len(self.conf)

            boxes = FakeBoxes()
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setattr(clubhead, "_load_model", lambda _path: FakeModel())
    frame = np.zeros((100, 200, 3), np.uint8)
    proposals = clubhead.propose_frames([frame], weights=weights, config=Config())
    assert proposals[0].x == 50.0
    assert proposals[0].y == 30.0
    assert proposals[0].confidence == pytest.approx(0.8)


def test_detector_proposals_are_secondary_fit_evidence() -> None:
    observations = [
        Observation(1, 1.1, 10.0, 20.0, source="detector"),
        Observation(2, 1.2, 20.0, 10.0, source="detector"),
    ]
    fit = fit_label_constrained(
        "followthrough", observations, [], Config(), observation_weight=1.0,
        forced_start=(1.0, 0.0, 30.0, 0),
        forced_start_source="impact_anchor",
        forced_start_calibration_phase="followthrough",
    )
    assert fit is not None
    assert len(fit.accepted) == 2
