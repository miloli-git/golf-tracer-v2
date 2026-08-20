"""Ultralytics nano inference restricted to the configured swing ROI."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import Config


def default_weights() -> Path:
    configured = os.environ.get("GOLFTRACER_DETECTOR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "weights" / "clubhead-yolo26n.pt"


def roi_bounds(frame: np.ndarray, config: Config) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x0, x1, y0, y1 = config.detect_roi
    left = int(np.clip(round(x0 * width), 0, width - 1))
    right = int(np.clip(round(x1 * width), left + 1, width))
    top = int(np.clip(round(y0 * height), 0, height - 1))
    bottom = int(np.clip(round(y1 * height), top + 1, height))
    return left, top, right, bottom


@lru_cache(maxsize=4)
def _load_model(path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised on base-only installs
        raise RuntimeError('detector support requires: pip install -e ".[detect]"') from exc
    return YOLO(path)


def propose_frames(
    frames: np.ndarray | Sequence[np.ndarray],
    *,
    weights: str | Path | None = None,
    config: Config | None = None,
    device: str = "cpu",
) -> dict[int, object]:
    """Return the highest-confidence clubhead box centre per input frame."""
    from ..label.app import Proposal

    cfg = config or Config()
    path = Path(weights) if weights is not None else default_weights()
    if not path.is_file():
        raise FileNotFoundError(
            f"clubhead weights not found: {path}; run `golftracer detect train` first"
        )
    arrays = list(frames)
    if not arrays:
        return {}
    bounds = [roi_bounds(frame, cfg) for frame in arrays]
    crops = [frame[top:bottom, left:right] for frame, (left, top, right, bottom) in zip(arrays, bounds, strict=True)]
    model = _load_model(str(path.resolve()))
    results = model.predict(
        source=crops, imgsz=cfg.detect_input_size_px, conf=cfg.detect_confidence,
        device=device, verbose=False, stream=False,
    )
    proposals: dict[int, Proposal] = {}
    for index, (result, (left, top, _right, _bottom)) in enumerate(zip(results, bounds, strict=True)):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        confidence = boxes.conf.detach().cpu().numpy()
        winner = int(np.argmax(confidence))
        xyxy = boxes.xyxy[winner].detach().cpu().numpy()
        proposals[index] = Proposal(
            index,
            float(left + 0.5 * (xyxy[0] + xyxy[2])),
            float(top + 0.5 * (xyxy[1] + xyxy[3])),
            float(confidence[winner]),
        )
    return proposals
