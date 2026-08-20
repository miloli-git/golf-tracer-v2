"""Render a portable club-arc QA strip with human click crosses overlaid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from golftracer.decode import decode_window
from golftracer.label.schema import load_labels
from golftracer.render.compositor import fit_track
from golftracer.session import Session


def _cross(frame: np.ndarray, x: float, y: float) -> None:
    point = (int(round(x)), int(round(y)))
    cv2.drawMarker(frame, point, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 22, 6, cv2.LINE_AA)
    cv2.drawMarker(frame, point, (255, 40, 220), cv2.MARKER_TILTED_CROSS, 22, 3, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--swing", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = Session.from_json(args.session)
    swing = next(item for item in session.swings if item.id == args.swing)
    tracks = [item for item in swing.tracks if item.phase in {"club", "follow"}]
    fitted_tracks = [(track, fit_track(track)) for track in tracks]
    fitted_tracks = [(track, fitted) for track, fitted in fitted_tracks if fitted is not None]
    if not fitted_tracks:
        raise RuntimeError("club/follow tracks have no fitted path")
    phase_docs = []
    for phase in ("backswing", "downswing", "followthrough"):
        for name in (f"{args.swing}.{phase}.json", f"swing-{args.swing:03d}.{phase}.json", f"{phase}.json"):
            path = args.labels / name if args.labels.is_dir() else args.labels
            if path.is_file():
                phase_docs.append(load_labels(path))
                break
    labels = [label for document in phase_docs for label in document.labels]
    first_t = min(track.observations[0].t for track, _ in fitted_tracks)
    last_t = max(track.observations[-1].t for track, _ in fitted_tracks)
    labels = [item for item in labels if first_t - 1 / session.fps <= item.t <= last_t + 1 / session.fps]
    polylines = []
    for track, fitted in fitted_tracks:
        full_path = fitted.samples_until(last_t, fps=240.0, fade_length_s=10.0)
        polylines.append((
            track.phase,
            np.asarray([(round(item.x), round(item.y)) for item in full_path], np.int32),
        ))

    lead = 0.10
    start = max(0.0, first_t - lead)
    frames, _ = decode_window(args.video, start, last_t - start + 1 / session.fps, fps=session.fps)
    panel_times = np.linspace(first_t, last_t, 8)
    panels = []
    for time_s in panel_times:
        index = int(np.clip(round((time_s - start) * session.fps), 0, len(frames) - 1))
        panel = frames[index].copy()
        for phase, polyline in polylines:
            if len(polyline) <= 1:
                continue
            colour = (60, 220, 255) if phase == "club" else (255, 180, 60)
            cv2.polylines(panel, [polyline], False, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.polylines(panel, [polyline], False, colour, 3, cv2.LINE_AA)
        for label in labels:
            _cross(panel, label.x, label.y)
        caption = f"{time_s - swing.impact_t:+.3f}s  magenta=click  yellow/blue=fit"
        cv2.putText(panel, caption, (18, panel.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(panel, caption, (18, panel.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(panel, (180, 320), interpolation=cv2.INTER_AREA))
    strip = np.hstack(panels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), strip):
        raise RuntimeError(f"could not write {args.out}")
    print(f"wrote {args.out} with {len(labels)} in-phase click rows")


if __name__ == "__main__":
    main()
