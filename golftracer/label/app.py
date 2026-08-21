"""Resume-safe OpenCV click labeller using the shared one-seek decoder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Sequence

import cv2
import numpy as np

from ..config import Config
from ..decode import decode_window, probe
from .schema import Label, LabelDocument, load_labels, save_labels


WINDOW = "golftracer label"


def _same_video(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve())
    )


@dataclass(frozen=True)
class Proposal:
    frame_index: int
    x: float
    y: float
    confidence: float = 1.0


def _proposal_rows(payload: object, phase: str, swing_id: int | None) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("proposal JSON must be a list, track, session, or proposals mapping")
    if isinstance(payload.get("proposals"), list):
        return [item for item in payload["proposals"] if isinstance(item, dict)]
    if isinstance(payload.get("observations"), list):
        return [item for item in payload["observations"] if isinstance(item, dict)]
    swings = payload.get("swings")
    if not isinstance(swings, list):
        raise ValueError("proposal JSON contains no proposals or session swings")
    candidates = [item for item in swings if isinstance(item, dict)]
    if swing_id is not None:
        candidates = [item for item in candidates if int(item.get("id", -1)) == swing_id]
    if len(candidates) != 1:
        raise ValueError("proposal session needs --swing-id unless it contains one swing")
    swing = candidates[0]
    wanted_track = "club" if phase in {"backswing", "downswing"} else (
        "follow" if phase == "followthrough" else phase
    )
    tracks = [
        item for item in swing.get("tracks", [])
        if isinstance(item, dict) and item.get("phase") == wanted_track
    ]
    if len(tracks) != 1:
        raise ValueError(f"proposal session has no unique {wanted_track} track")
    rows = [
        {**item, "_time_based": True}
        for item in tracks[0].get("observations", []) if isinstance(item, dict)
    ]
    if phase in {"backswing", "downswing"}:
        top_t = float(tracks[0].get("metadata", {}).get("top_t", swing["impact_t"]))
        if phase == "backswing":
            rows = [item for item in rows if float(item.get("t", 0.0)) <= top_t + 1e-7]
        else:
            rows = [item for item in rows if float(item.get("t", 0.0)) >= top_t - 1e-7]
    return rows


def load_proposals(
    path: str | Path,
    phase: str,
    window_start: float,
    fps: float,
    *,
    swing_id: int | None = None,
) -> dict[int, Proposal]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    proposals: dict[int, Proposal] = {}
    for row in _proposal_rows(payload, phase, swing_id):
        if row.get("_time_based") and "t" in row:
            index = int(round((float(row["t"]) - window_start) * fps))
        elif "frame_index" in row:
            index = int(row["frame_index"])
        elif "t" in row:
            index = int(round((float(row["t"]) - window_start) * fps))
        else:
            continue
        if index < 0 or "x" not in row or "y" not in row:
            continue
        proposals[index] = Proposal(
            index, float(row["x"]), float(row["y"]),
            float(row.get("confidence", row.get("conf", 1.0))),
        )
    return proposals


def _append_time_log(output: Path, document: LabelDocument, elapsed_s: float) -> Path:
    path = output.with_suffix(output.suffix + ".time.jsonl")
    payload = {
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "phase": document.phase,
        "correction_mode": document.correction_mode,
        "minutes": elapsed_s / 60.0,
        "labels": len(document.merged_labels()),
        "accepted": sum(item.source == "accepted" for item in document.labels),
        "corrected": sum(item.source == "corrected" for item in document.labels),
        "missing": len(document.missing_frames),
        "skipped": len(document.skipped_frames),
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def phase_window(phase: str, impact_t: float, config: Config) -> tuple[float, float]:
    if phase == "backswing":
        return max(0.0, impact_t - config.backswing_pre_s), config.backswing_pre_s
    if phase == "downswing":
        return max(0.0, impact_t - config.downswing_pre_s), config.downswing_pre_s
    if phase == "ball":
        return impact_t, config.ball_post_s
    if phase == "followthrough":
        return impact_t, config.follow_through_post_s
    raise ValueError(f"unknown phase: {phase}")


def selected_indices(count: int, full: bool) -> list[int]:
    if full or count <= 10:
        return list(range(count))
    return sorted(set(int(round(value)) for value in np.linspace(0, count - 1, 10)))


class LabelApp:
    def __init__(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
        document: LabelDocument,
        output: Path,
        indices: Sequence[int],
        proposals: dict[int, Proposal] | None = None,
    ):
        self.frames = frames
        self.timestamps = timestamps
        self.document = document
        self.output = output
        labelled = {
            item.frame_index for item in document.labels
        } | set(document.missing_frames) | set(document.skipped_frames)
        self.indices = [index for index in indices if index not in labelled]
        self.position = 0
        self.cursor: tuple[int, int] | None = None
        self.skipped: set[int] = set()
        self.proposals = proposals or {}
        self.adjusted: dict[int, tuple[float, float]] = {}
        self.history: list[tuple[str, tuple[int, ...]]] = []
        self.started = time.perf_counter()
        self.session_elapsed_s = 0.0

    def _save(self) -> None:
        elapsed = time.perf_counter() - self.started
        self.document.time_on_task_s += elapsed
        self.session_elapsed_s += elapsed
        self.started = time.perf_counter()
        save_labels(self.output, self.document)

    def _clear(self, index: int) -> None:
        """Drop any prior label/missing/skipped entry for a revisited frame."""
        self.document.labels[:] = [item for item in self.document.labels if item.frame_index != index]
        for bucket in (self.document.missing_frames, self.document.skipped_frames):
            while index in bucket:
                bucket.remove(index)

    def _record(self, index: int, x: float, y: float, source: str) -> None:
        self._clear(index)
        proposal = self.proposals.get(index)
        delta = None if proposal is None else math.hypot(x - proposal.x, y - proposal.y)
        self.document.labels.append(Label(
            frame_index=index, t=float(self.timestamps[index]), x=float(x), y=float(y),
            phase=self.document.phase, source=source, delta_px=delta,
        ))
        self.history.append(("label", (index,)))
        self.position += 1
        self._save()

    def _mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        self.cursor = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN or self.position >= len(self.indices):
            return
        index = self.indices[self.position]
        self._record(index, x, y, "corrected" if index in self.proposals else "human")

    def _view(self) -> np.ndarray:
        index = self.indices[self.position]
        frame = self.frames[index].copy()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 76), (18, 18, 18), -1)
        if self.document.correction_mode:
            text = f"{self.document.phase} frame {index} {self.position + 1}/{len(self.indices)} | a/space accept | arrows/hjkl nudge | click correct | m missing | s skip | b/n back/fwd | u undo | f finish"
        else:
            text = f"{self.document.phase} frame {index} {self.position + 1}/{len(self.indices)} | click head COM | s skip (offscreen) | b/n back/fwd | u undo | q save"
        cv2.putText(frame, text, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        existing = next((item for item in self.document.labels if item.frame_index == index), None)
        if existing is not None:
            centre = (int(round(existing.x)), int(round(existing.y)))
            cv2.circle(frame, centre, 9, (80, 255, 80), 2, cv2.LINE_AA)
            cv2.putText(frame, f"labelled ({existing.source}); relabel replaces", (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 80), 1, cv2.LINE_AA)
        elif index in self.document.missing_frames or index in self.document.skipped_frames:
            cv2.putText(frame, "marked missing/skipped; relabel replaces", (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 200, 255), 1, cv2.LINE_AA)
        proposal = self.proposals.get(index)
        if proposal is not None and existing is None:
            point = self.adjusted.get(index, (proposal.x, proposal.y))
            centre = (int(round(point[0])), int(round(point[1])))
            cv2.circle(frame, centre, 12, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(frame, centre, (255, 255, 255), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
            cv2.putText(frame, f"proposal {proposal.confidence:.2f}", (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 160, 255), 1, cv2.LINE_AA)
        if self.cursor is not None:
            x, y = self.cursor
            crop = cv2.getRectSubPix(frame, (72, 72), (float(x), float(y)))
            inset = cv2.resize(crop, (216, 216), interpolation=cv2.INTER_CUBIC)
            cv2.drawMarker(inset, (108, 108), (80, 255, 80), cv2.MARKER_CROSS, 30, 1, cv2.LINE_AA)
            # Float the loupe beside the cursor, flipping sides so it stays
            # on-screen and never covers the point being clicked.
            gap = 28
            left = x + gap if x + gap + 216 <= frame.shape[1] - 4 else x - gap - 216
            top = min(max(80, y - 108), frame.shape[0] - 220)
            left = min(max(4, left), frame.shape[1] - 220)
            cv2.rectangle(inset, (0, 0), (215, 215), (18, 18, 18), 1)
            frame[top:top + 216, left:left + 216] = inset
        return frame

    def _handle_key(self, key: int) -> None:
        ascii_key = key & 0xFF
        if ascii_key == ord("u") and self.history:
            action, indices = self.history.pop()
            if action == "label":
                self.document.labels.pop()
            elif action == "missing":
                for index in reversed(indices):
                    self.document.missing_frames.remove(index)
            else:
                for index in reversed(indices):
                    self.document.skipped_frames.remove(index)
            self.position = max(0, self.position - len(indices))
            self._save()
            return
        if self.position >= len(self.indices):
            if ascii_key == ord("b"):
                self.position = max(0, len(self.indices) - 1)
            return
        index = self.indices[self.position]
        if ascii_key == ord("b"):
            self.position = max(0, self.position - 1)
            return
        if ascii_key == ord("n"):
            self.position = min(len(self.indices) - 1, self.position + 1)
            return
        if ascii_key == ord("s"):
            self._clear(index)
            self.document.skipped_frames.append(index)
            self.history.append(("skipped", (index,)))
            self.position += 1
            self._save()
            return
        if ascii_key == ord("f"):
            remaining = tuple(self.indices[self.position:])
            self.document.skipped_frames.extend(remaining)
            self.history.append(("skipped", remaining))
            self.position = len(self.indices)
            self._save()
            return
        if ascii_key == ord("m"):
            self._clear(index)
            self.document.missing_frames.append(index)
            self.history.append(("missing", (index,)))
            self.position += 1
            self._save()
            return
        if ascii_key in (ord("a"), ord(" "), 13) and any(item.frame_index == index for item in self.document.labels):
            self.position += 1
            return
        if ascii_key in (ord("a"), ord(" "), 13) and index in self.proposals:
            proposal = self.proposals[index]
            point = self.adjusted.get(index, (proposal.x, proposal.y))
            source = "accepted" if point == (proposal.x, proposal.y) else "corrected"
            self._record(index, point[0], point[1], source)
            return
        nudges = {
            2424832: (-1, 0), 2555904: (1, 0),
            2490368: (0, -1), 2621440: (0, 1),
            ord("h"): (-1, 0), ord("l"): (1, 0),
            ord("k"): (0, -1), ord("j"): (0, 1),
        }
        nudge = nudges.get(key, nudges.get(ascii_key))
        if nudge is not None and index in self.proposals:
            proposal = self.proposals[index]
            current = self.adjusted.get(index, (proposal.x, proposal.y))
            self.adjusted[index] = (current[0] + nudge[0], current[1] + nudge[1])

    def run(self) -> None:
        if not self.indices:
            _append_time_log(self.output, self.document, 0.0)
            return
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, self._mouse)
        except cv2.error as exc:
            raise RuntimeError("OpenCV GUI is unavailable; use --selftest for a headless check") from exc
        try:
            while self.position < len(self.indices):
                cv2.imshow(WINDOW, self._view())
                key = cv2.waitKeyEx(20)
                ascii_key = key & 0xFF
                if ascii_key in (ord("q"), 27):
                    break
                self._handle_key(key)
        finally:
            self._save()
            _append_time_log(self.output, self.document, self.session_elapsed_s)
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass


def run_labeller(
    video: Path,
    output: Path,
    phase: str,
    impact_t: float,
    full: bool,
    config: Config,
    *,
    propose: str | Path | None = None,
    weights: Path | None = None,
    swing_id: int | None = None,
) -> LabelDocument:
    start, duration = phase_window(phase, impact_t, config)
    meta = probe(video)
    frames, timestamps = decode_window(video, start, duration, fps=meta.fps)
    if output.is_file():
        document = load_labels(output)
        if not _same_video(document.video, video) or document.phase != phase or abs(document.window_start - start) > 1e-6:
            raise ValueError("existing label document does not match this video/phase/window")
    else:
        document = LabelDocument(str(video.resolve()), start, meta.fps, phase)
    proposals: dict[int, Proposal] = {}
    if propose is not None:
        if str(propose) == "detector":
            from ..detect.clubhead import propose_frames
            proposal_config = config.with_overrides(
                detect_confidence=config.label_proposal_confidence,
            )
            proposals = propose_frames(frames, weights=weights, config=proposal_config)
            proposal_name = "detector"
        else:
            proposals = load_proposals(
                Path(propose), phase, start, meta.fps, swing_id=swing_id,
            )
            proposal_name = Path(propose).name
        document.correction_mode = True
        document.proposal_source = proposal_name
    LabelApp(
        frames, timestamps, document, output, selected_indices(len(frames), full), proposals,
    ).run()
    return document


def selftest() -> None:
    document = LabelDocument("synthetic.mp4", 1.25, 60.0, "backswing", [
        Label(0, 1.25, 10.0, 20.0, "backswing"),
        Label(1, 1.25 + 1 / 60, 11.0, 19.0, "backswing", "accepted"),
    ], time_on_task_s=2.5)
    with tempfile.TemporaryDirectory(prefix="golftracer-label-") as directory:
        path = Path(directory) / "labels.json"
        save_labels(path, document)
        loaded = load_labels(path)
        if loaded != document:
            raise AssertionError("label JSON round-trip changed the document")
