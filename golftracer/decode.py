"""One-seek, sequential, count-indexed video decoding."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import subprocess
from typing import Iterator

import av
import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    rotation: int


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"media command failed: {message}")
    return result


def _rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def probe(video: str | Path) -> VideoMeta:
    """Return timing metadata and geometry measured from an autorotated frame."""
    return _probe_cached(str(Path(video).resolve()))


@lru_cache(maxsize=16)
def _probe_cached(path_text: str) -> VideoMeta:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(path)
    details = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    document = json.loads(details.stdout)
    try:
        stream = next(item for item in document["streams"] if item["codec_type"] == "video")
    except (KeyError, StopIteration) as exc:
        raise RuntimeError(f"no video stream found in {path}") from exc
    rotation = 0
    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            rotation = int(item["rotation"])
            break
    if not rotation:
        rotation = int(stream.get("tags", {}).get("rotate", 0))
    encoded = _run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
    ]).stdout
    frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"could not decode a display-oriented frame from {path}")
    height, width = frame.shape[:2]
    fmt = document.get("format", {})
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    fps = _rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    return VideoMeta(str(path), width, height, fps, duration, rotation)


def _canonical_start(t0: float, source_fps: float) -> float:
    # v1's canonical club windows seek at the measured window start itself; frame
    # identity then comes only from the sequential decoded count. Rounding the seek to
    # a source-frame boundary selects a different image on variable-rate phone clips.
    del source_fps
    return float(t0)


def decode_window(
    video: str | Path,
    t0: float,
    dur: float,
    fps: float | None = None,
    gray: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a window after one input seek and index frames from its start."""
    if t0 < 0 or dur < 0:
        raise ValueError("t0 and dur must be non-negative")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive")
    meta = probe(video)
    output_fps = float(fps or meta.fps)
    if output_fps <= 0:
        raise RuntimeError("video frame rate is unavailable")
    start = _canonical_start(t0, meta.fps or output_fps)
    channels = 1 if gray else 3
    shape = (meta.height, meta.width) if gray else (meta.height, meta.width, 3)
    if dur == 0:
        return np.empty((0, *shape), np.uint8), np.empty(0, np.float64)
    filters: list[str] = []
    if fps is not None:
        filters.append(f"fps={output_fps:.12g}")
    if gray:
        filters.append("format=gray")
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.9f}",
        "-i", str(video), "-t", f"{dur:.9f}", "-map", "0:v:0", "-an", "-sn",
    ]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "-"]
    result = _run(command)
    frame_bytes = meta.width * meta.height * channels
    count = len(result.stdout) // frame_bytes
    if len(result.stdout) != count * frame_bytes:
        raise RuntimeError("ffmpeg returned a truncated raw frame")
    frames = np.frombuffer(result.stdout, np.uint8).reshape((count, *shape)).copy()
    pts = start + np.arange(count, dtype=np.float64) / output_fps
    return frames, pts


def _rotate_for_display(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply the stream display rotation to an encoded-orientation frame."""
    normalized = rotation % 360
    if normalized == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if normalized == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def _iter_pts_frames(
    video: str | Path,
    t0: float,
    dur: float,
    fps: float | None,
    gray: bool,
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield source frames using v1's PTS-driven maximum-rate sampling."""
    if t0 < 0 or dur < 0:
        raise ValueError("t0 and dur must be non-negative")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive")
    if dur == 0:
        return
    meta = probe(video)
    end = t0 + dur
    with av.open(str(video)) as container:
        try:
            stream = container.streams.video[0]
        except IndexError as exc:
            raise RuntimeError(f"no video stream found in {video}") from exc
        seek_pts = int(max(0.0, t0) / float(stream.time_base))
        container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
        next_sample = t0
        interval = 1.0 / fps if fps is not None else None
        for decoded in container.decode(stream):
            if decoded.pts is None:
                continue
            pts_time = float(decoded.pts * stream.time_base)
            if pts_time + 1e-9 < t0:
                continue
            if pts_time >= end - 1e-9:
                break
            if interval is not None:
                if pts_time + 1e-9 < next_sample:
                    continue
                skipped = max(1, int((pts_time - next_sample) / interval) + 1)
                next_sample += skipped * interval
            image = decoded.to_ndarray(format="gray" if gray else "bgr24")
            yield _rotate_for_display(image, meta.rotation), pts_time


def read_window_pts(
    video: str | Path,
    t0: float,
    dur: float,
    fps: float | None = None,
    gray: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode with v1 ``read_window`` semantics and preserve source PTS.

    This is deliberately separate from :func:`decode_window`: club tracking is
    count-indexed on an ffmpeg output grid, while the calibrated v1 ball chain
    samples source frames by media time and retains their PTS.
    """
    frames: list[np.ndarray] = []
    times: list[float] = []
    for frame, pts_time in _iter_pts_frames(video, t0, dur, fps, gray):
        frames.append(frame)
        times.append(pts_time)
    if frames:
        return np.stack(frames), np.asarray(times, dtype=np.float64)
    meta = probe(video)
    channels = () if gray else (3,)
    return (
        np.empty((0, meta.height, meta.width, *channels), dtype=np.uint8),
        np.empty(0, dtype=np.float64),
    )
