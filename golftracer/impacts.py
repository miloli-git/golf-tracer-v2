"""Visual-first impact detection with audio onset timing refinement."""

from __future__ import annotations

import json
from importlib import metadata as importlib_metadata
import logging
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
from typing import Sequence
import urllib.request

import cv2
import numpy as np

from .config import Config
from .decode import decode_window, probe
from .session import Swing


LOG = logging.getLogger("golftracer.impacts")
_SHOWINFO_PTS = re.compile(r"showinfo.*?pts_time:\s*([-+0-9.eE]+)")


def _extract_audio(video: str | Path, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vn", "-sn",
        "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"audio decode failed: {message}")
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def _preceding_floor(energy: np.ndarray, window: int, guard: int) -> np.ndarray:
    if len(energy) == 0:
        return energy.copy()
    width = max(guard + 1, window)
    padded = np.pad(energy, (width, 0), mode="edge")
    views = np.lib.stride_tricks.sliding_window_view(padded, width)[: len(energy)]
    usable = views[:, : width - guard]
    floor = np.empty(len(energy), np.float32)
    chunk = 20_000
    for start in range(0, len(energy), chunk):
        stop = min(len(energy), start + chunk)
        floor[start:stop] = np.median(usable[start:stop], axis=1)
    return floor


def _dedupe(times: np.ndarray, scores: np.ndarray, min_gap_s: float) -> list[int]:
    order = sorted(range(len(times)), key=lambda i: (-float(scores[i]), float(times[i]), i))
    kept: list[int] = []
    for index in order:
        if all(abs(float(times[index] - times[other])) >= min_gap_s for other in kept):
            kept.append(index)
    return sorted(kept, key=lambda i: float(times[i]))


def _validate_roi(roi: Sequence[float]) -> tuple[float, float, float, float]:
    if len(roi) != 4:
        raise ValueError("ROI must contain x0,x1,y0,y1")
    x0, x1, y0, y1 = (float(value) for value in roi)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"invalid normalized ROI: {roi}")
    return x0, x1, y0, y1


def _read_exact(pipe, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        block = pipe.read(remaining)
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def motion_signal(
    video: str | Path,
    config: Config,
    *,
    duration_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode low-resolution frames once and return PTS-aligned ROI motion."""
    x0, x1, y0, y1 = _validate_roi(config.motion_roi)
    meta = probe(video)
    height = int(round(meta.height * config.motion_width / meta.width / 2) * 2)
    interval = 1.0 / config.motion_fps
    select_gap = interval * 0.99
    filters = (
        f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{select_gap:.9f}),"
        f"scale={config.motion_width}:{height},format=gray,showinfo"
    )
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "info"]
    if config.motion_hwaccel:
        command += ["-hwaccel", config.motion_hwaccel]
    command += ["-i", str(video)]
    if duration_s is not None:
        command += ["-t", str(duration_s)]
    command += [
        "-vf", filters, "-fps_mode", "passthrough", "-f", "rawvideo",
        "-pix_fmt", "gray", "-",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
    )
    assert process.stdout is not None and process.stderr is not None
    pts_queue: queue.Queue[float | None] = queue.Queue()
    stderr_tail: list[str] = []

    def consume_stderr() -> None:
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace")
            match = _SHOWINFO_PTS.search(line)
            if match:
                pts_queue.put(float(match.group(1)))
            elif line.strip():
                stderr_tail.append(line.strip())
                del stderr_tail[:-20]
        pts_queue.put(None)

    stderr_thread = threading.Thread(target=consume_stderr, daemon=True)
    stderr_thread.start()
    frame_bytes = config.motion_width * height
    crop = (
        slice(int(y0 * height), max(int(y0 * height) + 1, int(y1 * height))),
        slice(
            int(x0 * config.motion_width),
            max(int(x0 * config.motion_width) + 1, int(x1 * config.motion_width)),
        ),
    )
    times: list[float] = []
    motion: list[float] = []
    previous: np.ndarray | None = None
    while True:
        raw = _read_exact(process.stdout, frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            process.kill()
            raise RuntimeError("ffmpeg returned a truncated motion frame")
        try:
            pts_time = pts_queue.get(timeout=30)
        except queue.Empty as exc:
            process.kill()
            raise RuntimeError("timed out waiting for ffmpeg motion PTS") from exc
        if pts_time is None:
            process.kill()
            raise RuntimeError("ffmpeg ended before emitting a PTS for every motion frame")
        frame = np.frombuffer(raw, np.uint8).reshape(height, config.motion_width)
        region = frame[crop]
        score = 0.0 if previous is None else float(cv2.absdiff(region, previous).mean())
        times.append(pts_time)
        motion.append(score)
        previous = region.copy()
    return_code = process.wait()
    stderr_thread.join(timeout=5)
    if return_code:
        raise RuntimeError(
            f"ffmpeg motion decode failed ({return_code}): " + "\n".join(stderr_tail[-8:])
        )
    if not times:
        return np.empty(0, np.float64), np.empty(0, np.float32)
    return np.asarray(times, np.float64), np.asarray(motion, np.float32)


def find_motion_candidates(
    times: Sequence[float], motion: Sequence[float], config: Config
) -> list[tuple[float, float]]:
    """Find locally impulsive motion peaks and apply score-first refractory NMS."""
    t = np.asarray(times, dtype=float)
    energy = np.asarray(motion, dtype=float)
    if t.shape != energy.shape or t.ndim != 1:
        raise ValueError("times and motion must be equally-sized 1-D arrays")
    if len(t) < 3:
        return []
    floor = float(np.percentile(energy, config.motion_floor_percentile))
    proposed: dict[int, float] = {}
    for index in np.flatnonzero(energy >= floor):
        local = np.flatnonzero(
            (t >= t[index] - config.motion_peak_window_s)
            & (t <= t[index] + config.motion_peak_window_s)
        )
        if not len(local):
            continue
        peak = int(local[np.argmax(energy[local])])
        background = energy[
            (t >= t[peak] - config.motion_background_window_s)
            & (t <= t[peak] + config.motion_background_window_s)
        ]
        baseline = float(np.median(background)) if len(background) else 0.0
        if (
            energy[peak] > max(floor, np.finfo(np.float32).eps)
            and energy[peak] >= config.motion_impulse_ratio * baseline
        ):
            proposed[peak] = float(energy[peak])
    indices = sorted(proposed)
    kept = _dedupe(
        np.asarray([t[index] for index in indices]),
        np.asarray([proposed[index] for index in indices]),
        config.impact_min_gap_s,
    )
    return [(float(t[indices[index]]), proposed[indices[index]]) for index in kept]


def _pose_imports():
    """Return MediaPipe Tasks symbols across supported package layouts."""
    try:
        import mediapipe as mp
        try:
            from mediapipe.tasks import python as mp_python
        except ImportError:
            mp_python = mp.tasks.python
        try:
            from mediapipe.tasks.python import vision
        except ImportError:
            vision = mp.tasks.python.vision
    except (ImportError, AttributeError):
        return None
    if not hasattr(vision, "PoseLandmarker"):
        return None
    try:
        version = importlib_metadata.version("mediapipe")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"
    return mp, mp_python, vision, version


def pose_available() -> bool:
    return _pose_imports() is not None


def pose_model_path(config: Config) -> Path:
    """Download the public lite landmarker once into the user's cache."""
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    destination = cache_root / "golftracer" / "pose_landmarker_lite.task"
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    LOG.info("downloading pose model to %s", destination)
    try:
        urllib.request.urlretrieve(config.pose_model_url, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def pose_gate_candidates(
    video: str | Path,
    candidates: Sequence[tuple[float, float]],
    config: Config,
) -> list[tuple[bool, float]] | None:
    """Gate candidates when MediaPipe Tasks is installed; otherwise return None."""
    if not config.pose_enabled:
        return None
    imports = _pose_imports()
    if imports is None:
        LOG.warning(
            "MediaPipe pose gate is off; install it with `pip install -e .[pose]`"
        )
        return None
    mp, mp_python, vision, version = imports
    LOG.info("using MediaPipe %s Tasks PoseLandmarker API", version)
    try:
        model = pose_model_path(config)
    except (OSError, ValueError) as exc:
        LOG.warning("MediaPipe pose gate is off because its model is unavailable: %s", exc)
        return None
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    results: list[tuple[bool, float]] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for candidate_time, _ in candidates:
            start = max(0.0, candidate_time - config.pose_window_s)
            duration = candidate_time + config.pose_window_s - start
            frames, _ = decode_window(
                video, start, duration, fps=1.0 / config.pose_step_s, gray=False
            )
            best_gap = 99.0
            for frame in frames:
                if config.pose_scale != 1.0:
                    frame = cv2.resize(
                        frame, (0, 0), fx=config.pose_scale, fy=config.pose_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )
                if not detected.pose_landmarks:
                    continue
                landmarks = detected.pose_landmarks[0]
                shoulder_y = min(landmarks[11].y, landmarks[12].y)
                wrist_y = min(landmarks[15].y, landmarks[16].y)
                best_gap = min(best_gap, wrist_y - shoulder_y)
            results.append((best_gap <= config.pose_wrist_shoulder_gap, best_gap))
    return results


def _onset_signal(video: str | Path, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = _extract_audio(video, config.impact_sample_rate)
    size = config.impact_fft_samples
    hop = config.impact_hop_samples
    count = 1 + (len(samples) - size) // hop
    if count < 3:
        return np.empty(0), np.empty(0), np.empty(0)
    frames = np.lib.stride_tricks.sliding_window_view(samples, size)[::hop][:count]
    window = np.hanning(size).astype(np.float32)
    frequencies = np.fft.rfftfreq(size, 1 / config.impact_sample_rate)
    band = (frequencies >= config.impact_band_low_hz) & (frequencies <= config.impact_band_high_hz)
    energy = np.empty(count, np.float32)
    for start in range(0, count, 4_096):
        transformed = np.fft.rfft(frames[start : start + 4_096] * window, axis=1)
        energy[start : start + len(transformed)] = np.abs(transformed[:, band]).mean(axis=1) / size
    floor = _preceding_floor(energy, config.impact_floor_window, config.impact_floor_guard)
    onset = np.maximum(0.0, energy - floor)
    attack = energy / np.maximum(floor, np.finfo(np.float32).eps)
    times = (
        np.arange(count, dtype=np.float64) * hop + size / 2
    ) / config.impact_sample_rate
    return times, onset, attack


def detect_impacts(
    video: str | Path,
    config: Config,
    candidate_video_times: Sequence[float] | None = None,
) -> list[dict[str, float | bool]]:
    """Detect visual swing candidates, pose-gate them, then refine by audio.

    ``candidate_video_times`` is an explicit parity/debug hook.  Normal calls
    are unseeded and run the visual candidate stage.
    """
    signal_times, onset, attack = _onset_signal(video, config)
    if len(signal_times) == 0:
        return []
    candidate_scores: list[float]
    pose_flags: list[bool]
    pose_gate_enabled = False
    if candidate_video_times is not None:
        candidates = [(float(value), 1.0) for value in candidate_video_times]
        pose_flags = [True] * len(candidates)
    else:
        motion_times, motion = motion_signal(video, config)
        candidates = find_motion_candidates(motion_times, motion, config)
        LOG.info("found %d impulsive motion candidates", len(candidates))
        gated = pose_gate_candidates(video, candidates, config)
        pose_gate_enabled = gated is not None
        pose_flags = [True] * len(candidates) if gated is None else [item[0] for item in gated]
        if gated is not None:
            LOG.info("pose gate retained %d/%d candidates", sum(pose_flags), len(candidates))
    indices: list[int] = []
    candidate_scores = []
    retained_pose_flags: list[bool] = []
    for (video_time, motion_score), pose_passed in zip(candidates, pose_flags, strict=True):
        if not pose_passed:
            continue
        expected_audio = video_time + config.av_offset_s
        eligible = np.flatnonzero(
            np.abs(signal_times - expected_audio) <= config.impact_search_radius_s
        )
        if not len(eligible):
            continue
        local = eligible[np.argmax(onset[eligible])]
        if onset[local] < config.impact_min_onset:
            continue
        indices.append(int(local))
        candidate_scores.append(float(motion_score))
        retained_pose_flags.append(bool(pose_passed))
    if not indices:
        return []
    indices_array = np.asarray(indices, dtype=int)
    times = signal_times[indices_array]
    scores = onset[indices_array]
    selected = _dedupe(times, scores, config.impact_min_gap_s)
    if not selected:
        return []
    ceiling = max(float(np.max(scores[selected])), np.finfo(np.float32).eps)
    motion_ceiling = max(
        (candidate_scores[index] for index in selected),
        default=np.finfo(np.float32).eps,
    )
    rows: list[dict[str, float | bool]] = []
    for selected_index in selected:
        audio_time = float(times[selected_index])
        rows.append({
            "t_audio": round(audio_time, 6),
            "t_video": round(max(0.0, audio_time - config.av_offset_s), 6),
            "confidence": round(float(scores[selected_index]) / ceiling, 6),
            "motion_score": round(candidate_scores[selected_index], 6),
            "motion_confidence": round(candidate_scores[selected_index] / motion_ceiling, 6),
            "pose_gated": pose_gate_enabled and retained_pose_flags[selected_index],
        })
    return rows


def select_impacts(impacts: Sequence[dict[str, float]], only: str | None) -> list[dict[str, float]]:
    if not only:
        return list(impacts)
    wanted = {int(value.strip()) for value in only.split(",") if value.strip()}
    if not wanted or min(wanted) < 1:
        raise ValueError("--only uses one-based positive impact numbers")
    return [item for index, item in enumerate(impacts, 1) if index in wanted]


def read_candidate_times(path: str | Path) -> list[float]:
    """Read explicit visual candidate times from common JSON row fields."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("candidate file must contain a JSON list")
    times: list[float] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("accepted", True):
            continue
        for key in ("t_video", "t_impact_s", "time"):
            if key in row:
                times.append(float(row[key]))
                break
    return times


def make_swings(impacts: Sequence[dict[str, float]], config: Config) -> list[Swing]:
    swings: list[Swing] = []
    for index, item in enumerate(impacts, 1):
        impact = float(item["t_video"])
        swings.append(Swing(
            id=index,
            window_start=max(0.0, impact - config.window_pre_s),
            window_end=impact + config.window_post_s,
            impact_t=impact,
        ))
    return swings


def write_impacts(path: str | Path, impacts: Sequence[dict[str, float]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(list(impacts), indent=2) + "\n", encoding="utf-8")
    return destination


def _caption(image: np.ndarray, text: str) -> None:
    cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def write_calibration_strips(
    video: str | Path,
    impacts: Sequence[dict[str, float]],
    out_dir: str | Path,
    config: Config,
    count: int,
) -> list[Path]:
    destination = Path(out_dir) / "qa"
    destination.mkdir(parents=True, exist_ok=True)
    meta = probe(video)
    paths: list[Path] = []
    for ordinal, impact in enumerate(list(impacts)[:count], 1):
        center = float(impact["t_audio"])
        start = max(0.0, center - config.calibration_span_s / 2)
        frames, pts = decode_window(video, start, config.calibration_span_s, fps=meta.fps, gray=False)
        if len(frames) == 0:
            continue
        targets = np.linspace(start, start + config.calibration_span_s, config.calibration_frames)
        indices = [int(np.argmin(np.abs(pts - target))) for target in targets]
        panels: list[np.ndarray] = []
        for frame_index in indices:
            panel = frames[frame_index].copy()
            _caption(panel, f"{pts[frame_index] - center:+.3f}s")
            panels.append(panel)
        strip = cv2.hconcat(panels)
        path = destination / f"calibration-{ordinal:03d}.png"
        if not cv2.imwrite(str(path), strip):
            raise RuntimeError(f"failed to write {path}")
        paths.append(path)
    return paths
