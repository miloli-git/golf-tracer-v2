"""Address-to-top clubhead tracker and label-constrained backswing fit."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..config import Config
from ..impacts import pose_model_path
from ..session import AuditReport, Observation, Swing
from .base import (
    GeometryOverlength, Phase, SpatialArc, SpatialSpline, audit_positions, dp_sweep,
    fit_label_constrained,
)


LOG = logging.getLogger("golftracer.phases.club")
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24


def _pose_series(frames: np.ndarray, config: Config) -> list[np.ndarray | None]:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        model = pose_model_path(config)
    except (ImportError, OSError, RuntimeError) as exc:
        LOG.info("club phase abstains because pose is unavailable: %s", exc)
        return [None] * len(frames)
    height, width = frames.shape[1:3]
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    output: list[np.ndarray | None] = []
    with vision.PoseLandmarker.create_from_options(options) as detector:
        for index, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = detector.detect_for_video(image, int(index * 1000 / 60))
            if not result.pose_landmarks:
                output.append(None)
            else:
                output.append(np.asarray(
                    [(point.x * width, point.y * height) for point in result.pose_landmarks[0]],
                    dtype=np.float32,
                ))
    return output


def _fill_pose(values: list[np.ndarray | None]) -> list[np.ndarray | None]:
    valid = [index for index, value in enumerate(values) if value is not None]
    if not valid:
        return values
    result = list(values)
    for index, value in enumerate(result):
        if value is not None:
            continue
        left = max((item for item in valid if item < index), default=None)
        right = min((item for item in valid if item > index), default=None)
        if left is None:
            result[index] = values[right]  # type: ignore[index]
        elif right is None:
            result[index] = values[left]
        else:
            fraction = (index - left) / (right - left)
            result[index] = (1.0 - fraction) * values[left] + fraction * values[right]  # type: ignore[operator]
    return result


def _club_endpoint(
    gray: np.ndarray, background: np.ndarray, wrist: np.ndarray, scale: float,
    angle_hint: float | None,
) -> tuple[float, float, float] | None:
    """Port of v1's wrist-anchored dark-ridge ray search, at a compact resolution."""
    motion = cv2.dilate(cv2.absdiff(gray, background), np.ones((7, 7), np.uint8))
    angles = np.deg2rad(np.arange(-180.0, 180.0, 1.5))
    if angle_hint is not None:
        wrapped = np.abs((angles - angle_hint + np.pi) % (2 * np.pi) - np.pi)
        angles = angles[wrapped <= np.deg2rad(42.0)]
    radii = np.arange(max(30.0, 1.05 * scale), max(40.0, 2.25 * scale), 3.0)
    if not len(radii) or not len(angles):
        return None
    ca, sa = np.cos(angles), np.sin(angles)
    x = wrist[0] + ca[:, None] * radii[None, :]
    y = wrist[1] + sa[:, None] * radii[None, :]
    xi = np.clip(np.rint(x).astype(int), 0, gray.shape[1] - 1)
    yi = np.clip(np.rint(y).astype(int), 0, gray.shape[0] - 1)
    nx, ny = -sa[:, None], ca[:, None]
    centre = gray[yi, xi].astype(np.float32)
    ridge = np.zeros_like(centre)
    for half_width in (6.0, 11.0, 18.0):
        xp = np.clip(np.rint(x + half_width * nx).astype(int), 0, gray.shape[1] - 1)
        yp = np.clip(np.rint(y + half_width * ny).astype(int), 0, gray.shape[0] - 1)
        xm = np.clip(np.rint(x - half_width * nx).astype(int), 0, gray.shape[1] - 1)
        ym = np.clip(np.rint(y - half_width * ny).astype(int), 0, gray.shape[0] - 1)
        ridge = np.maximum(ridge, np.minimum(gray[yp, xp], gray[ym, xm]).astype(np.float32) - centre)
    support = np.clip((ridge - 5.0) / 24.0, 0.0, 1.0)
    support *= 0.15 + 0.85 * np.clip(motion[yi, xi].astype(np.float32) / 28.0, 0.0, 1.0)
    cumulative = np.cumsum(support - 0.28, axis=1)
    flat = int(np.argmax(cumulative))
    angle_index, radius_index = np.unravel_index(flat, cumulative.shape)
    confidence = float(cumulative[angle_index, radius_index] / max(1, len(radii)))
    if confidence <= 0.035:
        return None
    radius = float(radii[radius_index])
    return (
        float(wrist[0] + ca[angle_index] * radius),
        float(wrist[1] + sa[angle_index] * radius),
        float(angles[angle_index]),
    )


def _find_phases(pose: list[np.ndarray | None], fps: float) -> dict[str, Any]:
    """Exact v1 wrist-motion phase finder (address, takeaway, top)."""
    points = np.asarray(pose, dtype=np.float32)
    wrists = 0.5 * (points[:, L_WRIST] + points[:, R_WRIST])
    shoulders = 0.5 * (points[:, L_SHOULDER] + points[:, R_SHOULDER])
    hips = 0.5 * (points[:, L_HIP] + points[:, R_HIP])
    scale = float(np.median(np.linalg.norm(shoulders - hips, axis=1)))
    lo = int(0.25 * len(points))
    top = lo + int(np.argmin(wrists[lo:, 1]))
    smooth = wrists.copy()
    for column in (0, 1):
        smooth[:, column] = np.convolve(wrists[:, column], np.ones(5) / 5.0, mode="same")
    smooth[:2], smooth[-2:] = wrists[:2], wrists[-2:]
    speed = np.r_[0.0, np.linalg.norm(np.diff(smooth, axis=0), axis=1)] / scale
    start, run = top, 0
    for index in range(top - 1, 0, -1):
        if speed[index] < 0.020:
            run += 1
            if run >= 4:
                start = index + run - 1
                break
        else:
            run = 0
    else:
        start = max(0, top - int(0.9 * fps))
    address = wrists[max(0, start - 3):start + 1].mean(0)
    takeaway = start
    for index in range(start, top):
        if np.linalg.norm(wrists[index] - address) > 0.05 * scale:
            takeaway = max(start, index - 2)
            break
    return {"takeaway": int(takeaway), "top": int(top), "scale": scale, "wrists": wrists}


def club_observations(
    frames: np.ndarray, swing: Swing, config: Config, metadata: dict[str, Any] | None = None,
) -> tuple[list[Observation], int]:
    pose = _fill_pose(_pose_series(frames, config))
    if sum(item is not None for item in pose) < max(4, len(frames) // 2):
        return [], 0
    phase_info = _find_phases(pose, 60.0)
    wrists = phase_info["wrists"]
    scale = float(phase_info["scale"])
    impact_index = int(np.clip(round((swing.impact_t - swing.window_start) * 60.0), 1, len(frames) - 1))
    search_start = int(phase_info["takeaway"])
    top_index = int(phase_info["top"])
    if metadata is not None:
        metadata["pose_scale_px"] = scale
        metadata["takeaway_frame"] = search_start
        metadata["takeaway_t"] = swing.window_start + search_start / 60.0
        metadata["top_frame"] = top_index
        metadata["top_t"] = swing.window_start + top_index / 60.0
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    background = np.median(gray[: max(3, min(search_start + 1, 12))], axis=0).astype(np.uint8)
    observations: list[Observation] = []
    hint = None
    for index in range(search_start, impact_index + 1):
        endpoint = _club_endpoint(gray[index], background, wrists[index], scale, hint)
        if endpoint is None:
            continue
        x, y, hint = endpoint
        observations.append(Observation(
            index, swing.window_start + index / 60.0, x, y,
            confidence=1.0, source="observed",
        ))
    return observations, top_index


def _sample(array: np.ndarray, x: np.ndarray, y: np.ndarray, outside: float = 0.0) -> np.ndarray:
    height, width = array.shape
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    np.clip(xi, 0, width - 1, out=xi)
    np.clip(yi, 0, height - 1, out=yi)
    values = array[yi, xi].astype(np.float32)
    values[~inside] = outside
    return values


def _ray_support(
    gray: np.ndarray, motion: np.ndarray, ox: np.ndarray, oy: np.ndarray,
    angles: np.ndarray, radii: np.ndarray, *, motion_floor: float,
) -> np.ndarray:
    cosine, sine = np.cos(angles), np.sin(angles)
    ox = np.asarray(ox, np.float32).reshape(-1, 1)
    oy = np.asarray(oy, np.float32).reshape(-1, 1)
    x = ox + cosine[:, None] * radii[None, :]
    y = oy + sine[:, None] * radii[None, :]
    nx, ny = -sine[:, None], cosine[:, None]
    centre = np.minimum.reduce([
        _sample(gray, x + offset * nx, y + offset * ny, 255.0)
        for offset in (-2.0, 0.0, 2.0)
    ])
    ridge = None
    for half_width in (6.0, 11.0, 18.0):
        plus = _sample(gray, x + half_width * nx, y + half_width * ny, 0.0)
        minus = _sample(gray, x - half_width * nx, y - half_width * ny, 0.0)
        value = np.minimum(plus, minus) - centre
        ridge = value if ridge is None else np.maximum(ridge, value)
    support = np.clip((ridge - 5.0) / 24.0, 0.0, 1.0)
    return support * (motion_floor + (1.0 - motion_floor) * _sample(motion, x, y, 0.0))


def _align_background(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    first = cv2.resize(background, None, fx=0.25, fy=0.25).astype(np.float32)
    second = cv2.resize(gray, None, fx=0.25, fy=0.25).astype(np.float32)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(first, second)
    except cv2.error:
        return background
    dx, dy = dx * 4.0, dy * 4.0
    if abs(dx) > 40 or abs(dy) > 40:
        return background
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(background, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _motion_map(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    difference = cv2.absdiff(gray, _align_background(gray, background)).astype(np.float32)
    difference = cv2.dilate(difference, np.ones((7, 7), np.uint8))
    return np.clip((difference - 9.0) / 20.0, 0.0, 1.0)


def _curve_evidence(
    gray: np.ndarray, motion: np.ndarray, wrist: np.ndarray, xy: np.ndarray,
) -> np.ndarray:
    """v1 retimer emission: shaft ridge + head motion + local head contrast."""
    vectors = xy - wrist[None, :]
    radius = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    radii = np.arange(12.0, max(32.0, float(radius.max()) + 3.0), 3.0)
    support = _ray_support(
        gray, motion, np.full(len(angles), wrist[0]),
        np.full(len(angles), wrist[1]), angles.astype(np.float32),
        radii.astype(np.float32), motion_floor=0.10,
    )
    cumulative = np.cumsum(support, axis=1)
    end = np.clip(np.searchsorted(radii, radius), 1, len(radii) - 1)
    begin = np.clip((0.18 * end).astype(int), 0, len(radii) - 2)
    row = np.arange(len(xy))
    shaft = (cumulative[row, end] - np.where(begin > 0, cumulative[row, begin - 1], 0.0)) / np.maximum(1, end - begin + 1)
    head_motion = _sample(motion, xy[:, 0], xy[:, 1], 0.0)
    small = cv2.GaussianBlur(gray, (0, 0), 3.0)
    broad = cv2.GaussianBlur(gray, (0, 0), 15.0)
    contrast = np.clip(np.abs(broad.astype(np.float32) - small) / 38.0, 0.0, 1.0)
    head_blob = np.clip(_sample(contrast, xy[:, 0], xy[:, 1], 0.0), 0.0, 1.0) * (0.30 + 0.70 * head_motion)
    raw = 2.2 * shaft + 1.30 * head_motion + 1.15 * head_blob
    median = float(np.median(raw))
    scale = max(0.08, float(np.percentile(raw, 90) - median))
    return np.clip((raw - median) / scale, -2.5, 4.0).astype(np.float32)


def retime_spatial_arc(
    geometry: SpatialArc,
    frames: np.ndarray,
    *,
    fps: float,
    start_t: float,
    labels: Sequence[Any],
    config: Config,
    background: np.ndarray | None = None,
    debug: dict[str, Any] | None = None,
    wrists: np.ndarray | None = None,
    initial_speed_px: float | None = None,
    initial_speed_weight: float = 0.0,
    initial_speed_frames: int = 0,
    soft_pins: Sequence[Observation] | None = None,
    soft_pin_weight: float = 0.0,
    soft_pin_cap: float = 0.0,
) -> list[Observation]:
    """Exact v1 image emissions, label pins, endpoints, and DP transitions.

    ``soft_pins`` are timed detector observations: each adds a confidence-scaled,
    capped arc-distance penalty on its own frame row so the walk is timed by the
    detections without any single stray proposal dominating. Human ``labels``
    keep their exact v1 pin behaviour.
    """
    if not len(frames):
        return []
    height, width = frames.shape[1:3]
    arc_limit_px = config.club_max_arc_frame_diagonals * float(np.hypot(width, height))
    if geometry.length > arc_limit_px:
        raise GeometryOverlength(geometry.length, arc_limit_px)
    xy = geometry.xy
    if wrists is None:
        pose = _fill_pose(_pose_series(frames, config))
        if not pose or any(item is None for item in pose):
            raise RuntimeError("club retiming requires pose on every filled frame")
        landmarks = np.asarray(pose, np.float32)
        wrists = 0.5 * (landmarks[:, L_WRIST] + landmarks[:, R_WRIST])
    else:
        wrists = np.asarray(wrists, dtype=float)
        if len(wrists) != len(frames):
            raise ValueError("wrists must have one row per retimed frame")
    ray_radius_px = max(
        float(np.linalg.norm(xy - wrist[None, :], axis=1).max())
        for wrist in wrists
    )
    if ray_radius_px > arc_limit_px:
        raise GeometryOverlength(
            geometry.length, arc_limit_px, ray_radius_px=ray_radius_px,
        )
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    if background is None:
        background = np.median(gray[:max(1, min(5, len(gray)))], axis=0).astype(np.uint8)
    emissions = []
    for index, frame in enumerate(gray):
        motion = _motion_map(frame, background)
        emissions.append(_curve_evidence(frame, motion, wrists[index], xy))
    score = np.asarray(emissions, np.float32)
    pin_rows: list[int] = []
    for label in labels:
        label_t = float(label.get("t") if isinstance(label, Mapping) else label.t)
        label_x = float(label.get("x") if isinstance(label, Mapping) else label.x)
        label_y = float(label.get("y") if isinstance(label, Mapping) else label.y)
        row = int(np.clip(round((label_t - start_t) * fps), 0, len(score) - 1))
        target = geometry.nearest_arc((label_x, label_y))
        distance = np.abs(geometry.arc - target)
        score[row] -= (3.2 * (distance / config.club_label_tolerance_px) ** 2).astype(np.float32)
        score[row, distance > 2 * config.club_label_tolerance_px] -= 80.0
        pin_rows.append(row)
    if soft_pins and soft_pin_weight > 0.0:
        for item in soft_pins:
            row = int(round((float(item.t) - start_t) * fps))
            if row < 0 or row >= len(score):
                continue
            confidence = float(item.confidence if item.confidence is not None else 1.0)
            target = geometry.nearest_arc((float(item.x), float(item.y)))
            distance = np.abs(geometry.arc - target)
            penalty = soft_pin_weight * (distance / config.club_label_tolerance_px) ** 2
            if soft_pin_cap > 0.0:
                penalty = np.minimum(penalty, soft_pin_cap)
            score[row] -= (confidence * penalty).astype(np.float32)
    score[0, 1:] = -1e6
    score[-1, :-1] = -1e6
    acceleration = np.full(len(frames), config.club_retime_frame_accel_weight)
    if pin_rows:
        acceleration[max(pin_rows) + 1:] = config.club_retime_delivery_accel_weight
    arc_step = geometry.arc[1] - geometry.arc[0]
    max_step = min(len(geometry.arc) - 1, max(1, int(np.ceil(config.club_max_arc_step_px / max(1e-6, arc_step)))))
    path = dp_sweep(
        score, 0, max_step, config.club_retime_accel_weight,
        config.club_retime_start_weight, config.club_retime_decel_weight,
        acceleration,
        None if initial_speed_px is None else int(round(initial_speed_px / max(1e-6, arc_step))),
        initial_speed_weight,
        initial_speed_frames,
    )
    selected = xy[path]
    if debug is not None:
        debug["emissions"] = score
        debug["path"] = path
        debug["arc"] = geometry.arc[path]
        debug["geometry"] = geometry
    label_frames = set(pin_rows)
    return [
        Observation(index, start_t + index / fps, float(point[0]), float(point[1]),
                    source="labelled" if index in label_frames else "interpolated")
        for index, point in enumerate(selected)
    ]


def retime_club_with_pose(
    spline: SpatialSpline, frames: np.ndarray, *, fps: float, start_t: float,
    config: Config,
) -> list[Observation]:
    records = []
    for index in range(len(frames)):
        t = start_t + index / fps
        x, y = spline.xy_at_time(t)
        records.append({"t_s": t, "x": float(x), "y": float(y)})
    geometry = SpatialArc.from_records(records, config)
    return retime_spatial_arc(
        geometry, frames, fps=fps, start_t=start_t, labels=spline.labels,
        config=config,
    )


class BackswingPhase(Phase):
    name = "backswing"

    def __init__(self, config: Config = Config()):
        self.config = config

    def track(self, frames: np.ndarray, swing: Swing, config: Config) -> list[Observation]:
        observations, top = club_observations(frames, swing, config, swing.metadata)
        return [item for item in observations if item.frame_index <= top]

    def fit(self, observations: Sequence[Observation], labels: Sequence[Any]) -> SpatialSpline | None:
        return fit_label_constrained(
            self.name, observations, labels, self.config, observation_weight=0.65,
        )

    def retime(self, spline: SpatialSpline, frames: np.ndarray, *, fps: float, start_t: float) -> list[Observation]:
        return retime_club_with_pose(spline, frames, fps=fps, start_t=start_t, config=self.config)

    def gates(self) -> Mapping[str, float | bool]:
        return {"label_tolerance_px": self.config.club_label_tolerance_px}

    def audit(self, positions: Sequence[Observation], labels: Sequence[Any], observations: Sequence[Observation] = ()) -> AuditReport:
        return audit_positions(positions, labels, observations)
