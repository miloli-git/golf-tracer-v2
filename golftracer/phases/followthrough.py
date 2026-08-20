"""Impact-to-finish clubhead phase with a shared impact anchor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..config import Config
from ..session import AuditReport, Observation, Swing
from .base import Phase, SpatialArc, SpatialSpline, audit_positions, fit_label_constrained
from .backswing import (
    L_WRIST, R_WRIST, _club_endpoint, _fill_pose, _pose_series,
    retime_spatial_arc,
)


@dataclass(frozen=True)
class Finish:
    frame_index: int
    t: float
    reason: str


def detect_finish(
    observations: Sequence[Observation],
    *,
    frame_count: int,
    frame_shape: tuple[int, int],
    fps: float,
    config: Config,
) -> Finish | None:
    """Return the first supported rest, or the last point before frame exit."""
    ordered = sorted(observations, key=lambda item: item.frame_index)
    if not ordered:
        return None
    height, width = frame_shape
    margin = config.follow_exit_margin_px
    first_eligible = int(round(config.follow_finish_min_time_s * fps))
    run = 0
    for previous, current in zip(ordered, ordered[1:]):
        gap = current.frame_index - previous.frame_index
        if gap != 1 or current.frame_index < first_eligible:
            run = 0
            continue
        speed = float(np.hypot(current.x - previous.x, current.y - previous.y) * fps)
        run = run + 1 if speed <= config.follow_finish_velocity_floor_px_s else 0
        if run >= config.follow_finish_stationary_frames:
            return Finish(current.frame_index, current.t, "velocity_floor")
    last = ordered[-1]
    at_edge = (
        last.x <= margin or last.y <= margin
        or last.x >= width - 1 - margin or last.y >= height - 1 - margin
    )
    missing = frame_count - 1 - last.frame_index
    if at_edge or missing >= config.follow_exit_missing_frames:
        return Finish(last.frame_index, last.t, "frame_exit")
    return Finish(last.frame_index, last.t, "last_evidence")


class FollowthroughPhase(Phase):
    name = "followthrough"

    def __init__(
        self,
        config: Config = Config(),
        *,
        impact_point: tuple[float, float] | None = None,
        impact_speed_px_per_frame: float | None = None,
    ):
        self.config = config
        self.impact_point = impact_point
        self.impact_speed_px_per_frame = impact_speed_px_per_frame
        self.impact_t: float | None = None
        self.finish_t: float | None = None
        self.finish_reason: str | None = None
        self.background: np.ndarray | None = None
        self.post_wrists: np.ndarray | None = None
        self.retime_debug: dict[str, Any] = {}
        self.detector_observations: list[Observation] = []

    def track(self, frames: np.ndarray, swing: Swing, config: Config) -> list[Observation]:
        if not len(frames):
            return []
        pose = _fill_pose(_pose_series(frames, config))
        if not pose or any(item is None for item in pose):
            return []
        landmarks = np.asarray(pose, dtype=np.float32)
        wrists = 0.5 * (landmarks[:, L_WRIST] + landmarks[:, R_WRIST])
        impact = int(np.clip(
            round((swing.impact_t - swing.window_start) * 60.0), 0, len(frames) - 1,
        ))
        self.impact_t = float(swing.impact_t)
        self.post_wrists = wrists[impact:]
        gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
        quiet_count = max(1, min(12, impact))
        self.background = np.median(gray[:quiet_count], axis=0).astype(np.uint8)

        # Mirror the backswing temporal search: establish the slow finish first,
        # then follow the angle hint backwards toward the impact.
        reverse: list[Observation] = []
        hint = None
        scale = float(np.median(np.linalg.norm(
            0.5 * (landmarks[:, 11] + landmarks[:, 12])
            - 0.5 * (landmarks[:, 23] + landmarks[:, 24]), axis=1,
        )))
        for absolute in range(len(frames) - 1, impact - 1, -1):
            endpoint = _club_endpoint(
                gray[absolute], self.background, wrists[absolute], scale, hint,
            )
            if endpoint is None:
                continue
            x, y, hint = endpoint
            relative = absolute - impact
            reverse.append(Observation(
                relative, swing.impact_t + relative / 60.0, x, y,
                confidence=1.0, source="observed",
            ))
        observations = sorted(reverse, key=lambda item: item.frame_index)
        finish = detect_finish(
            observations, frame_count=len(frames) - impact,
            frame_shape=frames.shape[1:3], fps=60.0, config=config,
        )
        if finish is not None:
            self.finish_t, self.finish_reason = finish.t, finish.reason
            observations = [item for item in observations if item.frame_index <= finish.frame_index]
        return observations

    def fit(self, observations: Sequence[Observation], labels: Sequence[Any]) -> SpatialSpline | None:
        if self.impact_point is None or self.impact_t is None:
            return None
        if labels:
            ordered_labels = sorted(
                labels,
                key=lambda item: int(item.get("frame_index") if isinstance(item, Mapping) else item.frame_index),
            )
            last = ordered_labels[-1]
            last_frame = int(last.get("frame_index") if isinstance(last, Mapping) else last.frame_index)
            self.finish_t = float(last.get("t") if isinstance(last, Mapping) else last.t)
            slow_frames = 0
            for previous, current in zip(ordered_labels, ordered_labels[1:]):
                previous_frame = int(previous.get("frame_index") if isinstance(previous, Mapping) else previous.frame_index)
                current_frame = int(current.get("frame_index") if isinstance(current, Mapping) else current.frame_index)
                dt_frames = current_frame - previous_frame
                if dt_frames <= 0:
                    continue
                px = float(np.hypot(
                    float(current.get("x") if isinstance(current, Mapping) else current.x)
                    - float(previous.get("x") if isinstance(previous, Mapping) else previous.x),
                    float(current.get("y") if isinstance(current, Mapping) else current.y)
                    - float(previous.get("y") if isinstance(previous, Mapping) else previous.y),
                ))
                speed = px * 60.0 / dt_frames
                slow_frames = slow_frames + dt_frames if speed <= self.config.follow_finish_velocity_floor_px_s else 0
            expected_last = int(round(self.config.follow_through_post_s * 60.0))
            if slow_frames >= self.config.follow_finish_stationary_frames:
                self.finish_reason = "velocity_floor"
            elif last_frame < expected_last - 2:
                self.finish_reason = "frame_exit"
            else:
                self.finish_reason = "last_evidence"
            observations = [item for item in observations if item.t <= self.finish_t + 1e-7]
        self.detector_observations = [
            item for item in observations if getattr(item, "source", None) == "detector"
        ]
        fit_labels = [
            item for item in labels
            if abs(float(item.get("t") if isinstance(item, Mapping) else item.t) - self.impact_t) > 1e-7
        ]
        spline = fit_label_constrained(
            self.name, observations, fit_labels, self.config,
            observation_weight=self.config.club_tracker_weight_followthrough,
            forced_start=(self.impact_t, self.impact_point[0], self.impact_point[1], 0),
            bias=self.config.club_tracker_bias_followthrough,
            forced_start_source="impact_anchor",
            forced_start_calibration_phase="followthrough",
        )
        return spline

    def retime(
        self, spline: SpatialSpline, frames: np.ndarray, *, fps: float, start_t: float,
    ) -> list[Observation]:
        finish_t = min(
            self.finish_t if self.finish_t is not None else spline.time_knots[-1],
            float(spline.time_knots[-1]),
        )
        count = max(2, int(round((finish_t - start_t) * fps)) + 1)
        records = []
        for index in range(count):
            t = start_t + index / fps
            x, y = spline.xy_at_time(t)
            records.append({"t_s": t, "x": float(x), "y": float(y)})
        geometry = SpatialArc.from_records(records, self.config)
        wrists = self.post_wrists[:count] if self.post_wrists is not None else None
        debug: dict[str, Any] = {}
        positions = retime_spatial_arc(
            geometry, frames[:count], fps=fps, start_t=start_t,
            labels=spline.labels, config=self.config, background=self.background,
            wrists=wrists, debug=debug,
            initial_speed_px=self.impact_speed_px_per_frame,
            initial_speed_weight=self.config.follow_retime_initial_speed_weight,
            initial_speed_frames=self.config.follow_retime_initial_speed_frames,
            soft_pins=self.detector_observations,
            soft_pin_weight=self.config.follow_detector_pin_weight,
            soft_pin_cap=self.config.follow_detector_pin_cap,
        )
        if positions and self.impact_point is not None:
            positions[0].x, positions[0].y = self.impact_point
        self.retime_debug = debug
        return positions

    def gates(self) -> Mapping[str, float | bool]:
        return {
            "label_tolerance_px": self.config.club_label_tolerance_px,
            "frame_exit_is_finish": True,
            "no_extrapolation": True,
        }

    def audit(
        self, positions: Sequence[Observation], labels: Sequence[Any],
        observations: Sequence[Observation] = (),
    ) -> AuditReport:
        return audit_positions(positions, labels, observations)
