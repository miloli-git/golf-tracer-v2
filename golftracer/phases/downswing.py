"""Top-to-impact clubhead phase with shared cusp and exported impact point."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..config import Config
from ..session import AuditReport, Observation, Swing
from .base import Phase, SpatialSpline, audit_positions, fit_label_constrained
from .backswing import club_observations, retime_club_with_pose


class DownswingPhase(Phase):
    name = "downswing"

    def __init__(self, config: Config = Config()):
        self.config = config
        self.impact_point: tuple[float, float] | None = None

    def track(self, frames: np.ndarray, swing: Swing, config: Config) -> list[Observation]:
        observations, detected_top = club_observations(frames, swing, config)
        top = int(swing.metadata.get("top_frame", detected_top))
        # Delivery interpolations are never promoted to observations (LESSONS 3).
        return [item for item in observations if item.frame_index >= top and item.source == "observed"]

    def fit(self, observations: Sequence[Observation], labels: Sequence[Any]) -> SpatialSpline | None:
        spline = fit_label_constrained(
            self.name, observations, labels, self.config, observation_weight=2.0,
        )
        if spline is not None:
            endpoint = spline.xy_at_arc(spline.length)
            self.impact_point = (float(endpoint[0]), float(endpoint[1]))
        return spline

    def retime(self, spline: SpatialSpline, frames: np.ndarray, *, fps: float, start_t: float) -> list[Observation]:
        positions = retime_club_with_pose(
            spline, frames, fps=fps, start_t=start_t, config=self.config,
        )
        if positions:
            self.impact_point = (positions[-1].x, positions[-1].y)
        return positions

    def gates(self) -> Mapping[str, float | bool]:
        return {
            "label_tolerance_px": self.config.club_label_tolerance_px,
            "drop_delivery_interpolations": True,
        }

    def audit(self, positions: Sequence[Observation], labels: Sequence[Any], observations: Sequence[Observation] = ()) -> AuditReport:
        return audit_positions(positions, labels, observations)
