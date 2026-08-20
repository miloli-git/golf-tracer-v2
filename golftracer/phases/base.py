"""Shared phase contract and the faithfully transcribed v1 club spatial fit."""

from __future__ import annotations

# Legacy v1 label sets mark human clicks with this source string; keep accepting it.
LEGACY_HUMAN_SOURCE = "mi" "lo_label"
HUMAN_SOURCES = {"human", "accepted", "corrected", LEGACY_HUMAN_SOURCE}

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator, splev, splprep

from ..config import Config
from ..session import (
    AuditFrame, AuditReport, INTERPOLATED, LABELLED, OBSERVED, UNCONSTRAINED,
    Observation, Swing,
)


@dataclass
class Constraint:
    frame_index: int
    t: float
    x: float
    y: float
    source: str
    weight: float
    calibration_phase: str = ""
    pseudo_time: float | None = None
    original_x: float | None = None
    original_y: float | None = None

    @property
    def fit_time(self) -> float:
        return self.t if self.pseudo_time is None else self.pseudo_time

    @property
    def hard(self) -> bool:
        return self.source in {"top_join", "impact_anchor"}


@dataclass
class SpatialSpline:
    """FITPACK curve sampled and numerically parameterised by pixel arc length."""

    tck: tuple[Any, ...]
    parameter: np.ndarray
    arc: np.ndarray
    xy_samples: np.ndarray
    constraints: list[Constraint]
    time_knots: np.ndarray
    arc_knots: np.ndarray
    phase: str
    labels: list[Constraint] = field(default_factory=list)
    accepted: list[Constraint] = field(default_factory=list)
    rejected: list[Constraint] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return float(self.arc[-1])

    def xy_at_u(self, value: float | np.ndarray) -> np.ndarray:
        values = np.atleast_1d(np.asarray(value, dtype=float))
        x, y = splev(np.clip(values, 0.0, 1.0), self.tck)
        result = np.column_stack((x, y))
        return result[0] if np.ndim(value) == 0 else result

    def xy_at_arc(self, value: float | np.ndarray) -> np.ndarray:
        values = np.atleast_1d(np.asarray(value, dtype=float))
        u = np.interp(np.clip(values, 0.0, self.length), self.arc, self.parameter)
        result = self.xy_at_u(u)
        return result[0] if np.ndim(value) == 0 else result

    def arc_at_time(self, value: float | np.ndarray) -> np.ndarray:
        return np.interp(value, self.time_knots, self.arc_knots)

    def xy_at_time(self, value: float | np.ndarray) -> np.ndarray:
        return self.xy_at_arc(self.arc_at_time(value))

    def nearest_arc(self, point: tuple[float, float]) -> float:
        delta = self.xy_samples - np.asarray(point, dtype=float)
        return float(self.arc[int(np.argmin(np.einsum("ij,ij->i", delta, delta)))])

    def distance_to_curve(self, x: float, y: float) -> float:
        delta = self.xy_samples - np.asarray((x, y), dtype=float)
        return float(np.sqrt(np.min(np.einsum("ij,ij->i", delta, delta))))


@dataclass
class SpatialArc:
    """v1 retimer geometry reconstructed from refit records using PCHIP."""

    arc: np.ndarray
    xy: np.ndarray

    @classmethod
    def from_records(
        cls, records: Sequence[Mapping[str, Any]], config: Config = Config()
    ) -> "SpatialArc":
        points = np.asarray([[float(item["x"]), float(item["y"])] for item in records])
        chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        keep = np.r_[True, np.diff(chord) > 1e-7]
        chord, points = chord[keep], points[keep]
        if len(points) < 2:
            raise ValueError("spatial phase curve has fewer than two distinct points")
        x_fit = PchipInterpolator(chord, points[:, 0])
        y_fit = PchipInterpolator(chord, points[:, 1])
        parameter = np.linspace(0.0, chord[-1], max(8192, len(points) * 512))
        dense_xy = np.column_stack((x_fit(parameter), y_fit(parameter)))
        dense_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense_xy, axis=0), axis=1))]
        count = max(2, int(math.ceil(dense_arc[-1] / config.club_arc_bin_px)) + 1)
        arc = np.linspace(0.0, dense_arc[-1], count)
        at_arc = np.interp(arc, dense_arc, parameter)
        xy = np.column_stack((x_fit(at_arc), y_fit(at_arc)))
        return cls(arc.astype(float), xy.astype(float))

    @property
    def length(self) -> float:
        return float(self.arc[-1])

    def xy_at(self, value: float | np.ndarray) -> np.ndarray:
        values = np.atleast_1d(np.asarray(value, dtype=float))
        result = np.column_stack((
            np.interp(values, self.arc, self.xy[:, 0]),
            np.interp(values, self.arc, self.xy[:, 1]),
        ))
        return result[0] if np.ndim(value) == 0 else result

    def nearest_arc(self, point: tuple[float, float]) -> float:
        delta = self.xy - np.asarray(point, dtype=float)
        return float(self.arc[int(np.argmin(np.einsum("ij,ij->i", delta, delta)))])


class Phase(ABC):
    name: str

    @abstractmethod
    def track(self, frames: np.ndarray, swing: Swing, config: Config) -> list[Observation]:
        raise NotImplementedError

    @abstractmethod
    def fit(self, observations: Sequence[Observation], labels: Sequence[Any]) -> SpatialSpline | None:
        raise NotImplementedError

    @abstractmethod
    def retime(self, spline: SpatialSpline, frames: np.ndarray, *, fps: float, start_t: float) -> list[Observation]:
        raise NotImplementedError

    @abstractmethod
    def gates(self) -> Mapping[str, float | bool]:
        raise NotImplementedError

    @abstractmethod
    def audit(self, positions: Sequence[Observation], labels: Sequence[Any], observations: Sequence[Observation] = ()) -> AuditReport:
        raise NotImplementedError


def _label_value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def constraints_from_labels(labels: Sequence[Any], weight: float) -> list[Constraint]:
    """Normalise labels; later rows replace collisions and human rows always win."""
    by_identity: dict[tuple[str, int], Constraint] = {}
    for item in labels:
        frame = int(_label_value(item, "frame_index"))
        source = str(_label_value(item, "source", "human"))
        phase = str(_label_value(item, "calibration_phase", _label_value(item, "phase", "")))
        candidate = Constraint(
            frame, float(_label_value(item, "t")), float(_label_value(item, "x")),
            float(_label_value(item, "y")), source, weight, phase,
        )
        key = (str(_label_value(item, "phase", phase)), frame)
        incumbent = by_identity.get(key)
        if incumbent is None or source == "human" or incumbent.source != "human":
            by_identity[key] = candidate
    return sorted(by_identity.values(), key=lambda item: (item.fit_time, item.frame_index))


def calibration_phase(t: float, takeaway_t: float, top_t: float, impact_t: float) -> str:
    if t <= top_t + 1e-9:
        progress = (t - takeaway_t) / max(top_t - takeaway_t, 1e-9)
        if progress <= 0.30:
            return "takeaway"
        if progress < 0.80:
            return "mid-backswing"
        return "top"
    progress = (t - top_t) / max(impact_t - top_t, 1e-9)
    return "downswing" if progress <= 0.65 else "delivery"


def dedupe_supports(supports: Sequence[Constraint]) -> list[Constraint]:
    ordered = sorted(supports, key=lambda item: item.fit_time)
    result: list[Constraint] = []
    for item in ordered:
        if result and abs(item.fit_time - result[-1].fit_time) < 1e-9:
            challenger_label = item.source in HUMAN_SOURCES
            incumbent_label = result[-1].source in HUMAN_SOURCES
            if challenger_label != incumbent_label:
                wins = challenger_label
            else:
                wins = item.weight >= result[-1].weight
            if wins:
                result[-1] = item
            continue
        result.append(item)
    return result


class InsufficientSupport(ValueError):
    """Raised when a phase has too few supports to fit; callers abstain the club track."""


class GeometryOverlength(ValueError):
    """Raised before retiming allocations when a fitted arc is not frame-plausible."""

    reason = "geometry_overlength"

    def __init__(
        self, length_px: float, limit_px: float, *, ray_radius_px: float | None = None,
    ):
        self.length_px = float(length_px)
        self.limit_px = float(limit_px)
        self.ray_radius_px = None if ray_radius_px is None else float(ray_radius_px)
        measured_px = max(
            self.length_px,
            self.ray_radius_px if self.ray_radius_px is not None else 0.0,
        )
        super().__init__(
            f"{self.reason}: emission geometry {measured_px:.1f}px exceeds "
            f"{self.limit_px:.1f}px limit"
        )


def fit_curve(
    supports: Sequence[Constraint], config: Config, label_smoothing_px: float
) -> SpatialSpline:
    ordered = dedupe_supports(supports)
    fit_times = np.asarray([item.fit_time for item in ordered], dtype=float)
    if len(ordered) < 2 or np.any(np.diff(fit_times) <= 0):
        raise ValueError("support timestamps must be strictly increasing")
    u = (fit_times - fit_times[0]) / (fit_times[-1] - fit_times[0])
    xy = np.asarray([(item.x, item.y) for item in ordered], dtype=float)
    weights = np.asarray([item.weight for item in ordered], dtype=float)
    def _tolerance(item: Constraint) -> float:
        if item.source in HUMAN_SOURCES:
            return label_smoothing_px
        if item.source == "detector":
            return config.club_fit_tolerance_detector_px
        return config.club_fit_tolerance_px

    smoothing = sum(
        (item.weight * _tolerance(item)) ** 2
        for item in ordered if not item.hard
    )
    degree = min(3, len(ordered) - 1)
    tck, fitted_u = splprep([xy[:, 0], xy[:, 1]], u=u, w=weights, k=degree, s=max(0.0, smoothing))
    parameter = np.linspace(0.0, 1.0, config.club_spline_samples)
    sx, sy = splev(parameter, tck)
    sampled = np.column_stack((sx, sy))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(sampled, axis=0), axis=1))]
    arc_knots = np.interp(fitted_u, parameter, arc)
    labels = [item for item in ordered if item.source in HUMAN_SOURCES]
    return SpatialSpline(tck, parameter, arc, sampled, ordered, fit_times, arc_knots, "", labels)


def support_u(curve: SpatialSpline, target: Constraint) -> float:
    for item, arc in zip(curve.constraints, curve.arc_knots, strict=True):
        if item is target:
            return float(np.interp(arc, curve.arc, curve.parameter))
    raise ValueError("support was not used by curve")


def choose_label_smoothing(
    phase_supports: Sequence[Sequence[Constraint]], config: Config
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for smoothing in config.club_label_smoothing_candidates_px:
        held_out_residuals: list[float] = []
        training_residuals: list[float] = []
        for supports in phase_supports:
            curve = fit_curve(supports, config, smoothing)
            for support in supports:
                if support.source not in HUMAN_SOURCES:
                    continue
                fitted = curve.xy_at_u(support_u(curve, support))
                training_residuals.append(float(np.linalg.norm(fitted - (support.x, support.y))))
                remaining = [item for item in supports if item is not support]
                if len(remaining) < 2:
                    continue
                predicted = fit_curve(remaining, config, smoothing).xy_at_time(support.fit_time)
                held_out_residuals.append(float(np.linalg.norm(predicted - (support.x, support.y))))
        if not held_out_residuals:
            continue
        data = np.asarray(held_out_residuals)
        training_max = max(training_residuals, default=float("inf"))
        rows.append({
            "smoothing_px": float(smoothing),
            "loo_rmse_px": float(np.sqrt(np.mean(data * data))),
            "training_max_px": float(training_max),
            "eligible": training_max <= config.club_max_training_label_residual_px,
        })
    if not rows:
        return config.club_label_smoothing_px, []
    pool = [row for row in rows if row["eligible"]] or rows
    chosen = min(pool, key=lambda row: (row["loo_rmse_px"], -row["smoothing_px"]))
    for row in rows:
        row["chosen"] = row is chosen
    return float(chosen["smoothing_px"]), rows


def _phase_weight(phase: str, config: Config) -> float:
    return {
        "takeaway": config.club_tracker_weight_takeaway,
        "mid-backswing": config.club_tracker_weight_mid_backswing,
        "top": config.club_tracker_weight_top,
        "downswing": config.club_tracker_weight_downswing,
        "followthrough": config.club_tracker_weight_followthrough,
        "delivery": 0.0,
    }[phase]


def _phase_gate(phase: str, config: Config) -> tuple[float, float]:
    return {
        "takeaway": (config.club_outlier_floor_takeaway_px, config.club_outlier_cap_takeaway_px),
        "mid-backswing": (config.club_outlier_floor_mid_backswing_px, config.club_outlier_cap_mid_backswing_px),
        "top": (config.club_outlier_floor_top_px, config.club_outlier_cap_top_px),
        "downswing": (config.club_outlier_floor_downswing_px, config.club_outlier_cap_downswing_px),
        "followthrough": (config.club_outlier_floor_followthrough_px, config.club_outlier_cap_followthrough_px),
        "delivery": (0.0, 0.0),
    }[phase]


def robust_threshold(values: Sequence[float], phase: str, config: Config) -> float:
    floor, cap = _phase_gate(phase, config)
    if not values:
        return floor
    data = np.asarray(values, dtype=float)
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    return float(np.clip(median + 4.0 * 1.4826 * mad, floor, cap))


def reject_and_fit(
    hard: list[Constraint], candidates: list[Constraint], config: Config,
    label_smoothing_px: float,
) -> tuple[SpatialSpline, list[Constraint], list[Constraint], dict[str, float]]:
    seed = hard
    if len(dedupe_supports(hard)) < 2:
        # Unlabelled swing (v1 only ever refit labelled swings): seed the robust
        # rejection loop from the tracker's own points so the arc still renders.
        seed = [*hard, *candidates]
    if len(dedupe_supports(seed)) < 2:
        raise InsufficientSupport("fewer than two supports for the phase fit")
    initial = fit_curve(seed, config, label_smoothing_px)
    distances = [float(np.linalg.norm(initial.xy_at_time(item.t) - (item.x, item.y))) for item in candidates]
    thresholds = {
        phase: robust_threshold([distance for item, distance in zip(candidates, distances) if item.calibration_phase == phase], phase, config)
        for phase in ("takeaway", "mid-backswing", "top", "downswing", "followthrough")
    }
    retained = [item for item, distance in zip(candidates, distances) if distance <= thresholds[item.calibration_phase]]
    rejected = [item for item in candidates if item not in retained]
    for _ in range(8):
        curve = fit_curve([*hard, *retained], config, label_smoothing_px)
        newly = [
            item for item in retained
            if float(np.linalg.norm(curve.xy_at_time(item.t) - (item.x, item.y))) > thresholds[item.calibration_phase]
        ]
        if not newly:
            break
        retained = [item for item in retained if item not in newly]
        rejected.extend(newly)
    curve = fit_curve([*hard, *retained], config, label_smoothing_px)
    curve.accepted, curve.rejected, curve.thresholds = retained, rejected, thresholds
    return curve, retained, rejected, thresholds


def phase_bias(
    observations: Sequence[Observation], labels: Sequence[Constraint]
) -> tuple[float, float]:
    detected = [item for item in observations if item.source in {"observed", "detected"}]
    deltas: list[tuple[float, float]] = []
    for label in labels:
        candidates = [item for item in detected if item.frame_index == label.frame_index]
        if not candidates:
            candidates = [item for item in detected if abs(item.t - label.t) < 1e-7]
        if candidates:
            item = candidates[0]
            deltas.append((label.x - item.x, label.y - item.y))
    if not deltas:
        return 0.0, 0.0
    median = np.median(np.asarray(deltas), axis=0)
    return float(median[0]), float(median[1])


def fit_label_constrained(
    phase: str,
    observations: Sequence[Observation],
    labels: Sequence[Any],
    config: Config,
    *,
    observation_weight: float,
    forced_start: tuple[float, float, float, int] | None = None,
    forced_end: tuple[float, float, float, int] | None = None,
    label_smoothing_px: float | None = None,
    bias: tuple[float, float] | None = None,
    forced_start_source: str = "top_join",
    forced_start_calibration_phase: str = "top",
) -> SpatialSpline | None:
    """Compatibility entry point using the exact v1 weighting and rejection loop."""
    human = constraints_from_labels(labels, config.club_label_weight)
    bx, by = phase_bias(observations, human) if bias is None else bias
    hard = list(human)
    if forced_start is not None:
        t, x, y, frame = forced_start
        hard.append(Constraint(
            frame, t, x, y, forced_start_source, config.club_hard_weight,
            forced_start_calibration_phase,
        ))
    if forced_end is not None:
        t, x, y, frame = forced_end
        hard.append(Constraint(frame, t, x, y, "impact_anchor", config.club_hard_weight, "delivery"))
    hard = dedupe_supports(hard)
    labelled_frames = {item.frame_index for item in human}
    candidates = [
        Constraint(
            item.frame_index, item.t, item.x + bx, item.y + by,
            "detector" if item.source == "detector" else "tracker_detected",
            observation_weight, phase, original_x=item.x, original_y=item.y,
        )
        for item in observations
        if item.source in {"observed", "detected", "detector"}
        and item.frame_index not in labelled_frames
    ]
    if len(dedupe_supports([*hard, *candidates])) < 2:
        return None
    curve, _, _, _ = reject_and_fit(hard, candidates, config, label_smoothing_px if label_smoothing_px is not None else config.club_label_smoothing_px)
    curve.phase = phase
    return curve


def dp_sweep(
    emissions: np.ndarray,
    j0: int,
    max_step: int,
    accel_w: float,
    start_w: float = 3.0,
    decel_w: float = 0.0,
    accel_weights: Sequence[float] | None = None,
    initial_step: int | None = None,
    initial_step_weight: float = 0.0,
    initial_step_frames: int = 0,
) -> np.ndarray:
    """Exact v1 second-order monotone DP over (previous step, arc state)."""
    scores = np.asarray(emissions, dtype=np.float32)
    n, bins = scores.shape
    max_step = min(bins - 1, max(0, int(max_step)))
    steps = np.arange(max_step + 1)
    neg = -1e9
    current = np.full((len(steps), bins), neg, np.float32)
    back = np.zeros((n, len(steps), bins), np.int16)
    initial_penalty = start_w * (np.abs(np.arange(bins) - j0) / max(1.0, bins / 6.0)) ** 2
    if initial_step is None:
        current[0] = scores[0] - initial_penalty
    else:
        initial_step = int(np.clip(initial_step, 0, max_step))
        for speed_index, step in enumerate(steps):
            speed_penalty = initial_step_weight * float(step - initial_step) ** 2
            current[speed_index] = scores[0] - initial_penalty - speed_penalty
    if accel_weights is not None and len(accel_weights) != n:
        raise ValueError("accel_weights must have one entry per frame")
    for row in range(1, n):
        nxt = np.full((len(steps), bins), neg, np.float32)
        for speed_index, step in enumerate(steps):
            speed_change = (steps[:, None] - step).astype(np.float32)
            local_accel = accel_w if accel_weights is None else float(accel_weights[row])
            penalty = local_accel * speed_change**2
            if decel_w:
                penalty += decel_w * np.maximum(speed_change, 0.0) ** 2
            transition = current - penalty
            source = transition.max(0)
            source_speed = transition.argmax(0).astype(np.int16)
            if step == 0:
                shifted, shifted_speed = source, source_speed
            else:
                shifted = np.full(bins, neg, np.float32)
                shifted_speed = np.zeros(bins, np.int16)
                shifted[step:] = source[:-step]
                shifted_speed[step:] = source_speed[:-step]
            nxt[speed_index] = shifted + scores[row]
            if initial_step is not None and row <= initial_step_frames:
                nxt[speed_index] -= initial_step_weight * float(step - initial_step) ** 2
            back[row, speed_index] = shifted_speed
        current = nxt
    speed_index, state = np.unravel_index(int(np.argmax(current)), current.shape)
    path = np.zeros(n, np.int32)
    for row in range(n - 1, -1, -1):
        path[row] = state
        if row == 0:
            break
        previous_speed = int(back[row, speed_index, state])
        state -= int(steps[speed_index])
        speed_index = previous_speed
    return path


def monotone_viterbi(
    emissions: np.ndarray, max_step: int, *, accel_weight: float = 0.0002,
    decel_weight: float = 0.0015,
) -> np.ndarray:
    return dp_sweep(emissions, 0, max_step, accel_weight, 0.0, decel_weight)


def audit_positions(
    positions: Sequence[Observation], labels: Sequence[Any], observations: Sequence[Observation]
) -> AuditReport:
    label_constraints = constraints_from_labels(labels, 1.0)
    label_by_time = {round(item.t, 6): item for item in label_constraints}
    observed_times = {round(item.t, 6) for item in observations if item.source in {"observed", "detected"}}
    rows: list[AuditFrame] = []
    failures: list[str] = []
    for item in positions:
        label = label_by_time.get(round(item.t, 6))
        distance = None
        if label is not None:
            status = LABELLED
            distance = math.hypot(item.x - label.x, item.y - label.y)
        elif round(item.t, 6) in observed_times:
            status = OBSERVED
        elif item.source == "unconstrained":
            status = UNCONSTRAINED
            failures.append(f"frame {item.frame_index} is unconstrained")
        else:
            status = INTERPOLATED
        rows.append(AuditFrame(item.frame_index, item.t, status, distance))
    if not positions:
        failures.append("phase abstained: no per-frame positions")
    distances = [item.distance_to_label_px for item in rows if item.distance_to_label_px is not None]
    return AuditReport(
        passed=not failures,
        failures=failures,
        metrics={
            "frames": float(len(rows)),
            "labelled": float(sum(item.status == LABELLED for item in rows)),
            "observed": float(sum(item.status == OBSERVED for item in rows)),
            "interpolated": float(sum(item.status == INTERPOLATED for item in rows)),
            "unconstrained": float(sum(item.status == UNCONSTRAINED for item in rows)),
            "max_label_distance_px": float(max(distances, default=0.0)),
        },
        frames=rows,
    )
