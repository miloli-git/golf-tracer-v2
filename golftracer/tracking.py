"""Phase orchestration for ``track`` and the one-command ``reel`` path."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import time
from typing import Iterable, Sequence

import cv2
import numpy as np

from .config import Config
from .decode import decode_window
from .label.schema import Label, LabelDocument, load_labels
from .phases import BackswingPhase, BallPhase, DownswingPhase, FollowthroughPhase
from .phases.ball import constrained_observations, estimate_session_tees
from .phases.followthrough import detect_finish
LOG = logging.getLogger(__name__)

from .phases.base import (
    Constraint, GeometryOverlength, InsufficientSupport, SpatialArc,
    calibration_phase, dedupe_supports, reject_and_fit,
)
from .phases.backswing import (
    L_WRIST, R_WRIST, _fill_pose, _pose_series, club_observations,
    retime_spatial_arc,
)
from .session import AuditReport, Observation, Session, Swing, Track


PHASES = ("backswing", "downswing", "followthrough", "ball")
_TRACK_PHASES = {"club", "follow", "ball", "swing"}
_WORKER_START_TIMEOUT_S = 30.0


@dataclass
class ClubSpatialFit:
    backswing: object
    downswing: object
    backswing_records: list[dict]
    downswing_records: list[dict]
    accepted: list[Constraint]
    rejected: list[Constraint]
    label_smoothing_px: float


@dataclass(frozen=True)
class _WorkerFailure:
    reason: str
    stage: str


@dataclass
class _WorkerTask:
    operation: str
    session: Session
    config: Config
    swing: Swing | None = None
    selected: tuple[str, ...] = ()
    labels_root: Path | None = None
    debug_dir: Path | None = None
    tee_roi: tuple[int, int, int, int] | None = None
    tee_xy: tuple[float, float] | None = None
    detector_weights: Path | None = None
    impact_times: tuple[float, ...] = ()
    shot_priors: dict[float, tuple[float, float]] | None = None
    measurement_times: dict[float, float] | None = None


@dataclass(frozen=True)
class _WorkerOutcome:
    value: object | None
    failure: _WorkerFailure | None
    elapsed_s: float


def phase_biases_from_v1_samples(samples: Sequence[dict]) -> dict[str, tuple[float, float]]:
    """Exact v1 global clicked-minus-detected medians from merged label rows."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for sample in samples:
        tracker = sample.get("tracker", {})
        clicked = sample.get("clicked")
        if not sample.get("completed") or sample.get("skipped") or clicked is None or tracker.get("source") != "detected":
            continue
        if isinstance(clicked, dict):
            x, y = float(clicked["u"]), float(clicked["v"])
        else:
            x, y = float(clicked[0]), float(clicked[1])
        grouped.setdefault(str(sample["phase"]), []).append((x - float(tracker["u"]), y - float(tracker["v"])))
    result = {}
    for phase in ("takeaway", "mid-backswing", "top", "downswing", "delivery"):
        values = np.asarray(grouped.get(phase, []), dtype=float)
        result[phase] = tuple(np.median(values, axis=0)) if values.size else (0.0, 0.0)
    return {key: (float(value[0]), float(value[1])) for key, value in result.items()}


def _raw_value(item, key: str, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _raw_source(item) -> str:
    source = _raw_value(item, "source")
    if source:
        return str(source)
    return "interpolated" if _raw_value(item, "interpolated", False) else "detected"


def _constraint_for_label(label: Label, phase: str, impact_t: float, config: Config) -> Constraint:
    pseudo = impact_t - config.club_impact_label_lead_s if abs(label.t - impact_t) < 1e-7 else None
    return Constraint(
        label.frame_index, label.t, label.x, label.y, "mi" "lo_label",  # legacy v1 label source string
        config.club_label_weight, phase, pseudo_time=pseudo,
    )


def _supports_for_club(
    raw_points: Sequence,
    labels: Sequence[Label],
    *,
    takeaway_t: float,
    top_t: float,
    impact_t: float,
    biases: dict[str, tuple[float, float]],
    config: Config,
) -> tuple[list[Constraint], list[Constraint], list[Constraint]]:
    points = sorted(raw_points, key=lambda item: float(_raw_value(item, "t_s", _raw_value(item, "t"))))
    top_point = min(points, key=lambda item: abs(float(_raw_value(item, "t_s", _raw_value(item, "t"))) - top_t))
    top_bias = biases.get("top", (0.0, 0.0))
    join = Constraint(
        int(_raw_value(top_point, "rel_frame", _raw_value(top_point, "frame_index"))),
        top_t,
        float(_raw_value(top_point, "x")) + top_bias[0],
        float(_raw_value(top_point, "y")) + top_bias[1],
        "top_join", config.club_hard_weight, "top",
    )
    impact_points = [item for item in points if _raw_source(item) == "anchor" or str(_raw_value(item, "phase", "")) == "impact"]
    impact_labels = [item for item in labels if abs(item.t - impact_t) <= 1.5 / 60.0]
    if impact_points:
        anchor_point = impact_points[-1]
        anchor_xy = (float(_raw_value(anchor_point, "x")), float(_raw_value(anchor_point, "y")))
        anchor_frame = int(_raw_value(anchor_point, "rel_frame", _raw_value(anchor_point, "frame_index")))
    elif impact_labels:
        anchor_xy = (impact_labels[-1].x, impact_labels[-1].y)
        anchor_frame = impact_labels[-1].frame_index
    else:
        anchor_xy = (float(_raw_value(points[0], "x")), float(_raw_value(points[0], "y")))
        anchor_frame = int(round((impact_t - takeaway_t) * 60.0))
    anchor = Constraint(anchor_frame, impact_t, anchor_xy[0], anchor_xy[1], "impact_anchor", config.club_hard_weight, "delivery")

    back_hard: list[Constraint] = []
    down_hard: list[Constraint] = []
    for label in sorted(labels, key=lambda item: item.t):
        if label.t < takeaway_t - 1e-7 or label.t > impact_t + 1.5 / 60.0:
            continue
        detail = calibration_phase(label.t, takeaway_t, top_t, impact_t)
        support = _constraint_for_label(label, detail, impact_t, config)
        (back_hard if label.t <= top_t + 1e-7 else down_hard).append(support)
    back_hard = dedupe_supports([*back_hard, join])
    down_hard = dedupe_supports([join, *down_hard, anchor])

    labelled_times = [item.t for item in labels]
    candidates: list[Constraint] = []
    for item in points:
        source = _raw_source(item)
        t = float(_raw_value(item, "t_s", _raw_value(item, "t")))
        if source not in {"detected", "observed"} or any(abs(t - value) < 1e-7 for value in labelled_times) or abs(t - top_t) < 1e-7:
            continue
        detail = calibration_phase(t, takeaway_t, top_t, impact_t)
        if detail == "delivery":
            continue
        weight = {
            "takeaway": config.club_tracker_weight_takeaway,
            "mid-backswing": config.club_tracker_weight_mid_backswing,
            "top": config.club_tracker_weight_top,
            "downswing": config.club_tracker_weight_downswing,
        }[detail]
        bx, by = biases.get(detail, (0.0, 0.0))
        x, y = float(_raw_value(item, "x")), float(_raw_value(item, "y"))
        candidates.append(Constraint(
            int(_raw_value(item, "rel_frame", _raw_value(item, "frame_index"))),
            t, x + bx, y + by, "tracker_detected", weight, detail,
            original_x=x, original_y=y,
        ))
    return back_hard, down_hard, candidates


def fit_club_spatial_v1(
    raw_points: Sequence,
    labels: Sequence[Label],
    *,
    takeaway_t: float,
    top_t: float,
    impact_t: float,
    config: Config,
    biases: dict[str, tuple[float, float]] | None = None,
    label_smoothing_px: float | None = None,
    fps: float = 60.0,
) -> ClubSpatialFit:
    """Faithful v1 label-constrained, cusp-split refit for one swing."""
    biases = biases or {phase: (0.0, 0.0) for phase in ("takeaway", "mid-backswing", "top", "downswing", "delivery")}
    smoothing = config.club_label_smoothing_px if label_smoothing_px is None else label_smoothing_px
    back_hard, down_hard, candidates = _supports_for_club(
        raw_points, labels, takeaway_t=takeaway_t, top_t=top_t,
        impact_t=impact_t, biases=biases, config=config,
    )
    back_candidates = [item for item in candidates if item.t <= top_t + 1e-9]
    down_candidates = [item for item in candidates if item.t > top_t + 1e-9]
    back, back_kept, back_rejected, _ = reject_and_fit(back_hard, back_candidates, config, smoothing)
    down, down_kept, down_rejected, _ = reject_and_fit(down_hard, down_candidates, config, smoothing)
    back.phase, down.phase = "backswing", "downswing"
    back_records: list[dict] = []
    down_records: list[dict] = []
    count = int(round((impact_t - takeaway_t) * fps))
    for index in range(count + 1):
        t = impact_t if index == count else takeaway_t + index / fps
        curve = back if t <= top_t + 1e-9 else down
        x, y = curve.xy_at_time(t)
        record = {"t_s": t, "x": float(x), "y": float(y), "phase": "backswing" if t <= top_t + 1e-9 else ("impact" if abs(t-impact_t)<1e-7 else "downswing")}
        (back_records if record["phase"] == "backswing" else down_records).append(record)
    impact_labels = [item for item in labels if abs(item.t - impact_t) < 1e-7]
    if impact_labels and down_records:
        label = impact_labels[-1]
        down_records.insert(max(0, len(down_records) - 1), {
            "t_s": impact_t - config.club_impact_label_lead_s,
            "x": label.x, "y": label.y, "phase": "downswing",
        })
    # The retimer prepends the shared top record to the independent downswing arc.
    down_records = [back_records[-1], *down_records]
    return ClubSpatialFit(back, down, back_records, down_records, [*back_kept, *down_kept], [*back_rejected, *down_rejected], smoothing)


def retime_club_spatial_v1(
    fit: ClubSpatialFit,
    frames: np.ndarray,
    *,
    window_start: float,
    takeaway_t: float,
    top_t: float,
    impact_t: float,
    labels: Sequence[Label],
    fps: float,
    config: Config,
) -> tuple[list[Observation], dict]:
    back_geometry = SpatialArc.from_records(fit.backswing_records, config)
    down_geometry = SpatialArc.from_records(fit.downswing_records, config)
    first = int(round((takeaway_t - window_start) * fps))
    top = int(round((top_t - window_start) * fps))
    impact = int(round((impact_t - window_start) * fps))
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    quiet = gray[:max(3, min(first, 12))]
    background = np.median(quiet, axis=0).astype(np.uint8)
    pose = _fill_pose(_pose_series(frames[:impact + 1], config))
    if not pose or any(item is None for item in pose):
        raise RuntimeError("club retiming requires pose on the full canonical window")
    landmarks = np.asarray(pose, dtype=np.float32)
    wrists = 0.5 * (landmarks[:, L_WRIST] + landmarks[:, R_WRIST])
    back_labels = [item for item in labels if takeaway_t - 1e-7 <= item.t <= top_t + 1e-7]
    down_labels = [item for item in labels if top_t - 1e-7 <= item.t < impact_t - 1e-7]
    back_debug: dict = {}
    down_debug: dict = {}
    back = retime_spatial_arc(
        back_geometry, frames[first:top + 1], fps=fps,
        start_t=window_start + first / fps, labels=back_labels,
        config=config, background=background, debug=back_debug,
        wrists=wrists[first:top + 1],
    )
    down = retime_spatial_arc(
        down_geometry, frames[top:impact + 1], fps=fps,
        start_t=window_start + top / fps, labels=down_labels,
        config=config, background=background, debug=down_debug,
        wrists=wrists[top:impact + 1],
    )
    points = back + down[1:]
    for index, point in enumerate(points):
        point.frame_index = index
    return points, {
        "backswing_curve_xy": back_geometry.xy.tolist(),
        "downswing_curve_xy": down_geometry.xy.tolist(),
        "backswing_arc_knots": back_debug.get("arc", np.empty(0)).tolist(),
        "downswing_arc_knots": down_debug.get("arc", np.empty(0)).tolist(),
        "backswing_path": back_debug.get("path", np.empty(0, int)).tolist(),
        "downswing_path": down_debug.get("path", np.empty(0, int)).tolist(),
        "backswing_emissions": back_debug.get("emissions"),
        "downswing_emissions": down_debug.get("emissions"),
    }


def _label_candidates(root: Path, swing_id: int, phase: str) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for name in (
        f"{swing_id}.{phase}.json", f"swing-{swing_id:03d}.{phase}.json",
        f"{phase}.json",
    ):
        path = root / name
        if path.is_file():
            yield path


def labels_for(root: Path | None, swing_id: int, phase: str) -> list[Label]:
    if root is None:
        return []
    for path in _label_candidates(root, swing_id, phase):
        document = load_labels(path)
        return [item for item in document.labels if item.phase == phase]
    return []


def label_document_for(
    root: Path | None, swing_id: int, phase: str,
) -> LabelDocument | None:
    if root is None:
        return None
    for path in _label_candidates(root, swing_id, phase):
        return load_labels(path)
    return None


def _club_handoff_impact_t(
    raw_impact_t: float,
    down_labels: Sequence[Label],
    follow_document: LabelDocument | None,
    fps: float,
) -> float:
    """Keep a precise impact click when a rounded follow window names the same frame."""
    last_down = max((item.t for item in down_labels), default=None)
    if follow_document is not None:
        follow_start = float(follow_document.window_start)
        if last_down is not None and abs(last_down - follow_start) <= 0.5 / fps:
            return float(last_down)
        return follow_start
    if last_down is not None and abs(last_down - raw_impact_t) <= 0.05:
        return float(last_down)
    return float(raw_impact_t)


def _phase_window(swing: Swing, phase: str, config: Config) -> tuple[float, float]:
    if phase == "backswing":
        start = max(0.0, swing.impact_t - config.backswing_pre_s)
        return start, swing.impact_t - start
    if phase == "downswing":
        start = max(0.0, swing.impact_t - config.downswing_pre_s)
        return start, swing.impact_t - start + 1.0 / config.ball_fps
    start = max(0.0, swing.impact_t - config.tee_pre_s)
    return start, config.tee_pre_s + max(config.ball_post_s, config.ball_descent_post_s)


def _club_sanity_abstention(
    positions: Sequence[Observation], *, impact_t: float,
    frame_width: int, frame_height: int, config: Config,
) -> Track | None:
    """Reject a retimed club fit whose impact state is physically invalid."""
    if not positions:
        return None
    impact_xy = (float(positions[-1].x), float(positions[-1].y))
    steps = [
        float(np.hypot(second.x - first.x, second.y - first.y))
        for first, second in zip(positions[-5:-1], positions[-4:])
    ]
    impact_speed = float(np.median(steps)) if steps else 0.0
    margin = float(config.club_impact_frame_margin_px)
    in_frame = (
        -margin <= impact_xy[0] <= float(frame_width - 1) + margin
        and -margin <= impact_xy[1] <= float(frame_height - 1) + margin
    )
    if not in_frame:
        reason = "impact_out_of_frame"
    elif impact_speed < float(config.club_min_impact_speed_px_per_frame):
        reason = "impact_speed_too_low"
    else:
        return None

    measurements = {
        "impact_x_px": impact_xy[0],
        "impact_y_px": impact_xy[1],
        "impact_speed_px_per_frame": impact_speed,
        "frame_width_px": float(frame_width),
        "frame_height_px": float(frame_height),
        "impact_frame_margin_px": margin,
        "min_impact_speed_px_per_frame": float(
            config.club_min_impact_speed_px_per_frame
        ),
    }
    return Track(
        "club", [],
        AuditReport(False, [f"phase abstained: {reason}"], measurements),
        {
            "impact_t": impact_t,
            "impact_xy": impact_xy,
            "impact_speed_px_per_frame": impact_speed,
            "frame_width_px": frame_width,
            "frame_height_px": frame_height,
            "impact_frame_margin_px": margin,
            "min_impact_speed_px_per_frame": float(
                config.club_min_impact_speed_px_per_frame
            ),
        },
        True, reason,
    )


def _club_track(
    session: Session,
    swing: Swing,
    config: Config,
    phases: Sequence[str],
    labels_root: Path | None,
) -> Track | None:
    if not any(name in phases for name in ("backswing", "downswing", "followthrough")):
        return None
    back_labels = labels_for(labels_root, swing.id, "backswing")
    down_labels = labels_for(labels_root, swing.id, "downswing")
    follow_document = (
        label_document_for(labels_root, swing.id, "followthrough")
        if "followthrough" in phases else None
    )
    club_fps = 60.0
    impact_t = _club_handoff_impact_t(
        float(swing.impact_t), down_labels, follow_document, club_fps,
    )
    start = max(0.0, impact_t - config.backswing_pre_s)
    duration = impact_t - start + 1.0 / club_fps
    frames, _ = decode_window(session.video, start, duration, fps=club_fps)
    local = replace(swing, window_start=start, window_end=start + duration, impact_t=impact_t, metadata={})
    raw, _ = club_observations(frames, local, config, local.metadata)
    if not raw:
        return None
    # Label documents use independent phase-window frame bases. Rebase both onto
    # the one canonical v1 club window, then merge by count-indexed image identity;
    # the later downswing document wins collisions exactly like club_finalize.py.
    by_frame: dict[int, Label] = {}
    for label in [*back_labels, *down_labels]:
        frame = int(round((label.t - start) * club_fps))
        if frame < 0:
            continue
        by_frame[frame] = Label(
            frame, start + frame / club_fps, label.x, label.y,
            label.phase, label.source, label.convention,
        )
    labels = [by_frame[key] for key in sorted(by_frame)]
    takeaway_t = float(local.metadata.get("takeaway_t", raw[0].t))
    top_t = float(local.metadata.get("top_t", raw[-1].t))
    # A fixed-window v2 label document can contain address frames before the raw
    # phase finder. Apply v1's own 5%-of-torso takeaway rule to those hard clicks so
    # the first moving labelled image becomes the curve start instead of being lost.
    before_top = [item for item in labels if item.t <= top_t + 1e-7]
    if len(before_top) >= 2:
        baseline = np.median(np.asarray([(item.x, item.y) for item in before_top[:min(3, len(before_top))]]), axis=0)
        threshold = 0.05 * float(local.metadata.get("pose_scale_px", 0.0))
        moving = [
            item for item in before_top
            if item.t <= takeaway_t + 1e-7
            and np.linalg.norm(np.asarray((item.x, item.y)) - baseline) > threshold
        ]
        if moving:
            takeaway_t = moving[0].t
    anchor_label = labels[-1] if labels and abs(labels[-1].t - impact_t) <= 1.5 / club_fps else None
    raw_points: list = list(raw)
    if anchor_label is not None:
        raw_points.append({
            "rel_frame": int(round((impact_t - takeaway_t) * club_fps)),
            "t_s": impact_t, "x": anchor_label.x, "y": anchor_label.y,
            "phase": "impact", "source": "anchor",
        })
    if follow_document is not None:
        frame_zero = [item for item in follow_document.labels if item.frame_index == 0]
        if frame_zero:
            label = frame_zero[-1]
            raw_points.append({
                "rel_frame": int(round((impact_t - takeaway_t) * club_fps)),
                "t_s": impact_t, "x": label.x, "y": label.y,
                "phase": "impact", "source": "anchor",
            })
    try:
        fit = fit_club_spatial_v1(
            raw_points, labels, takeaway_t=takeaway_t, top_t=top_t,
            impact_t=impact_t, config=config, fps=club_fps,
        )
    except InsufficientSupport as exc:
        LOG.warning("swing %s: club track abstained (%s)", swing.id, exc)
        return None
    try:
        points, trusted = retime_club_spatial_v1(
            fit, frames, window_start=start, takeaway_t=takeaway_t, top_t=top_t,
            impact_t=impact_t, labels=labels, fps=club_fps, config=config,
        )
    except GeometryOverlength as exc:
        LOG.warning("swing %s: club track abstained (%s)", swing.id, exc)
        return Track(
            "club", [],
            AuditReport(False, [f"phase abstained: {exc.reason}"], {
                "geometry_length_px": exc.length_px,
                "geometry_limit_px": exc.limit_px,
            }),
            {
                "impact_t": impact_t,
                "geometry_length_px": exc.length_px,
                "geometry_limit_px": exc.limit_px,
                "geometry_ray_radius_px": exc.ray_radius_px,
            },
            True, exc.reason,
        )
    sanity_abstention = _club_sanity_abstention(
        points, impact_t=impact_t, frame_width=session.width,
        frame_height=session.height, config=config,
    )
    if sanity_abstention is not None:
        LOG.warning(
            "swing %s: club track abstained (%s)",
            swing.id, sanity_abstention.reason,
        )
        return sanity_abstention
    back_positions = [item for item in points if item.t <= top_t + 1e-7]
    down_positions = [item for item in points if item.t >= top_t - 1e-7]
    normalized_back = [item for item in labels if item.t <= top_t + 1e-7]
    normalized_down = [item for item in labels if item.t >= top_t - 1e-7]
    back_audit = BackswingPhase(config).audit(back_positions, normalized_back, raw)
    down_audit = DownswingPhase(config).audit(down_positions, normalized_down, raw)
    audit = AuditReport(
        back_audit.passed and down_audit.passed,
        [*[f"backswing: {item}" for item in back_audit.failures], *[f"downswing: {item}" for item in down_audit.failures]],
    )
    metadata = {
        "impact_t": impact_t,
        "takeaway_t": takeaway_t,
        "top_t": top_t,
        "impact_xy": (points[-1].x, points[-1].y),
        "trusted_geometry": {key: value for key, value in trusted.items() if not key.endswith("emissions") and not key.endswith("path")},
        "phase_audits": {
            "backswing": {"passed": back_audit.passed, "metrics": back_audit.metrics},
            "downswing": {"passed": down_audit.passed, "metrics": down_audit.metrics},
        },
        "n_tracker_retained": len(fit.accepted),
        "n_tracker_rejected": len(fit.rejected),
        "label_smoothing_px": fit.label_smoothing_px,
    }
    swing.window_start = max(0.0, takeaway_t - config.club_render_lead_s)
    return Track("club", points, audit, metadata)


def _follow_track(
    session: Session,
    swing: Swing,
    club: Track,
    config: Config,
    labels_root: Path | None,
    detector_weights: Path | None = None,
) -> Track:
    impact_t = float(club.metadata["impact_t"])
    impact_xy = tuple(float(value) for value in club.metadata["impact_xy"])
    document = label_document_for(labels_root, swing.id, "followthrough")
    labels = [] if document is None else [
        Label(
            item.frame_index, impact_t + item.frame_index / 60.0,
            item.x, item.y, "followthrough", item.source, item.convention,
        )
        for item in document.labels
    ]
    start = max(0.0, impact_t - config.backswing_pre_s)
    duration = impact_t - start + config.follow_through_post_s + 1.0 / 60.0
    frames, _ = decode_window(session.video, start, duration, fps=60.0)
    local = replace(
        swing, window_start=start, window_end=start + duration,
        impact_t=impact_t, metadata={},
    )
    club_steps = [
        float(np.hypot(second.x - first.x, second.y - first.y))
        for first, second in zip(club.observations[-5:-1], club.observations[-4:])
    ]
    impact_speed = float(np.median(club_steps)) if club_steps else None
    phase = FollowthroughPhase(
        config, impact_point=impact_xy,
        impact_speed_px_per_frame=impact_speed,
    )
    observations = phase.track(frames, local, config)
    impact_frame = int(round((impact_t - start) * 60.0))
    evidence_source = "raw_tracker"
    if detector_weights is not None:
        from .detect.clubhead import propose_frames
        proposals = propose_frames(
            frames[impact_frame:], weights=detector_weights, config=config,
        )
        observations = [
            Observation(
                index, impact_t + index / 60.0, item.x, item.y,
                item.confidence, "detector",
            )
            for index, item in sorted(proposals.items())
        ]
        finish = detect_finish(
            observations, frame_count=len(frames) - impact_frame,
            frame_shape=frames.shape[1:3], fps=60.0, config=config,
        )
        phase.finish_t = None if finish is None else finish.t
        phase.finish_reason = None if finish is None else finish.reason
        if finish is not None:
            observations = [
                item for item in observations if item.frame_index <= finish.frame_index
            ]
        evidence_source = "detector"
    spline = phase.fit(observations, labels)
    if spline is None:
        audit = AuditReport(False, ["phase abstained: insufficient follow-through support"])
        return Track("follow", [], audit, {
            "impact_t": impact_t, "impact_xy": impact_xy,
        }, True, "insufficient_support")
    try:
        positions = phase.retime(
            spline, frames[impact_frame:], fps=60.0, start_t=impact_t,
        )
    except GeometryOverlength as exc:
        LOG.warning("swing %s: follow track abstained (%s)", swing.id, exc)
        return Track(
            "follow", [],
            AuditReport(False, [f"phase abstained: {exc.reason}"], {
                "geometry_length_px": exc.length_px,
                "geometry_limit_px": exc.limit_px,
            }),
            {
                "impact_t": impact_t,
                "impact_xy": impact_xy,
                "geometry_length_px": exc.length_px,
                "geometry_limit_px": exc.limit_px,
                "geometry_ray_radius_px": exc.ray_radius_px,
                "evidence_source": evidence_source,
            },
            True, exc.reason,
        )
    accepted_observations = [
        Observation(
            item.frame_index, item.t,
            float(item.original_x if item.original_x is not None else item.x),
            float(item.original_y if item.original_y is not None else item.y),
            source=item.source,
        )
        for item in spline.accepted
    ]
    audit = phase.audit(positions, labels, accepted_observations)
    by_frame = {item.frame_index: item for item in observations}
    tracker_errors = []
    for label in labels:
        point = by_frame.get(label.frame_index)
        tracker_errors.append({
            "frame_index": label.frame_index,
            "t": label.t,
            "error_px": None if point is None else float(np.hypot(point.x - label.x, point.y - label.y)),
        })
    debug = phase.retime_debug
    geometry = debug.get("geometry")
    metadata = {
        "impact_t": impact_t,
        "impact_xy": impact_xy,
        "finish_t": positions[-1].t if positions else phase.finish_t,
        "finish_reason": phase.finish_reason,
        "tracker_label_errors": tracker_errors,
        "tracker_bias_xy_px": config.club_tracker_bias_followthrough,
        "impact_speed_px_per_frame": impact_speed,
        "n_tracker_retained": len(spline.accepted),
        "n_tracker_rejected": len(spline.rejected),
        "evidence_source": evidence_source,
        "trusted_geometry": {
            "follow_curve_xy": [] if geometry is None else geometry.xy.tolist(),
            "follow_arc_knots": debug.get("arc", np.empty(0)).tolist(),
        },
    }
    swing.metadata["finish_t"] = metadata["finish_t"]
    return Track("follow", positions, audit, metadata)


def _ball_track(
    session: Session, swing: Swing, config: Config, debug_dir: Path | None,
    tee_roi: tuple[int, int, int, int] | None = None,
    tee_xy: tuple[float, float] | None = None,
    labels_root: Path | None = None,
) -> Track:
    start, duration = _phase_window(swing, "ball", config)
    local = replace(swing, window_start=start, window_end=start + duration)
    phase = BallPhase(
        config, tee_roi=tee_roi, tee_xy=tee_xy, debug_dir=debug_dir,
    )
    observations = phase.track_video(session.video, local, config)
    labels = labels_for(labels_root, swing.id, "ball")
    raw_observations = observations
    label_fit_applied = False
    if labels:
        fit = phase.fit(raw_observations, labels)
        if fit is not None:
            observations = constrained_observations(raw_observations, labels, fit)
            phase.abstained = False
            phase.reason = None
            label_fit_applied = True
    audit = phase.audit(observations, labels, raw_observations)
    metadata = {"tee_xy": phase.tee_xy, **phase.metrics, "shaft_rule_fired": phase.shaft_rule_fired}
    if label_fit_applied:
        metadata.update({
            "label_constrained": True,
            "label_constraint_mode": "exact_residual_correction",
            "n_ball_labels": len({item.frame_index for item in labels}),
            "max_label_residual_px": audit.metrics.get("max_label_residual_px"),
            "rms_label_residual_px": audit.metrics.get("rms_label_residual_px"),
        })
    return Track("ball", observations, audit, metadata, phase.abstained, phase.reason)


def _track_club_stage(task: _WorkerTask) -> Swing:
    if task.swing is None:
        raise ValueError("club task requires a swing")
    swing = task.swing
    club = _club_track(
        task.session, swing, task.config, task.selected, task.labels_root,
    )
    if club is not None:
        swing.tracks.append(club)
        if "followthrough" in task.selected and not club.abstained:
            swing.tracks.append(_follow_track(
                task.session, swing, club, task.config, task.labels_root,
                task.detector_weights,
            ))
    return swing


def _track_ball_stage(task: _WorkerTask) -> Swing:
    if task.swing is None:
        raise ValueError("ball task requires a swing")
    swing = task.swing
    swing.tracks.append(_ball_track(
        task.session, swing, task.config, task.debug_dir, task.tee_roi,
        task.tee_xy, task.labels_root,
    ))
    return swing


def _estimate_tee_table(task: _WorkerTask) -> dict[float, tuple[float, float] | None]:
    return estimate_session_tees(
        task.session.video, task.impact_times, task.config, roi=task.tee_roi,
        prior_xy_by_impact=task.shot_priors,
        measurement_time_by_impact=task.measurement_times,
    )


def _swing_worker_loop(connection: Connection) -> None:
    """Run tracking tasks in an expendable Windows-spawn-compatible process."""
    connection.send(("ready", None))
    try:
        while True:
            try:
                task = connection.recv()
            except EOFError:
                break
            if task is None:
                break
            try:
                if task.operation == "club":
                    result = _track_club_stage(task)
                elif task.operation == "ball":
                    result = _track_ball_stage(task)
                elif task.operation == "tee_table":
                    result = _estimate_tee_table(task)
                else:
                    raise ValueError(f"unknown worker operation: {task.operation}")
                connection.send(("ok", result))
            except MemoryError:
                connection.send(("error", _WorkerFailure("memory_error", task.operation)))
            except Exception:
                LOG.exception("tracking worker failed during %s", task.operation)
                connection.send(("error", _WorkerFailure("unexpected_exception", task.operation)))
    finally:
        connection.close()


class _SwingWorker:
    """Supervise a persistent worker and replace it after any failed task."""

    def __init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._connection: Connection | None = None
        self._process: mp.Process | None = None

    def _stop(self, *, graceful: bool) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if process is None:
            if connection is not None:
                connection.close()
            return
        if graceful and connection is not None and process.is_alive():
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        process.join(timeout=5.0 if graceful else 0.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        if connection is not None:
            connection.close()

    def _start(self) -> bool:
        if self._process is not None and self._process.is_alive():
            return True
        self._stop(graceful=False)
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(target=_swing_worker_loop, args=(child,))
        try:
            process.start()
        except Exception:
            parent.close()
            child.close()
            LOG.exception("could not start tracking worker")
            return False
        child.close()
        self._connection = parent
        self._process = process
        try:
            if not parent.poll(_WORKER_START_TIMEOUT_S):
                LOG.error("tracking worker did not become ready")
                self._stop(graceful=False)
                return False
            status, _payload = parent.recv()
        except (EOFError, OSError):
            LOG.exception("tracking worker exited during startup")
            self._stop(graceful=False)
            return False
        if status != "ready":
            LOG.error("tracking worker returned an invalid startup response")
            self._stop(graceful=False)
            return False
        return True

    def run(self, task: _WorkerTask, timeout_s: float) -> _WorkerOutcome:
        if not self._start():
            return _WorkerOutcome(None, _WorkerFailure("worker_exit", task.operation), 0.0)
        assert self._connection is not None
        started = time.monotonic()
        try:
            self._connection.send(task)
            if not self._connection.poll(max(0.0, float(timeout_s))):
                elapsed = time.monotonic() - started
                self._stop(graceful=False)
                return _WorkerOutcome(None, _WorkerFailure("timeout", task.operation), elapsed)
            status, payload = self._connection.recv()
        except (BrokenPipeError, EOFError, OSError):
            elapsed = time.monotonic() - started
            self._stop(graceful=False)
            return _WorkerOutcome(None, _WorkerFailure("worker_exit", task.operation), elapsed)
        elapsed = time.monotonic() - started
        if status == "ok":
            return _WorkerOutcome(payload, None, elapsed)
        failure = payload if isinstance(payload, _WorkerFailure) else _WorkerFailure("worker_exit", task.operation)
        self._stop(graceful=False)
        return _WorkerOutcome(None, failure, elapsed)

    def close(self) -> None:
        self._stop(graceful=True)


def _session_shell(session: Session) -> Session:
    return replace(session, impacts=[], swings=[])


def _apply_worker_swing(target: Swing, source: Swing) -> None:
    target.window_start = source.window_start
    target.window_end = source.window_end
    target.impact_t = source.impact_t
    target.tracks = source.tracks
    target.metadata = source.metadata


def _record_swing_failure(swing: Swing, failure: _WorkerFailure) -> None:
    swing.metadata["tracking_failure_reason"] = failure.reason
    swing.metadata["tracking_failure_stage"] = failure.stage
    swing.tracks = [track for track in swing.tracks if track.phase != "swing"]
    swing.tracks.append(Track(
        "swing", [],
        AuditReport(
            False, [f"phase abstained: {failure.reason}"],
            {"abstained": 1.0},
        ),
        {"failure_stage": failure.stage},
        True, failure.reason,
    ))
    LOG.warning(
        "swing %s: tracking abstained during %s (%s)",
        swing.id, failure.stage, failure.reason,
    )


def track_session(
    session: Session,
    config: Config,
    *,
    phase: str = "all",
    labels_root: Path | None = None,
    debug_dir: Path | None = None,
    tee_roi: tuple[int, int, int, int] | None = None,
    detector_weights: Path | None = None,
) -> Session:
    selected = PHASES if phase == "all" else (phase,)
    if any(item not in PHASES for item in selected):
        raise ValueError(f"unknown phase: {phase}")
    for swing in session.swings:
        swing.tracks = [item for item in swing.tracks if item.phase not in _TRACK_PHASES]
        swing.metadata.pop("tracking_failure_reason", None)
        swing.metadata.pop("tracking_failure_stage", None)
    shell = _session_shell(session)
    selected_tuple = tuple(selected)
    timeout = float(config.track_swing_timeout_s)
    remaining = {swing.id: timeout for swing in session.swings}
    failed: set[int] = set()
    worker = _SwingWorker()
    try:
        if any(name in selected for name in ("backswing", "downswing", "followthrough")):
            for swing in session.swings:
                outcome = worker.run(_WorkerTask(
                    "club", shell, config, swing=swing, selected=selected_tuple,
                    labels_root=labels_root, detector_weights=detector_weights,
                ), remaining[swing.id])
                remaining[swing.id] = max(0.0, remaining[swing.id] - outcome.elapsed_s)
                if outcome.failure is not None:
                    _record_swing_failure(swing, outcome.failure)
                    failed.add(swing.id)
                else:
                    assert isinstance(outcome.value, Swing)
                    _apply_worker_swing(swing, outcome.value)

        tee_table: dict[float, tuple[float, float] | None] = {}
        if "ball" in selected:
            impact_times = tuple(float(swing.impact_t) for swing in session.swings if swing.id not in failed)
            shot_priors = {
                float(swing.impact_t): tuple(track.metadata["impact_xy"])
                for swing in session.swings
                if swing.id not in failed
                for track in swing.tracks
                if track.phase == "club" and not track.abstained
                and track.metadata.get("impact_xy") is not None
            }
            measurement_times = {
                float(swing.impact_t): float(track.metadata["impact_t"])
                for swing in session.swings
                if swing.id not in failed
                for track in swing.tracks
                if track.phase == "club" and not track.abstained
                and track.metadata.get("impact_t") is not None
            }
            if impact_times:
                tee_outcome = worker.run(_WorkerTask(
                    "tee_table", shell, config, tee_roi=tee_roi,
                    impact_times=impact_times, shot_priors=shot_priors,
                    measurement_times=measurement_times,
                ), timeout)
                if tee_outcome.failure is None:
                    assert isinstance(tee_outcome.value, dict)
                    tee_table = tee_outcome.value
                else:
                    LOG.warning(
                        "session tee calibration failed (%s); falling back to isolated per-swing measurement",
                        tee_outcome.failure.reason,
                    )

            for swing in session.swings:
                if swing.id in failed:
                    continue
                root = debug_dir / f"swing-{swing.id:03d}" if debug_dir is not None else None
                outcome = worker.run(_WorkerTask(
                    "ball", shell, config, swing=swing, debug_dir=root,
                    tee_roi=tee_roi, tee_xy=tee_table.get(float(swing.impact_t)),
                    labels_root=labels_root,
                ), remaining[swing.id])
                remaining[swing.id] = max(0.0, remaining[swing.id] - outcome.elapsed_s)
                if outcome.failure is not None:
                    _record_swing_failure(swing, outcome.failure)
                    failed.add(swing.id)
                else:
                    assert isinstance(outcome.value, Swing)
                    _apply_worker_swing(swing, outcome.value)
    finally:
        worker.close()
    return session
