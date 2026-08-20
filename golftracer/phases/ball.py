"""Calibrated ball-flight phase: top-hat tee, launch vote, gates and abstention."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..candidates import CandidateConfig, candidate_overlay, extract_candidate_observations
from ..config import Config
from ..decode import read_window_pts
from ..session import (
    AuditFrame, AuditReport, LABELLED, OBSERVED, Observation, Swing,
)
from ..stabilize import stabilize_frames
from .base import Phase


LOG = logging.getLogger("golftracer.phases.ball")


def derive_tee_roi(
    frames: np.ndarray,
    impact_index: int,
    config: Config | None = None,
) -> tuple[int, int, int, int]:
    """Derive a mat/feet ROI from impact-time foreground, in decoded geometry."""
    cfg = config or Config()
    height, width = frames.shape[1:3]
    pre = np.median(frames[max(0, impact_index - 15):impact_index], axis=0).astype(np.uint8)
    after = frames[min(len(frames) - 1, impact_index + 5)]
    diff = cv2.cvtColor(cv2.absdiff(pre, after), cv2.COLOR_BGR2GRAY)
    binary = ((diff > max(18.0, float(np.percentile(diff, 92)))) & (np.indices(diff.shape)[0] > 0.28 * height)).astype(np.uint8)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = [int(value) for value in stats[index, :4]]
        centre_x = x + w // 2
        feet_y = y + h
    else:
        centre_x, feet_y = width // 2, int(0.72 * height)
    v0 = max(0, feet_y - int(cfg.tee_roi_above_feet_height_ratio * height))
    v1 = min(height, feet_y + int(cfg.tee_roi_below_feet_height_ratio * height))
    u0 = max(0, centre_x - int(cfg.tee_roi_left_of_golfer_width_ratio * width))
    u1 = min(width, centre_x + int(cfg.tee_roi_right_of_golfer_width_ratio * width))
    return v0, v1, u0, u1


def estimate_tee_tophat_frames(
    frames: np.ndarray,
    timestamps: np.ndarray,
    impact_t: float,
    config: Config,
    *,
    roi: tuple[int, int, int, int] | None = None,
    prior_xy: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """White top-hat present-before/absent-after estimator from v1 tee_tophat.py."""
    pre = frames[timestamps < impact_t - 0.10]
    post = frames[timestamps > impact_t + 0.30]
    if len(pre) < 5 or len(post) < 5:
        return None
    impact_index = int(np.clip(np.searchsorted(timestamps, impact_t), 1, len(frames) - 1))
    v0, v1, u0, u1 = roi or config.tee_roi or derive_tee_roi(
        frames, impact_index, config,
    )
    v0, v1 = max(0, v0), min(frames.shape[1], v1)
    u0, u1 = max(0, u0), min(frames.shape[2], u1)
    if v1 <= v0 or u1 <= u0:
        return None
    gray_pre = [cv2.cvtColor(frame[v0:v1, u0:u1], cv2.COLOR_BGR2GRAY) for frame in pre]
    median_pre = np.median(gray_pre, axis=0).astype(np.uint8)
    median_post = np.median([cv2.cvtColor(frame[v0:v1, u0:u1], cv2.COLOR_BGR2GRAY) for frame in post], axis=0).astype(np.uint8)
    size = int(config.tee_tophat_kernel_px) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    top_pre = cv2.morphologyEx(cv2.GaussianBlur(median_pre, (0, 0), 2), cv2.MORPH_TOPHAT, kernel)
    top_post = cv2.morphologyEx(cv2.GaussianBlur(median_post, (0, 0), 2), cv2.MORPH_TOPHAT, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((top_pre > config.tee_tophat_min).astype(np.uint8), 8)
    best, best_score = None, -1e20
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        if not config.tee_area_px[0] <= area <= config.tee_area_px[1]:
            continue
        if not (config.tee_side_px[0] <= width <= config.tee_side_px[1] and config.tee_side_px[0] <= height <= config.tee_side_px[1]):
            continue
        if not (0.5 <= width / height <= 2.2) or area / (width * height) < 0.45:
            continue
        mask = labels == index
        if float(top_post[mask].mean()) > 0.4 * float(top_pre[mask].mean()):
            continue
        ix, iy = int(centroids[index][0]), int(centroids[index][1])
        if ix < 4 or iy < 4 or ix > median_pre.shape[1] - 5 or iy > median_pre.shape[0] - 5:
            continue
        samples = np.asarray([image[iy - 3:iy + 4, ix - 3:ix + 4].mean() for image in gray_pre])
        if float((np.abs(samples - np.median(samples)) <= config.tee_static_tolerance).mean()) < config.tee_static_fraction:
            continue
        score = float(top_pre[mask].mean()) / 50.0 - abs(1.0 - width / height)
        point = (float(centroids[index][0] + u0), float(centroids[index][1] + v0))
        if prior_xy is not None:
            score -= float(np.hypot(point[0] - prior_xy[0], point[1] - prior_xy[1])) / 150.0
        if score > best_score:
            best, best_score = point, score
    return best


def estimate_tee_v1_frames(
    frames: np.ndarray,
    timestamps: np.ndarray,
    impact_t: float,
    config: Config,
    *,
    roi: tuple[int, int, int, int] | None = None,
    prior_xy: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Bright-difference ball-at-address estimator from v1 ``retrack.py``."""
    pre = frames[timestamps < impact_t - 0.10]
    post = frames[timestamps > impact_t + 0.30]
    if len(frames) < 20 or len(pre) < 5 or len(post) < 5:
        return None
    median_pre = np.median(pre, axis=0).astype(np.uint8)
    median_post = np.median(post, axis=0).astype(np.uint8)
    v0, v1, u0, u1 = roi or config.tee_roi or config.tee_v1_roi
    p = median_pre[v0:v1, u0:u1]
    q = median_post[v0:v1, u0:u1]
    if not p.size or not q.size:
        return None
    light_pre = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.int16)
    light_post = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY).astype(np.int16)
    delta = np.abs(p.astype(np.int16) - q.astype(np.int16)).max(axis=2)
    gone = ((delta > 35) & (light_pre - light_post > 20)).astype(np.uint8)
    gone = cv2.morphologyEx(gone, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(gone, 8)
    height, width = light_pre.shape
    pre_gray = [cv2.cvtColor(frame[v0:v1, u0:u1], cv2.COLOR_BGR2GRAY) for frame in pre]
    post_gray = [cv2.cvtColor(frame[v0:v1, u0:u1], cv2.COLOR_BGR2GRAY) for frame in post]
    low, high = config.tee_v1_area_px
    best: tuple[float, float] | None = None
    best_score = -1e9
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if not (low <= area <= high) or not (
            8 <= component_width <= 30 and 8 <= component_height <= 30
        ):
            continue
        if not (0.55 <= component_width / component_height <= 1.8):
            continue
        if area / (component_width * component_height) < 0.55:
            continue
        local_u, local_v = float(centroids[index][0]), float(centroids[index][1])
        ix, iy = int(round(local_u)), int(round(local_v))
        if ix < 12 or iy < 12 or ix > width - 13 or iy > height - 13:
            continue
        mask = labels == index
        ring = np.zeros_like(mask)
        ring[max(0, iy - 14):iy + 15, max(0, ix - 14):ix + 15] = True
        ring &= ~cv2.dilate(
            mask.astype(np.uint8), np.ones((7, 7), np.uint8)
        ).astype(bool)
        if not ring.any():
            continue
        contrast = float(light_pre[mask].mean() - light_pre[ring].mean())
        if contrast < 25.0:
            continue
        before = np.asarray([
            image[iy - 3:iy + 4, ix - 3:ix + 4].mean()
            for image in pre_gray
        ])
        after = np.asarray([
            image[iy - 3:iy + 4, ix - 3:ix + 4].mean()
            for image in post_gray
        ])
        if float(before.std()) > 18.0:
            continue
        drop = float(before.mean() - after.mean())
        if drop < 18.0:
            continue
        score = (
            contrast / 100.0 + drop / 100.0
            - abs(area - config.tee_v1_expected_area_px) / 120.0
            - abs(1.0 - component_width / component_height)
        )
        point = (local_u + u0, local_v + v0)
        if prior_xy is not None:
            score -= float(np.hypot(
                point[0] - prior_xy[0], point[1] - prior_xy[1]
            )) / 150.0
        if score > best_score:
            best_score, best = score, point
    return best


def estimate_tee_frames(
    frames: np.ndarray,
    timestamps: np.ndarray,
    impact_t: float,
    config: Config,
    *,
    roi: tuple[int, int, int, int] | None = None,
    prior_xy: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    if config.tee_method == "v1":
        return estimate_tee_v1_frames(
            frames, timestamps, impact_t, config, roi=roi, prior_xy=prior_xy,
        )
    if config.tee_method == "tophat":
        return estimate_tee_tophat_frames(
            frames, timestamps, impact_t, config, roi=roi, prior_xy=prior_xy,
        )
    raise ValueError(f"unknown tee_method: {config.tee_method}")


def estimate_session_tees(
    video: str, impact_times: Sequence[float], config: Config,
    *, roi: tuple[int, int, int, int] | None = None,
    prior_xy_by_impact: Mapping[float, tuple[float, float]] | None = None,
    measurement_time_by_impact: Mapping[float, float] | None = None,
    return_provenance: bool = False,
) -> Any:
    """V1 two-pass session table: measure, prior remeasure, neighbour fill."""
    windows: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    derived_candidates: list[tuple[int, int, int, int]] = []
    resolved_roi = roi or config.tee_roi
    for raw_impact in impact_times:
        impact = float(raw_impact)
        measurement_time = (
            float(measurement_time_by_impact.get(impact, impact))
            if measurement_time_by_impact is not None else impact
        )
        start = max(0.0, measurement_time - config.tee_pre_s)
        frames, timestamps = read_window_pts(
            video, start, config.tee_pre_s + config.tee_post_s,
            fps=30.0, gray=False,
        )
        windows[impact] = (frames, timestamps)
        if resolved_roi is None and config.tee_method == "tophat" and len(frames) >= 2:
            index = int(np.clip(np.searchsorted(timestamps, measurement_time), 1, len(frames) - 1))
            derived_candidates.append(derive_tee_roi(frames, index, config))
    if resolved_roi is None:
        if config.tee_method == "v1":
            resolved_roi = config.tee_v1_roi
        elif derived_candidates:
            resolved_roi = tuple(int(round(value)) for value in np.median(
                np.asarray(derived_candidates), axis=0
            ))
    if resolved_roi is None:
        raise RuntimeError("could not derive a tee ROI")
    table: dict[float, tuple[float, float] | None] = {}
    provenance: dict[float, dict[str, Any]] = {}
    for raw_impact in impact_times:
        impact = float(raw_impact)
        measurement_time = (
            float(measurement_time_by_impact.get(impact, impact))
            if measurement_time_by_impact is not None else impact
        )
        frames, timestamps = windows[impact]
        shot_prior = (
            prior_xy_by_impact.get(impact)
            if prior_xy_by_impact is not None else None
        )
        table[impact] = estimate_tee_frames(
            frames, timestamps, measurement_time, config, roi=resolved_roi,
            prior_xy=shot_prior,
        )
        provenance[impact] = {
            "method": config.tee_method,
            "source": (
                "measured_with_shot_prior"
                if table[impact] is not None and shot_prior is not None
                else "measured" if table[impact] is not None else "missing"
            ),
            "roi": list(resolved_roi),
            "measurement_time_s": measurement_time,
        }
        if shot_prior is not None:
            provenance[impact]["shot_prior_xy"] = list(shot_prior)
    found = [value for value in table.values() if value is not None]
    prior = tuple(np.median(np.asarray(found), axis=0)) if found else None
    if prior is not None:
        for impact, value in list(table.items()):
            if value is None:
                continue
            if float(np.hypot(value[0] - prior[0], value[1] - prior[1])) > 220.0:
                frames, timestamps = windows[impact]
                measurement_time = (
                    float(measurement_time_by_impact.get(impact, impact))
                    if measurement_time_by_impact is not None else impact
                )
                measured = estimate_tee_frames(
                    frames, timestamps, measurement_time, config,
                    roi=resolved_roi, prior_xy=prior,
                )
                if measured is not None:
                    table[impact] = measured
                    provenance[impact]["source"] = "remeasured_with_prior"
                    provenance[impact]["prior_xy"] = [float(prior[0]), float(prior[1])]
    known = [key for key in sorted(table) if table[key] is not None]
    for impact in sorted(table):
        if table[impact] is not None or not known:
            continue
        nearest = sorted(known, key=lambda key: abs(key - impact))[:2]
        points = np.asarray([table[key] for key in nearest], float)
        table[impact] = (float(points[:, 0].mean()), float(points[:, 1].mean()))
        provenance[impact]["source"] = "neighbour_fill"
        provenance[impact]["neighbours"] = nearest
    if return_provenance:
        return table, provenance
    return table


def track_metrics(points: Sequence[Mapping[str, float]], speed_factor: float = 1.7, local_factor: float = 2.0) -> dict[str, Any]:
    frame = np.asarray([point["rel_frame"] for point in points], float)
    u = np.asarray([point["u"] for point in points], float)
    v = np.asarray([point["v"] for point in points], float)
    order = np.argsort(frame)
    frame, u, v = frame[order], u[order], v[order]
    step = np.hypot(np.diff(u), np.diff(v)) / np.maximum(np.diff(frame), 1.0)
    third = max(1, len(step) // 3)
    launch_step = float(np.median(step[:min(3, len(step))])) if len(step) else 0.0
    local_violations = 0
    for index in range(1, len(step)):
        recent = float(np.median(step[max(0, index - 3):index]))
        local_violations += int(step[index] > max(5.0, local_factor * recent))
    count = min(6, len(frame))
    if count >= 3:
        direction = np.asarray((np.polyfit(frame[:count], u[:count], 1)[0], np.polyfit(frame[:count], v[:count], 1)[0]))
    else:
        direction = np.asarray((u[1] - u[0], v[1] - v[0]))
    direction /= max(1e-6, float(np.linalg.norm(direction)))
    area = np.asarray([point.get("area", 0.0) for point in points], float)[order]
    positive_area = area[area > 0]
    correlation = float(np.corrcoef(np.arange(len(step)), np.log(step + 0.5))[0, 1]) if len(step) >= 3 else 0.0
    return {
        "f": frame, "u": u, "v": v, "step": step,
        "rise": float(v[0] - v[-1]), "lateral": float(np.ptp(u)),
        "early": float(step[:third].mean()) if len(step) else 0.0,
        "late": float(step[-third:].mean()) if len(step) else 0.0,
        "direction": direction, "origin": np.asarray((u[0], v[0])),
        "launch_step": launch_step,
        "speed_violation": float((step > speed_factor * launch_step).mean()) if len(step) and launch_step > 0 else 1.0,
        "speed_decay_correlation": correlation,
        "local_speed_violations": local_violations,
        "median_area": float(np.median(positive_area)) if len(positive_area) else 0.0,
        "max_area": float(area.max()) if len(area) else 0.0,
    }


def gate_track(points: Sequence[Mapping[str, float]], tee: Sequence[float], config: Config) -> tuple[bool, str | None, dict[str, Any] | None]:
    if len(points) < config.ball_min_inliers:
        return False, "too_few_points", None
    metric = track_metrics(points, config.ball_speed_violation_factor, config.ball_local_speed_violation_factor)
    checks = [
        (metric["f"][0] > config.ball_max_launch_delay_frames, "launch_too_late"),
        (metric["rise"] <= 0, "net_descends"),
        (metric["rise"] < config.ball_min_rise_px, "rise_too_small"),
        (metric["lateral"] > config.ball_max_lateral_ratio * metric["rise"], "lateral_spread"),
        (len(metric["step"]) < 3 or float(np.median(metric["step"])) < config.ball_min_median_step_px, "stationary"),
        (int((metric["step"] >= config.ball_moving_step_px).sum()) < config.ball_min_moving_steps, "too_few_moving_steps"),
        (metric["early"] <= metric["late"], "no_speed_decay"),
        (not np.isfinite(metric["speed_decay_correlation"]) or metric["speed_decay_correlation"] > config.ball_min_speed_decay_correlation, "incoherent_speed_decay"),
        (metric["local_speed_violations"] > config.ball_max_local_speed_violations, "erratic_local_speed"),
        (metric["launch_step"] < config.ball_min_launch_step_px, "launch_too_slow"),
        (metric["speed_violation"] > config.ball_max_speed_violation_frac, "erratic_speed"),
        (0 < config.ball_max_median_area_px < metric["median_area"], "blob_too_large"),
        (0 < config.ball_max_blob_area_px < metric["max_area"], "blob_size_inconsistent"),
    ]
    for failed, reason in checks:
        if failed:
            return False, reason, metric
    offset = np.asarray(tee, float) - metric["origin"]
    if float(offset @ metric["direction"]) > 0:
        return False, "tee_ahead_of_start", metric
    gap = float(abs(offset[0] * metric["direction"][1] - offset[1] * metric["direction"][0]))
    metric["start_gap"] = gap
    if gap > config.ball_origin_tolerance_px:
        return False, "launch_ray_misses_tee", metric
    return True, None, metric


def _drop_static_repeats(
    points: list[dict[str, float]], config: Config,
) -> tuple[list[dict[str, float]], int]:
    """Drop repeats of one coordinate beyond the allowed count.

    A ball in flight never sits at the same pixel for three or more frames; a
    coordinate accepted that often is static clutter re-matched across frames,
    and the render spline visibly wobbles through the dwell. The default allows
    a coordinate twice so genuine near-apex slow frames (and the one duplicated
    pair in the calibrated golden set) are untouched.
    """
    limit = config.ball_max_coord_repeats
    if limit <= 0:
        return points, 0
    counts: dict[tuple[float, float], int] = {}
    kept: list[dict[str, float]] = []
    dropped = 0
    for point in points:
        key = (round(float(point["u"]), 1), round(float(point["v"]), 1))
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > limit:
            dropped += 1
            continue
        kept.append(point)
    return kept, dropped


def _longest_rising(frame: np.ndarray, vertical: np.ndarray, max_gap: int) -> np.ndarray:
    length = np.ones(len(frame), int)
    parent = np.full(len(frame), -1, int)
    for right in range(1, len(frame)):
        for left in range(right):
            if frame[right] == frame[left] or frame[right] - frame[left] > max_gap:
                continue
            if vertical[right] < vertical[left] and length[left] + 1 > length[right]:
                length[right], parent[right] = length[left] + 1, left
    current = int(np.argmax(length))
    chain = []
    while current >= 0:
        chain.append(current)
        current = int(parent[current])
    return np.asarray(chain[::-1], int)


def launch_vote(
    observations: np.ndarray,
    tee: Sequence[float],
    impact_t: float,
    config: Config,
    *,
    debug: dict[str, Any] | None = None,
) -> tuple[list[dict[str, float]] | None, str | None, dict[str, Any] | None]:
    if observations is None or len(observations) < config.ball_min_inliers:
        return None, "too_few_candidates", None
    values = np.asarray(observations, float)
    if values.shape[1] < 4:
        values = np.column_stack((values, np.zeros(len(values))))
    tee = np.asarray(tee, float)
    above = (values[:, 2] < tee[1] - config.ball_min_above_tee_px) & (values[:, 0] >= 1)
    if debug is not None:
        debug["above"] = above.tolist()
    values = values[above]
    if len(values) < config.ball_min_inliers:
        return None, "too_few_above_tee", None
    theta = np.degrees(np.arctan2(values[:, 1] - tee[0], tee[1] - values[:, 2]))
    physical = np.abs(theta) <= config.ball_max_launch_angle_deg
    values, theta = values[physical], theta[physical]
    if debug is not None:
        debug["physical_angles_deg"] = theta.tolist()
        debug["physical_observations"] = values.tolist()
        debug["windows"] = []
    if len(values) < config.ball_min_inliers:
        return None, "too_few_physical_angles", None
    best = None
    centres = np.arange(
        -config.ball_max_launch_angle_deg,
        config.ball_max_launch_angle_deg + 0.5 * config.ball_centre_step_deg,
        config.ball_centre_step_deg,
    )
    seen = set()
    last_reason = "no_angle_peak"
    for half_width in config.ball_bin_widths_deg:
        for centre in centres:
            selected = np.abs(theta - centre) <= half_width
            if selected.sum() < config.ball_min_inliers:
                continue
            key = (round(half_width, 2), tuple(np.flatnonzero(selected)))
            if key in seen:
                continue
            seen.add(key)
            candidates, angles = values[selected], theta[selected]
            chosen: dict[int, int] = {}
            for index, row in enumerate(candidates):
                frame = int(row[0])
                if frame not in chosen or abs(angles[index] - centre) < abs(angles[chosen[frame]] - centre):
                    chosen[frame] = index
            chosen_indices = np.asarray([chosen[key] for key in sorted(chosen)], int)
            candidates = candidates[chosen_indices]
            if len(candidates) < config.ball_min_inliers:
                continue
            rising = _longest_rising(
                candidates[:, 0], candidates[:, 2], config.ball_max_frame_gap,
            )
            if len(rising) < config.ball_min_inliers:
                continue
            candidates = candidates[rising]
            points = [
                {"rel_frame": int(row[0]), "u": float(row[1]), "v": float(row[2]),
                 "t_s": impact_t + float(row[0]) / config.ball_fps, "area": float(row[3])}
                for row in candidates
            ]
            ok, reason, metric = gate_track(points, tee, config)
            if debug is not None:
                debug["windows"].append({
                    "centre_deg": float(centre),
                    "half_width_deg": float(half_width),
                    "selected_indices": np.flatnonzero(selected).tolist(),
                    "chosen_frames": [int(value) for value in candidates[:, 0]],
                    "rising_indices": rising.tolist(),
                    "gate_ok": bool(ok),
                    "gate_reason": reason,
                })
            if not ok:
                last_reason = reason or last_reason
                continue
            score = len(points) / 3.0 + metric["rise"] / 60.0 - metric["start_gap"] / 150.0 - 4.0 * metric["lateral"] / max(metric["rise"], 1.0)
            if best is None or score > best[0]:
                best = (score, points, metric, float(centre), float(half_width))
    if best is None:
        return None, last_reason, None
    if debug is not None:
        debug["winner"] = {
            "score": float(best[0]),
            "centre_deg": best[3],
            "half_width_deg": best[4],
            "chosen_frames": [int(point["rel_frame"]) for point in best[1]],
        }
    return best[1], None, best[2]


def _descent_extension(
    ascent: Sequence[dict[str, float]], observations: np.ndarray, config: Config,
) -> tuple[list[dict[str, float]], dict[str, Any]] | None:
    """Continuity DP through apex/down; descent never rescues a failed ascent."""
    if observations is None or not ascent:
        return None
    values = np.asarray(observations, float)
    if values.ndim != 2 or not len(values):
        return None
    if values.shape[1] < 4:
        values = np.column_stack((values, np.zeros(len(values))))
    start = ascent[-1]
    sf, su, sv = int(start["rel_frame"]), float(start["u"]), float(start["v"])
    areas = np.asarray([point.get("area", 0.0) for point in ascent[-10:]], float)
    areas = areas[areas > 0]
    if not len(areas):
        return None
    tail_area = float(np.median(areas))
    maximum_area = max(24.0, config.ball_descent_area_factor * tail_area)
    candidates = values[
        (values[:, 0] > sf)
        & (values[:, 2] >= sv - config.ball_descent_apex_slack_px)
        & (values[:, 2] <= sv + config.ball_descent_max_drop_search_px)
        & (values[:, 3] > 0) & (values[:, 3] <= maximum_area)
    ]
    if len(candidates) < config.ball_descent_min_points:
        return None
    downward = np.maximum(candidates[:, 2] - sv, 0.0)
    corridor = config.ball_descent_lateral_slack_px + config.ball_descent_max_lateral_ratio * downward
    candidates = candidates[np.abs(candidates[:, 1] - su) <= corridor]
    if len(candidates) < config.ball_descent_min_points:
        return None
    pruned = []
    for frame in np.unique(candidates[:, 0]):
        rows = candidates[candidates[:, 0] == frame]
        if len(rows) > config.ball_descent_candidates_per_frame:
            proximity = np.hypot(rows[:, 1] - su, rows[:, 2] - sv)
            rows = rows[np.argsort(proximity)[:config.ball_descent_candidates_per_frame]]
        pruned.append(rows)
    candidates = np.vstack(pruned)
    candidates = candidates[np.lexsort((candidates[:, 2], candidates[:, 0]))]
    score = np.full(len(candidates), -np.inf)
    parent = np.full(len(candidates), -1, int)
    frames = sorted(map(int, np.unique(candidates[:, 0])))
    by_frame = {frame: np.flatnonzero(candidates[:, 0] == frame) for frame in frames}
    for frame in frames:
        indices = by_frame[frame]
        current = candidates[indices]
        dt = frame - sf
        du, dv = current[:, 1] - su, current[:, 2] - sv
        distance = np.hypot(du, dv)
        valid = (
            (dt <= config.ball_descent_max_gap)
            & (dv >= -1.5 * dt)
            & (dv <= config.ball_descent_max_vertical_step_px * dt)
            & (np.abs(du) <= config.ball_descent_max_lateral_step_px * dt)
            & (distance <= config.ball_descent_max_step_px * dt)
        )
        score[indices[valid]] = 1.0 + 0.18 * np.maximum(dv[valid], 0.0) - 0.03 * np.abs(du[valid])
        for previous_frame in range(max(sf + 1, frame - config.ball_descent_max_gap), frame):
            previous_indices = by_frame.get(previous_frame)
            if previous_indices is None:
                continue
            previous_indices = previous_indices[np.isfinite(score[previous_indices])]
            if not len(previous_indices):
                continue
            previous = candidates[previous_indices]
            gap = frame - previous_frame
            edge_u = current[:, None, 1] - previous[None, :, 1]
            edge_v = current[:, None, 2] - previous[None, :, 2]
            edge_distance = np.hypot(edge_u, edge_v)
            allowed = (
                (edge_v >= -1.5 * gap)
                & (edge_v <= config.ball_descent_max_vertical_step_px * gap)
                & (np.abs(edge_u) <= config.ball_descent_max_lateral_step_px * gap)
                & (edge_distance <= config.ball_descent_max_step_px * gap)
            )
            area_penalty = 0.03 * np.abs(np.log((current[:, None, 3] + 1.0) / (previous[None, :, 3] + 1.0)))
            proposed = score[previous_indices][None, :] + 1.0 + 0.18 * np.maximum(edge_v, 0.0) - 0.03 * np.abs(edge_u) - area_penalty - 0.08 * (gap - 1)
            proposed[~allowed] = -np.inf
            best_index = np.argmax(proposed, axis=1)
            best_score = proposed[np.arange(len(current)), best_index]
            take = best_score > score[indices]
            score[indices[take]] = best_score[take]
            parent[indices[take]] = previous_indices[best_index[take]]
    for endpoint in np.argsort(score)[::-1]:
        if not np.isfinite(score[endpoint]):
            break
        chain = []
        current = int(endpoint)
        while current >= 0:
            chain.append(current); current = int(parent[current])
        path = candidates[np.asarray(chain[::-1], int)]
        if len(path) < config.ball_descent_min_points:
            continue
        all_v = np.r_[sv, path[:, 2]]
        apex_index = int(np.argmin(all_v)) - 1
        apex_frame = sf if apex_index < 0 else int(path[apex_index, 0])
        apex_u = su if apex_index < 0 else float(path[apex_index, 1])
        apex_v = sv if apex_index < 0 else float(path[apex_index, 2])
        before = path[:apex_index + 1] if apex_index >= 0 else path[:0]
        descending = []
        last_v = apex_v
        for row in path[apex_index + 1:]:
            if row[2] + 0.05 < last_v:
                continue
            descending.append(row); last_v = float(row[2])
        if len(descending) < config.ball_descent_min_points:
            continue
        descent = np.asarray(descending, float)
        drop = float(descent[-1, 2] - apex_v)
        if drop < config.ball_descent_min_drop_px:
            continue
        df = np.diff(np.r_[apex_frame, descent[:, 0]])
        if (df <= 0).any() or float(df.max()) > config.ball_descent_max_gap:
            continue
        du = np.diff(np.r_[apex_u, descent[:, 1]])
        dv = np.diff(np.r_[apex_v, descent[:, 2]])
        steps = np.hypot(du, dv) / df
        third = max(1, len(steps) // 3)
        if float(steps[-third:].mean()) <= float(steps[:third].mean()):
            continue
        lateral = float(np.ptp(np.r_[apex_u, descent[:, 1]]))
        if lateral > config.ball_descent_lateral_slack_px + config.ball_descent_max_lateral_ratio * drop:
            continue
        area_ratio = float(np.median(descent[:, 3])) / max(tail_area, 1.0)
        if not config.ball_descent_min_area_ratio <= area_ratio <= config.ball_descent_area_factor:
            continue
        extension = np.vstack((before, descent)) if len(before) else descent
        points = [{
            "rel_frame": int(row[0]), "u": float(row[1]), "v": float(row[2]),
            "t_s": float(ascent[0]["t_s"] + (row[0] - ascent[0]["rel_frame"]) / config.ball_fps),
            "area": float(row[3]),
        } for row in extension]
        return points, {
            "apex_frame": apex_frame, "apex_xy": [apex_u, apex_v],
            "n_descent": len(descent), "descent_drop_px": drop,
            "descent_lateral_px": lateral, "descent_area_ratio": area_ratio,
        }
    return None


def candidate_observations_from_window(
    frames: np.ndarray,
    timestamps: np.ndarray,
    impact_t: float,
    config: Config,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Run v1 stabilization/candidates and return count-based observations."""
    if len(frames) < 8:
        return None, {
            "decoded_shape": list(frames.shape),
            "pts_s": timestamps.tolist(),
            "reason": "window_too_short",
        }
    gray = (
        frames
        if frames.ndim == 3
        else np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    )
    impact_index = int(np.clip(
        np.searchsorted(timestamps, impact_t, side="left"), 1, len(gray) - 1,
    ))
    registered, registration = stabilize_frames(gray, range(impact_index))
    records, projection, background = extract_candidate_observations(
        registered,
        impact_index,
        config.ball_fps,
        start_time_s=impact_t - config.ball_pre_s,
        timestamps_s=timestamps,
        config=CandidateConfig(),
    )
    values = []
    raw = []
    for record in records:
        frame_index = int(record["frame_index"])
        if frame_index < impact_index:
            continue
        for candidate in record.get("observations", record.get("candidates", [])):
            values.append((
                frame_index - impact_index,
                float(candidate["u"]),
                float(candidate["v"]),
                float(candidate.get("area", 0.0)),
            ))
            raw.append({
                "frame": frame_index - impact_index,
                "u": float(candidate["u"]),
                "v": float(candidate["v"]),
                "area": float(candidate.get("area", 0.0)),
                "polarity": candidate.get("polarity"),
            })
    array = np.asarray(values, float) if values else None
    return array, {
        "decoded_shape": list(frames.shape),
        "pts_s": timestamps.tolist(),
        "impact_index": impact_index,
        "registration": [item.to_dict() for item in registration],
        "raw_candidates": raw,
        "records": records,
        "projection": projection,
        "background": background,
    }


@dataclass(frozen=True)
class BallFit:
    t_origin: float
    t_scale: float
    u: tuple[float, ...]
    v: tuple[float, ...]
    inliers: tuple[bool, ...]
    rmse_px: float
    correction_times: tuple[float, ...] = ()
    correction_u: tuple[float, ...] = ()
    correction_v: tuple[float, ...] = ()

    def predict(self, times: float | Sequence[float]) -> np.ndarray:
        values = np.atleast_1d(np.asarray(times, float))
        x = (values - self.t_origin) / self.t_scale
        result = np.column_stack((np.polynomial.polynomial.polyval(x, self.u), np.polynomial.polynomial.polyval(x, self.v)))
        if self.correction_times:
            result[:, 0] += np.interp(
                values, self.correction_times, self.correction_u,
            )
            result[:, 1] += np.interp(
                values, self.correction_times, self.correction_v,
            )
        return result[0] if np.ndim(times) == 0 else result


def robust_fit_2d(observations: Sequence[Observation], degree: int = 2) -> BallFit:
    if len(observations) < 3:
        raise ValueError("at least three observations are required")
    ordered = sorted(observations, key=lambda item: item.t)
    times = np.asarray([item.t for item in ordered])
    points = np.asarray([(item.x, item.y) for item in ordered])
    origin, scale = float(times[0]), max(float(times[-1] - times[0]), 1 / 60)
    x = (times - origin) / scale
    degree = min(degree, len(ordered) - 1)
    initial = np.r_[np.polynomial.polynomial.polyfit(x, points[:, 0], degree), np.polynomial.polynomial.polyfit(x, points[:, 1], degree)]
    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted = np.column_stack((np.polynomial.polynomial.polyval(x, parameters[:degree + 1]), np.polynomial.polynomial.polyval(x, parameters[degree + 1:])))
        return (predicted - points).ravel()
    solved = least_squares(residual, initial, loss="huber", f_scale=1.5, max_nfev=400)
    predicted = np.column_stack((np.polynomial.polynomial.polyval(x, solved.x[:degree + 1]), np.polynomial.polynomial.polyval(x, solved.x[degree + 1:])))
    errors = np.linalg.norm(predicted - points, axis=1)
    centre = float(np.median(errors)); sigma = 1.4826 * float(np.median(np.abs(errors - centre)))
    inliers = errors <= max(4.0, centre + 3.0 * max(1.0, sigma))
    return BallFit(origin, scale, tuple(solved.x[:degree + 1]), tuple(solved.x[degree + 1:]), tuple(bool(value) for value in inliers), float(np.sqrt(np.mean(errors[inliers] ** 2))))


def label_constrained_fit(
    observations: Sequence[Observation], labels: Sequence[Any], degree: int = 2,
) -> BallFit | None:
    """Fit observed flight softly, then apply exact human correction residuals."""
    by_frame = {int(item.frame_index): item for item in labels}
    trusted = [by_frame[key] for key in sorted(by_frame)]
    supports: list[Observation] = list(observations)
    if len(supports) < 3:
        supports = [
            Observation(
                int(item.frame_index), float(item.t), float(item.x), float(item.y),
                source=str(getattr(item, "source", "human")),
            )
            for item in trusted
        ]
    if len(supports) < 3:
        return None
    base = robust_fit_2d(supports, degree=degree)
    if not trusted:
        return base

    label_times = np.asarray([float(item.t) for item in trusted], dtype=float)
    label_xy = np.asarray([(float(item.x), float(item.y)) for item in trusted])
    offsets = label_xy - base.predict(label_times)
    corrections: dict[float, tuple[float, float]] = {
        float(timestamp): (float(offset[0]), float(offset[1]))
        for timestamp, offset in zip(label_times, offsets, strict=True)
    }
    ordered_observations = sorted(observations, key=lambda item: item.t)
    before = [item for item in ordered_observations if item.t < label_times[0]]
    after = [item for item in ordered_observations if item.t > label_times[-1]]
    # A neighbouring zero-residual knot tapers a local correction into the
    # autonomous fit. With no evidence on one side, the nearest human residual
    # remains in force instead of snapping the flight back to a suspect track.
    if before:
        corrections[float(before[-1].t)] = (0.0, 0.0)
    if after:
        corrections[float(after[0].t)] = (0.0, 0.0)
    correction_times = tuple(sorted(corrections))
    return BallFit(
        base.t_origin, base.t_scale, base.u, base.v, base.inliers, base.rmse_px,
        correction_times,
        tuple(corrections[item][0] for item in correction_times),
        tuple(corrections[item][1] for item in correction_times),
    )


def constrained_observations(
    observations: Sequence[Observation], labels: Sequence[Any], fit: BallFit,
) -> list[Observation]:
    """Return fitted render knots, with human labels winning frame collisions."""
    supports: dict[int, tuple[float, str, float]] = {
        item.frame_index: (item.t, "interpolated", item.confidence)
        for item in observations
    }
    for item in labels:
        supports[int(item.frame_index)] = (
            float(item.t), str(getattr(item, "source", "human")), 1.0,
        )
    result: list[Observation] = []
    for frame, (timestamp, source, confidence) in sorted(
        supports.items(), key=lambda item: (item[1][0], item[0]),
    ):
        point = fit.predict(timestamp)
        result.append(Observation(
            frame, timestamp, float(point[0]), float(point[1]),
            confidence, source,
        ))
    return result


class BallPhase(Phase):
    name = "ball"

    def __init__(
        self, config: Config = Config(), *,
        tee_roi: tuple[int, int, int, int] | None = None,
        tee_xy: tuple[float, float] | None = None,
        debug_dir: Any = None,
    ):
        self.config = config
        self.tee_roi = tee_roi
        self.debug_dir = debug_dir
        self.abstained = True
        self.reason: str | None = None
        self.tee_xy: tuple[float, float] | None = tee_xy
        self.metrics: dict[str, Any] = {}
        self.shaft_rule_fired = False
        self.debug_trace: dict[str, Any] = {}

    def _track_values(
        self,
        short_values: np.ndarray | None,
        long_values: np.ndarray | None,
        swing: Swing,
        config: Config,
        frame_height: int,
        *,
        extend_descent: bool = True,
    ) -> list[Observation]:
        vote_debug: dict[str, Any] = {}
        points, reason, metric = launch_vote(
            short_values, self.tee_xy, swing.impact_t, config, debug=vote_debug,
        )
        self.debug_trace["vote"] = vote_debug
        if points is None:
            self.abstained, self.reason = True, reason or "no_launch_track"
            return []
        launch = float(np.median(np.degrees(np.arctan2(metric["u"] - self.tee_xy[0], self.tee_xy[1] - metric["v"]))))
        hands_height = (
            config.ball_hands_height_px
            if config.ball_hands_height_px is not None
            else config.ball_hands_height_ratio * frame_height
        )
        if config.ball_shaft_rule_enabled and launch <= config.ball_shaft_launch_max_deg and min(point["v"] for point in points) >= hands_height:
            LOG.info("shaft corridor rule rejected swing %s at launch %.2f deg", swing.id, launch)
            self.abstained, self.reason, self.shaft_rule_fired = True, "shaft_corridor", True
            return []
        self.abstained, self.reason = False, None
        ascent = list(points)
        descent = (
            _descent_extension(
                ascent,
                long_values if long_values is not None else short_values,
                config,
            )
            if extend_descent else None
        )
        if descent is not None:
            extension, descent_metrics = descent
            points = [*ascent, *extension]
        else:
            descent_metrics = {"n_descent": 0}
        self.metrics = {
            "rise_px": metric["rise"], "lateral_px": metric["lateral"],
            "start_gap_px": metric["start_gap"], "launch_deg": launch,
            "n_observed": len(ascent), "n_total_observed": len(points),
            "n_ascent": len(ascent), **descent_metrics,
        }
        self.debug_trace["longest_rising"] = [
            int(point["rel_frame"]) for point in ascent
        ]
        self.debug_trace["descent"] = descent_metrics
        points, dropped_static = _drop_static_repeats(points, config)
        if dropped_static:
            self.metrics["n_static_repeats_dropped"] = dropped_static
            self.metrics["n_total_observed"] = len(points)
        self.debug_trace["final_points"] = points
        return [
            Observation(
                int(point["rel_frame"]), float(point["t_s"]),
                float(point["u"]), float(point["v"]), source="observed",
            )
            for point in points
        ]

    def track_video(
        self, video: str, swing: Swing, config: Config | None = None,
    ) -> list[Observation]:
        """Run the calibrated chain from v1-style PTS-decoded windows."""
        cfg = config or self.config
        if self.tee_xy is None:
            tee_start = max(0.0, swing.impact_t - cfg.tee_pre_s)
            tee_frames, tee_timestamps = read_window_pts(
                video, tee_start, cfg.tee_pre_s + cfg.tee_post_s,
                fps=30.0, gray=False,
            )
            self.tee_xy = estimate_tee_frames(
                tee_frames, tee_timestamps, swing.impact_t, cfg,
                roi=self.tee_roi,
            )
        if self.tee_xy is None:
            if cfg.ball_require_measured_tee:
                self.abstained, self.reason = True, "tee_not_found"
                return []
            self.tee_xy = cfg.ball_tee_xy
        start = swing.impact_t - cfg.ball_pre_s
        short_frames, short_pts = read_window_pts(
            video, start, cfg.ball_pre_s + cfg.ball_post_s,
            fps=cfg.ball_fps, gray=True,
        )
        short_values, short_debug = candidate_observations_from_window(
            short_frames, short_pts, swing.impact_t, cfg,
        )
        self.debug_trace["short_window"] = short_debug
        initial = self._track_values(
            short_values, None, swing, cfg,
            int(short_frames.shape[1]) if len(short_frames) else 0,
            extend_descent=cfg.ball_descent_post_s <= cfg.ball_post_s,
        )
        if not initial or cfg.ball_descent_post_s <= cfg.ball_post_s:
            result = initial
        else:
            long_frames, long_pts = read_window_pts(
                video, start, cfg.ball_pre_s + cfg.ball_descent_post_s,
                fps=cfg.ball_fps, gray=True,
            )
            long_values, long_debug = candidate_observations_from_window(
                long_frames, long_pts, swing.impact_t, cfg,
            )
            self.debug_trace["long_window"] = long_debug
            result = self._track_values(
                short_values, long_values, swing, cfg,
                int(short_frames.shape[1]),
            )
        if self.debug_dir is not None:
            from pathlib import Path
            destination = Path(self.debug_dir)
            destination.mkdir(parents=True, exist_ok=True)
            projection = short_debug.get("projection")
            records = short_debug.get("records")
            if projection is not None and records is not None:
                cv2.imwrite(
                    str(destination / f"swing-{swing.id:03d}-candidates.png"),
                    candidate_overlay(projection, records),
                )
        return result

    def track(self, frames: np.ndarray, swing: Swing, config: Config) -> list[Observation]:
        """Array entry point for fixtures; production uses :meth:`track_video`."""
        timestamps = np.asarray(
            swing.metadata.get(
                "ball_timestamps_s",
                swing.window_start + np.arange(len(frames)) / config.ball_fps,
            ),
            dtype=float,
        )
        if self.tee_xy is None:
            self.tee_xy = estimate_tee_frames(
                frames, timestamps, swing.impact_t, config, roi=self.tee_roi,
            )
        if self.tee_xy is None:
            if config.ball_require_measured_tee:
                self.abstained, self.reason = True, "tee_not_found"
                return []
            self.tee_xy = config.ball_tee_xy
        short_start = int(np.searchsorted(
            timestamps, swing.impact_t - config.ball_pre_s, side="left",
        ))
        short_end = int(np.searchsorted(
            timestamps,
            swing.impact_t + config.ball_post_s - 1e-9,
            side="left",
        ))
        short_frames = frames[short_start:short_end]
        short_pts = timestamps[short_start:short_end]
        short_values, short_debug = candidate_observations_from_window(
            short_frames, short_pts, swing.impact_t, config,
        )
        long_end = int(np.searchsorted(
            timestamps,
            swing.impact_t + config.ball_descent_post_s - 1e-9,
            side="left",
        ))
        long_frames = frames[short_start:long_end]
        long_pts = timestamps[short_start:long_end]
        long_values, long_debug = candidate_observations_from_window(
            long_frames, long_pts, swing.impact_t, config,
        )
        self.debug_trace["short_window"] = short_debug
        self.debug_trace["long_window"] = long_debug
        return self._track_values(
            short_values, long_values, swing, config, frames.shape[1],
        )

    def fit(self, observations: Sequence[Observation], labels: Sequence[Any]) -> BallFit | None:
        return label_constrained_fit(observations, labels)

    def retime(self, spline: BallFit, frames: np.ndarray, *, fps: float, start_t: float) -> list[Observation]:
        times = start_t + np.arange(len(frames)) / fps
        points = spline.predict(times)
        return [Observation(index, float(times[index]), float(point[0]), float(point[1]), source="interpolated") for index, point in enumerate(points)]

    def gates(self) -> Mapping[str, float | bool]:
        return {"min_rise_px": self.config.ball_min_rise_px, "shaft_rule": self.config.ball_shaft_rule_enabled}

    def audit(self, positions: Sequence[Observation], labels: Sequence[Any], observations: Sequence[Observation] = ()) -> AuditReport:
        del observations
        if self.abstained:
            return AuditReport(True, metrics={"abstained": 1.0}, failures=[])
        if labels:
            labels_by_frame = {int(item.frame_index): item for item in labels}
            rows: list[AuditFrame] = []
            residuals: list[float] = []
            for item in positions:
                label = labels_by_frame.get(item.frame_index)
                if label is None:
                    rows.append(AuditFrame(item.frame_index, item.t, OBSERVED))
                    continue
                residual = float(np.hypot(item.x - label.x, item.y - label.y))
                residuals.append(residual)
                rows.append(AuditFrame(
                    item.frame_index, item.t, LABELLED, residual,
                ))
            maximum = max(residuals, default=float("inf"))
            passed = (
                bool(rows) and len(residuals) == len(labels_by_frame)
                and maximum <= 1e-6
            )
            return AuditReport(
                passed,
                [] if passed else ["ball label constraints were not preserved"],
                {
                    "frames": float(len(rows)), "abstained": 0.0,
                    "labels": float(len(labels_by_frame)),
                    "max_label_residual_px": maximum,
                    "rms_label_residual_px": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf"),
                },
                rows,
            )
        rows = [AuditFrame(item.frame_index, item.t, OBSERVED) for item in positions]
        return AuditReport(bool(rows), [] if rows else ["tracked ball has no positions"], {"frames": float(len(rows)), "abstained": 0.0}, rows)
