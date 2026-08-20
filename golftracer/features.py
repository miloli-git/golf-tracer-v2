"""Image-plane start and curve features from a robust fitted track."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .fit import FitResult


def _signed_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cross = float(first[0] * second[1] - first[1] * second[0])
    return float(np.degrees(np.arctan2(cross, float(np.dot(first, second)))))


def _initial_line(
    observations: Sequence[dict[str, Any]], count: int
) -> tuple[np.ndarray, np.ndarray, float]:
    initial = sorted(observations, key=lambda item: item["t_s"])[:count]
    times = np.asarray([float(item["t_s"]) for item in initial])
    origin = float(times[0])
    x = times - origin
    positions = np.asarray([(item["u"], item["v"]) for item in initial], dtype=float)
    sigmas = np.asarray([
        max(1.0, np.sqrt(item.get("sigma_major", 1.5) * item.get("sigma_minor", 1.0)))
        for item in initial
    ])
    weights = 1.0 / sigmas
    u_coefficients = np.polyfit(x, positions[:, 0], 1, w=weights)
    v_coefficients = np.polyfit(x, positions[:, 1], 1, w=weights)
    return (
        np.asarray((u_coefficients[1], v_coefficients[1]), dtype=float),
        np.asarray((u_coefficients[0], v_coefficients[0]), dtype=float),
        origin,
    )


def extract_features(
    observations: Sequence[dict[str, Any]], fit: FitResult
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return start-line and curve features in stabilized image coordinates."""
    ordered = sorted(observations, key=lambda item: item["t_s"])
    reliable = [item for item, keep in zip(ordered, fit.inlier_mask, strict=True) if keep]
    if len(reliable) < 5:
        raise ValueError("five reliable observations are required for features")
    initial_count = min(12, max(5, int(round(0.60 * len(reliable)))))
    anchor, tangent, initial_time = _initial_line(reliable, initial_count)
    speed = float(np.linalg.norm(tangent))
    if speed < 1e-6:
        raise ValueError("initial tangent is degenerate")
    direction = tangent / speed
    start_angle = float(np.degrees(np.arctan2(direction[0], -direction[1])))
    horizontal_fraction = float(direction[0])
    start_sign = (
        "vertical" if abs(horizontal_fraction) < 0.02
        else "right" if horizontal_fraction > 0 else "left"
    )
    start_feature = {
        "angle_from_image_up_deg": round(start_angle, 4),
        "sign": start_sign,
        "initial_observation_count": initial_count,
        "initial_tangent_px_per_s": [round(float(value), 4) for value in tangent],
        "reference": "image up (negative v); positive angle points image right",
        "units": "image-plane degrees",
    }
    late_time = float(reliable[-1]["t_s"])
    continued = anchor + tangent * (late_time - initial_time)
    fitted_late = np.asarray(fit.predict(late_time), dtype=float)
    residual = fitted_late - continued
    image_right_normal = np.asarray((-direction[1], direction[0]))
    signed_lateral = float(np.dot(residual, image_right_normal))
    late_tangent = np.asarray(fit.derivative(late_time), dtype=float)
    late_norm = float(np.linalg.norm(late_tangent))
    tangent_change = (
        _signed_angle_degrees(direction, late_tangent / late_norm)
        if late_norm > 1e-6 else 0.0
    )
    path_length = max(1.0, float(np.sum(np.linalg.norm(np.diff(
        np.asarray([(item["u"], item["v"]) for item in reliable]), axis=0
    ), axis=1))))
    composite = signed_lateral / max(5.0, 0.05 * path_length) + tangent_change / 12.0
    curve_sign = "neutral" if abs(composite) < 0.12 else "right" if composite > 0 else "left"
    curve_feature = {
        "sign": curve_sign,
        "signed_lateral_residual_px": round(signed_lateral, 4),
        "tangent_change_deg": round(tangent_change, 4),
        "composite_signed_score": round(float(composite), 4),
        "late_time_s": round(late_time, 6),
        "sign_convention": "positive is image-right of the initial tangent",
        "units": "image-plane pixels and degrees",
    }
    return start_feature, curve_feature
