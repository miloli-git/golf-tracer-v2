"""Robust image-plane polynomial fitting for a selected ball track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FitResult:
    degree: int
    t_origin_s: float
    t_scale_s: float
    u_coefficients: tuple[float, ...]
    v_coefficients: tuple[float, ...]
    inlier_mask: tuple[bool, ...]
    residuals_px: tuple[float, ...]
    rmse_px: float
    median_residual_px: float
    max_residual_px: float
    weighted_rmse: float
    success: bool

    def predict(self, times_s: float | Sequence[float]) -> np.ndarray:
        values = np.atleast_1d(np.asarray(times_s, dtype=float))
        x = (values - self.t_origin_s) / self.t_scale_s
        result = np.column_stack((
            np.polynomial.polynomial.polyval(x, self.u_coefficients),
            np.polynomial.polynomial.polyval(x, self.v_coefficients),
        ))
        return result[0] if np.ndim(times_s) == 0 else result

    def derivative(self, times_s: float | Sequence[float]) -> np.ndarray:
        values = np.atleast_1d(np.asarray(times_s, dtype=float))
        x = (values - self.t_origin_s) / self.t_scale_s
        du = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyder(self.u_coefficients)
        ) / self.t_scale_s
        dv = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyder(self.v_coefficients)
        ) / self.t_scale_s
        result = np.column_stack((du, dv))
        return result[0] if np.ndim(times_s) == 0 else result

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": f"polynomial_{self.degree}d",
            "degree": self.degree,
            "time_basis": {
                "t_origin_s": round(self.t_origin_s, 6),
                "t_scale_s": round(self.t_scale_s, 6),
                "definition": "x = (t_s - t_origin_s) / t_scale_s",
            },
            "u_coefficients": [round(value, 8) for value in self.u_coefficients],
            "v_coefficients": [round(value, 8) for value in self.v_coefficients],
            "inlier_mask": list(self.inlier_mask),
            "residuals_px": [round(value, 4) for value in self.residuals_px],
            "rmse_px": round(self.rmse_px, 4),
            "median_residual_px": round(self.median_residual_px, 4),
            "max_residual_px": round(self.max_residual_px, 4),
            "weighted_rmse": round(self.weighted_rmse, 4),
            "success": self.success,
            "coordinate_system": "stabilized image pixels",
        }


def _covariance(item: dict[str, Any], model_floor_px: float) -> np.ndarray:
    major = max(0.5, float(item.get("sigma_major", 1.5)))
    minor = max(0.5, float(item.get("sigma_minor", 1.0)))
    angle = np.radians(float(item.get("orientation_deg", 0.0)))
    major_axis = np.asarray((np.cos(angle), np.sin(angle)))
    minor_axis = np.asarray((-np.sin(angle), np.cos(angle)))
    covariance = (
        major * major * np.outer(major_axis, major_axis)
        + minor * minor * np.outer(minor_axis, minor_axis)
    )
    covariance += np.eye(2) * model_floor_px * model_floor_px
    return covariance


def _fit_degree(
    observations: Sequence[dict[str, Any]],
    degree: int,
    inlier_mask: np.ndarray | None = None,
    model_floor_px: float = 1.5,
) -> FitResult:
    times = np.asarray([float(item["t_s"]) for item in observations])
    positions = np.asarray(
        [(float(item["u"]), float(item["v"])) for item in observations], dtype=float
    )
    t_origin = float(times.min())
    t_scale = max(float(times.max() - times.min()), 1.0 / 60.0)
    x = (times - t_origin) / t_scale
    active = (
        np.ones(len(observations), dtype=bool)
        if inlier_mask is None else np.asarray(inlier_mask, dtype=bool)
    )
    if int(active.sum()) < degree + 1:
        raise ValueError("not enough observations for requested polynomial degree")
    covariances = np.stack([_covariance(item, model_floor_px) for item in observations])
    whiteners = np.stack([
        np.linalg.inv(np.linalg.cholesky(covariance)) for covariance in covariances
    ])
    u_initial = np.polynomial.polynomial.polyfit(x[active], positions[active, 0], degree)
    v_initial = np.polynomial.polynomial.polyfit(x[active], positions[active, 1], degree)
    initial = np.concatenate((u_initial, v_initial))

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        predicted = np.column_stack((
            np.polynomial.polynomial.polyval(x, parameters[:degree + 1]),
            np.polynomial.polynomial.polyval(x, parameters[degree + 1:]),
        ))
        whitened = np.einsum("nij,nj->ni", whiteners, predicted - positions)
        return whitened[active].ravel()

    optimized = least_squares(
        residual_function, initial, loss="huber", f_scale=1.5, max_nfev=400
    )
    u_coefficients = optimized.x[:degree + 1]
    v_coefficients = optimized.x[degree + 1:]
    predicted = np.column_stack((
        np.polynomial.polynomial.polyval(x, u_coefficients),
        np.polynomial.polynomial.polyval(x, v_coefficients),
    ))
    pixel_residual = np.linalg.norm(predicted - positions, axis=1)
    whitened_residual = np.einsum("nij,nj->ni", whiteners, predicted - positions)
    weighted_norm = np.linalg.norm(whitened_residual, axis=1)
    selected = pixel_residual[active]
    return FitResult(
        degree=degree,
        t_origin_s=t_origin,
        t_scale_s=t_scale,
        u_coefficients=tuple(float(value) for value in u_coefficients),
        v_coefficients=tuple(float(value) for value in v_coefficients),
        inlier_mask=tuple(bool(value) for value in active),
        residuals_px=tuple(float(value) for value in pixel_residual),
        rmse_px=float(np.sqrt(np.mean(selected * selected))),
        median_residual_px=float(np.median(selected)),
        max_residual_px=float(np.max(selected)),
        weighted_rmse=float(np.sqrt(np.mean(weighted_norm[active] ** 2))),
        success=bool(optimized.success),
    )


def robust_fit_2d(
    observations: Sequence[dict[str, Any]],
    max_degree: int = 3,
    model_floor_px: float = 1.5,
) -> FitResult:
    """Fit a quadratic/cubic path with Huber loss and anisotropic weights."""
    ordered = sorted(observations, key=lambda item: (item["t_s"], item.get("u", 0.0)))
    if len(ordered) < 3:
        raise ValueError("at least three observations are required")
    quadratic = _fit_degree(ordered, 2, model_floor_px=model_floor_px)
    selected = quadratic
    if max_degree >= 3 and len(ordered) >= 10:
        cubic = _fit_degree(ordered, 3, model_floor_px=model_floor_px)
        if cubic.rmse_px < 0.80 * quadratic.rmse_px and cubic.rmse_px < 8.0:
            selected = cubic
    residuals = np.asarray(selected.residuals_px)
    center = float(np.median(residuals))
    robust_sigma = 1.4826 * float(np.median(np.abs(residuals - center)))
    cutoff = max(4.0, center + 3.0 * max(1.0, robust_sigma))
    inliers = residuals <= cutoff
    if not np.all(inliers) and int(inliers.sum()) >= max(6, selected.degree + 2):
        selected = _fit_degree(
            ordered, selected.degree, inlier_mask=inliers,
            model_floor_px=model_floor_px,
        )
    return selected
