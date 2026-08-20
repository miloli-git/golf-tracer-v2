"""Dual-polarity streak candidates from stabilized post-impact frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CandidateConfig:
    background_frames: int = 15
    temporal_lag: int = 3
    min_contrast: float = 14.0
    noise_sigma: float = 5.0
    min_area: int = 2
    max_area: int = 1600
    analysis_height_fraction: float = 0.72
    close_kernel: int = 3
    background_level_weight: float = 0.08
    background_edge_weight: float = 0.25
    max_spatial_penalty: float = 30.0
    persistent_edge_floor: float = 12.0
    persistent_edge_weight: float = 0.30
    persistent_edge_max_penalty: float = 40.0
    persistent_edge_dilate: int = 9


def rolling_preimpact_median(frames: np.ndarray, impact_index: int, window: int = 15) -> np.ndarray:
    if not 0 < impact_index <= len(frames):
        raise ValueError("impact_index must have at least one pre-impact frame")
    return np.median(frames[max(0, impact_index - window):impact_index], axis=0).astype(np.uint8)


def _threshold(background_diff: np.ndarray, lag_diff: np.ndarray, mask: np.ndarray, config: CandidateConfig) -> float:
    valid = mask[::4, ::4] > 0
    sample = np.r_[background_diff[::4, ::4][valid], lag_diff[::4, ::4][valid]].astype(np.float32)
    centre = float(np.median(sample))
    sigma = 1.4826 * float(np.median(np.abs(sample - centre)))
    return float(max(config.min_contrast, config.noise_sigma * max(1.0, sigma)))


def _component(
    labels: np.ndarray, label: int, stats: np.ndarray, evidence: np.ndarray,
    threshold: np.ndarray, timestamp: float, polarity: str, edge_penalty: np.ndarray,
) -> dict[str, Any]:
    left, top, width, height = [int(value) for value in stats[label, :4]]
    local_y, local_x = np.nonzero(labels[top:top + height, left:left + width] == label)
    y, x = local_y + top, local_x + left
    raw = evidence[y, x].astype(np.float64)
    weights = np.maximum(raw, 1.0)
    total = float(weights.sum())
    u, v = float(x @ weights / total), float(y @ weights / total)
    centered = np.column_stack((x - u, y - v))
    covariance = (centered * weights[:, None]).T @ centered / total
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major, minor = np.sqrt(np.maximum(values[order], 0.0) + 0.25)
    vector = vectors[:, order[0]]
    orientation = (float(np.degrees(np.arctan2(vector[1], vector[0]))) + 90.0) % 180.0 - 90.0
    return {
        "t_s": round(float(timestamp), 6), "u": round(u, 4), "v": round(v, 4),
        "area": int(len(x)), "elongation": round(float(major / max(minor, 1e-6)), 4),
        "orientation_deg": round(orientation, 4), "contrast": round(float(raw.mean()), 4),
        "polarity": polarity,
        "local_threshold": round(float(np.average(threshold[y, x], weights=weights)), 4),
        "excess_contrast": round(float(np.average(raw - threshold[y, x], weights=weights)), 4),
        "persistent_edge_penalty": round(float(np.average(edge_penalty[y, x], weights=weights)), 4),
        "sigma_major": round(float(major), 4), "sigma_minor": round(float(minor), 4),
    }


def _components(evidence: np.ndarray, threshold: np.ndarray, mask: np.ndarray, timestamp: float, polarity: str, config: CandidateConfig, edge_penalty: np.ndarray) -> list[dict[str, Any]]:
    binary = ((evidence >= threshold) & (mask > 0)).astype(np.uint8)
    if config.close_kernel > 1:
        size = int(config.close_kernel) | 1
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    result = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if config.min_area <= area <= config.max_area:
            result.append(_component(labels, label, stats, evidence, threshold, timestamp, polarity, edge_penalty))
    return result


def _persistent_penalty(frames: np.ndarray, impact_index: int, config: CandidateConfig) -> np.ndarray:
    start = max(0, impact_index - config.background_frames)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stack = np.stack([cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel) for frame in frames[start:impact_index]])
    persistent = np.median(stack, axis=0).astype(np.float32)
    size = max(1, int(config.persistent_edge_dilate)) | 1
    if size > 1:
        persistent = cv2.dilate(persistent, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return np.clip(config.persistent_edge_weight * np.maximum(0.0, persistent - config.persistent_edge_floor), 0.0, config.persistent_edge_max_penalty)


def extract_candidate_observations(
    registered_frames: np.ndarray,
    impact_index: int,
    fps: float,
    start_time_s: float = 0.0,
    timestamps_s: Sequence[float] | None = None,
    config: CandidateConfig | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    cfg = config or CandidateConfig()
    frames = np.asarray(registered_frames)
    if frames.ndim != 3 or not 0 < impact_index < len(frames):
        raise ValueError("registered_frames must be grayscale and impact_index must split it")
    background = rolling_preimpact_median(frames, impact_index, cfg.background_frames)
    mask = np.zeros(frames.shape[1:], np.uint8)
    mask[:max(1, round(frames.shape[1] * cfg.analysis_height_fraction))] = 255
    edge = cv2.morphologyEx(background, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
    spatial = np.clip(
        cfg.background_level_weight * background.astype(np.float32)
        + cfg.background_edge_weight * edge,
        0.0,
        cfg.max_spatial_penalty,
    )
    persistent = _persistent_penalty(frames, impact_index, cfg)
    projection = np.zeros(frames.shape[1:], np.uint8)
    records = []
    for frame_index in range(impact_index, len(frames)):
        current = frames[frame_index].astype(np.int16)
        background_diff = current - background.astype(np.int16)
        lag = current - frames[max(0, frame_index - cfg.temporal_lag)].astype(np.int16)
        floor = _threshold(background_diff, lag, mask, cfg)
        threshold_map = floor + spatial + persistent
        bright = np.maximum(background_diff, lag).astype(np.float32)
        dark = np.maximum(-background_diff, -lag).astype(np.float32)
        projection = np.maximum(projection, np.clip(np.maximum(bright, dark), 0, 255).astype(np.uint8))
        timestamp = float(timestamps_s[frame_index]) if timestamps_s is not None else start_time_s + frame_index / fps
        candidates = _components(bright, threshold_map, mask, timestamp, "bright", cfg, persistent)
        candidates += _components(dark, threshold_map, mask, timestamp, "dark", cfg, persistent)
        candidates.sort(key=lambda item: (item["v"], item["u"], item["polarity"]))
        records.append({"frame_index": frame_index, "t_s": round(timestamp, 6), "threshold": round(floor, 4), "candidates": candidates})
    return records, projection, background


def candidate_overlay(projection: np.ndarray, records: Sequence[dict[str, Any]]) -> np.ndarray:
    canvas = cv2.cvtColor(projection, cv2.COLOR_GRAY2BGR)
    for row, record in enumerate(records):
        colour = tuple(int(value) for value in cv2.cvtColor(np.uint8([[[int(179 * row / max(1, len(records) - 1)), 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        for item in record["candidates"]:
            cv2.circle(canvas, (round(item["u"]), round(item["v"])), 5, colour, 1, cv2.LINE_AA)
    return canvas
