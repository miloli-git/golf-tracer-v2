"""Deterministic micro-shake registration for short golf-shot windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationQuality:
    frame_index: int
    matrix: list[list[float]]
    quality: float
    ecc: float
    residual_before: float
    residual_after: float
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def static_registration_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    top, bottom = int(0.34 * height), int(0.66 * height)
    mask[top:bottom, int(0.56 * width):] = 255
    mask[int(0.49 * height):bottom, :int(0.16 * width)] = 255
    return mask


def _gray_u8(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim == 4 and array.shape[-1] == 3:
        return np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in array])
    if array.ndim != 3:
        raise ValueError("frames must have shape (N,H,W) or (N,H,W,3)")
    return array if array.dtype == np.uint8 else np.clip(array, 0, 255).astype(np.uint8)


def _prepare(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(small, (0, 0), 3.0)
    high_pass = cv2.addWeighted(small, 1.0, smooth, -1.0, 128.0)
    return cv2.GaussianBlur(high_pass, (3, 3), 0).astype(np.float32) / 255.0


def _residual(reference: np.ndarray, other: np.ndarray, mask: np.ndarray) -> float:
    pixels = cv2.absdiff(reference, other)[mask > 0]
    return float(np.mean(pixels)) if pixels.size else 0.0


def stabilize_frames(
    frames: np.ndarray,
    reference_indices: Iterable[int] | None = None,
    mask: np.ndarray | None = None,
    max_dimension: int = 480,
) -> tuple[np.ndarray, list[RegistrationQuality]]:
    gray = _gray_u8(frames)
    if not len(gray):
        raise ValueError("at least one frame is required")
    height, width = gray.shape[1:]
    mask = static_registration_mask(height, width) if mask is None else np.asarray(mask, np.uint8)
    if mask.shape != (height, width):
        raise ValueError("mask shape must match frame shape")
    selected = np.arange(min(9, len(gray))) if reference_indices is None else np.asarray(list(reference_indices), int)
    selected = selected[(selected >= 0) & (selected < len(gray))]
    if not len(selected):
        raise ValueError("reference_indices selected no frames")
    reference = np.median(gray[selected], axis=0).astype(np.uint8)
    scale = min(1.0, max_dimension / max(height, width))
    size = (max(32, round(width * scale)), max(32, round(height * scale)))
    reference_small = _prepare(reference, size)
    mask_small = cv2.erode(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST), np.ones((5, 5), np.uint8))
    mask_float = mask_small.astype(np.float32) / 255.0
    reference_phase = (reference_small - 0.5) * mask_float
    hanning = cv2.createHanningWindow(size, cv2.CV_32F)
    registered = np.empty_like(gray)
    diagnostics: list[RegistrationQuality] = []
    previous = np.eye(2, 3, dtype=np.float32)
    for index, frame in enumerate(gray):
        current = (_prepare(frame, size) - 0.5) * mask_float
        matrix_small = np.eye(2, 3, dtype=np.float32)
        response = 0.0
        try:
            (dx, dy), response = cv2.phaseCorrelate(reference_phase, current, hanning)
            matrix_small[:, 2] = (dx, dy)
        except cv2.error:
            pass
        if np.linalg.norm(matrix_small[:, 2] - previous[:, 2]) < 0.08 * scale:
            matrix_small = previous.copy()
        matrix = matrix_small.copy()
        matrix[:, 2] /= scale
        aligned = cv2.warpAffine(
            frame, matrix, (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT101,
        )
        before = _residual(reference, frame, mask)
        after = _residual(reference, aligned, mask)
        translation = np.linalg.norm(matrix_small[:, 2] / scale)
        accepted = bool(response >= 0.08 and np.isfinite(matrix).all() and translation <= 18.0 and after + max(0.05, 0.005 * before) < before)
        if accepted:
            previous = matrix_small
            registered[index] = aligned
        else:
            matrix = np.eye(2, 3, dtype=np.float32)
            registered[index] = frame
            after = before
        improvement = 1.0 if before < 1e-6 else float(np.clip(1.0 - after / before, 0.0, 1.0))
        quality = float(np.clip(0.75 * max(0.0, response) + 0.25 * improvement, 0.0, 1.0)) if accepted else 0.0
        diagnostics.append(RegistrationQuality(
            index, np.round(matrix.astype(float), 7).tolist(), round(quality, 6),
            round(float(response), 6), round(before, 6), round(after, 6), accepted,
        ))
    return registered, diagnostics
