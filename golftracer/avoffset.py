"""Per-clip audio/video offset measurement from the tee-patch departure frame.

``Config.av_offset_s`` is a per-recording constant (LESSONS §9, issue #5). This
module measures it for a clip instead: for each audio impact, find the tee with
the configured tee estimator, sample a small grey patch at the tee on every
frame, and take the first frame at which the patch turns absent and stays
absent. ``offset = t_audio - t_departure``. Nothing here changes a default; the
caller decides whether to apply the measured value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .config import Config
from .decode import read_window_pts
from .phases.ball import estimate_tee_frames


@dataclass
class ImpactOffset:
    t_audio: float
    provisional_video: float
    tee_xy: tuple[float, float] | None
    t_departure: float | None
    offset_s: float | None
    contrast: float | None
    reason: str | None


def _patch_signal(frames: np.ndarray, tee_xy: tuple[float, float], half: int) -> np.ndarray:
    x, y = int(round(tee_xy[0])), int(round(tee_xy[1]))
    height, width = frames.shape[1], frames.shape[2]
    x0, x1 = max(0, x - half), min(width, x + half + 1)
    y0, y1 = max(0, y - half), min(height, y + half + 1)
    values = []
    for frame in frames:
        patch = frame[y0:y1, x0:x1]
        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        values.append(float(patch.mean()))
    return np.asarray(values, float)


def departure_time(
    signal: np.ndarray,
    timestamps: np.ndarray,
    provisional: float,
    *,
    pre_s: float = 0.10,
    post_s: float = 0.30,
    search_before_s: float = 0.25,
    search_after_s: float = 0.35,
    min_contrast: float = 12.0,
    min_absent_run: int = 5,
) -> tuple[float | None, float | None, str | None]:
    """First timestamp at which the tee patch is absent and stays absent.

    Returns ``(t_departure, contrast, reason)``; ``reason`` is set when no
    departure could be found.
    """
    pre = signal[timestamps < provisional - pre_s]
    post = signal[timestamps > provisional + post_s]
    if len(pre) < 5 or len(post) < 5:
        return None, None, "window_too_short"
    pre_level, post_level = float(np.median(pre)), float(np.median(post))
    contrast = abs(pre_level - post_level)
    if contrast < min_contrast:
        return None, contrast, "no_contrast"
    threshold = 0.5 * (pre_level + post_level)
    present = (signal > threshold) if pre_level > post_level else (signal < threshold)
    lo = np.searchsorted(timestamps, provisional - search_before_s)
    hi = np.searchsorted(timestamps, provisional + search_after_s, side="right")
    for index in range(max(lo, 1), min(hi, len(signal))):
        if present[index]:
            continue
        run = present[index:index + min_absent_run]
        if len(run) < min_absent_run or run.any():
            continue
        if not present[max(lo, 0):index].any():
            # never seen present inside the search window before this point
            continue
        return float(timestamps[index]), contrast, None
    return None, contrast, "no_departure_in_window"


def measure_impact_offset(
    video: str,
    t_audio: float,
    config: Config,
    *,
    roi: tuple[int, int, int, int] | None = None,
    patch_half_px: int = 3,
) -> ImpactOffset:
    provisional = max(0.0, t_audio - config.av_offset_s)
    start = max(0.0, provisional - config.tee_pre_s)
    frames, timestamps = read_window_pts(
        video, start, config.tee_pre_s + config.tee_post_s, fps=None, gray=False,
    )
    if len(frames) < 10:
        return ImpactOffset(t_audio, provisional, None, None, None, None, "decode_failed")
    tee = estimate_tee_frames(frames, timestamps, provisional, config, roi=roi)
    if tee is None:
        return ImpactOffset(t_audio, provisional, None, None, None, None, "no_tee")
    signal = _patch_signal(frames, tee, patch_half_px)
    t_departure, contrast, reason = departure_time(signal, timestamps, provisional)
    offset = None if t_departure is None else t_audio - t_departure
    return ImpactOffset(t_audio, provisional, tee, t_departure, offset, contrast, reason)


def measure_av_offset(
    video: str,
    audio_times: Sequence[float],
    config: Config,
    *,
    roi: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    rows = [measure_impact_offset(video, float(t), config, roi=roi) for t in audio_times]
    offsets = np.asarray([row.offset_s for row in rows if row.offset_s is not None], float)
    summary: dict[str, Any] = {
        "video": str(video),
        "configured_av_offset_s": config.av_offset_s,
        "impacts": len(rows),
        "measured": int(len(offsets)),
        "median_offset_s": None,
        "mad_s": None,
        "departure_lag_s": config.av_departure_lag_s,
        "impact_offset_estimate_s": None,
        "rows": [asdict(row) for row in rows],
    }
    if len(offsets):
        median = float(np.median(offsets))
        summary["median_offset_s"] = round(median, 4)
        summary["mad_s"] = round(float(np.median(np.abs(offsets - median))), 4)
        summary["impact_offset_estimate_s"] = round(median + config.av_departure_lag_s, 4)
    return summary


def summary_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        f"video: {summary['video']}",
        f"configured av_offset_s: {summary['configured_av_offset_s']:.3f}",
        f"impacts: {summary['impacts']}  measured: {summary['measured']}",
    ]
    if summary["median_offset_s"] is not None:
        lines.append(
            f"departure offset: median {summary['median_offset_s']:.3f} s, "
            f"MAD {summary['mad_s']:.3f} s"
        )
        lines.append(
            f"impact offset estimate (median + {summary['departure_lag_s']:.3f} s lag): "
            f"{summary['impact_offset_estimate_s']:.3f} s  -> pass as --av-offset"
        )
    lines.append("t_audio    provisional  departure   offset   contrast  reason")
    for row in summary["rows"]:
        dep = "--" if row["t_departure"] is None else f"{row['t_departure']:.3f}"
        off = "--" if row["offset_s"] is None else f"{row['offset_s']:+.3f}"
        con = "--" if row["contrast"] is None else f"{row['contrast']:.1f}"
        lines.append(
            f"{row['t_audio']:9.3f}  {row['provisional_video']:9.3f}    {dep:>8}  "
            f"{off:>7}  {con:>7}  {row['reason'] or ''}"
        )
    return lines


def apply_measured_av_offset(
    video: str,
    impacts: Sequence[dict[str, Any]],
    config: Config,
    *,
    roi: tuple[int, int, int, int] | None = None,
    min_measured: int = 1,
) -> tuple[list[dict[str, Any]], Config, dict[str, Any]]:
    """Re-derive ``t_video`` from a measured per-clip offset.

    Rows keep ``t_audio``; ``t_video`` becomes ``t_audio - estimate`` and each row
    records ``av_offset_applied_s``. When too few impacts measure, rows and
    config are returned unchanged and the summary says why.
    """
    audio_times = [float(row["t_audio"]) for row in impacts if "t_audio" in row]
    summary = measure_av_offset(video, audio_times, config, roi=roi)
    estimate = summary.get("impact_offset_estimate_s")
    if estimate is None or summary["measured"] < min_measured:
        summary["applied"] = False
        return list(impacts), config, summary
    updated: list[dict[str, Any]] = []
    for row in impacts:
        item = dict(row)
        if "t_audio" in item:
            item["t_video"] = round(max(0.0, float(item["t_audio"]) - estimate), 6)
            item["av_offset_applied_s"] = estimate
        updated.append(item)
    summary["applied"] = True
    return updated, config.with_overrides(av_offset_s=float(estimate)), summary
