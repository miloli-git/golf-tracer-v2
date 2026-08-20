"""Golden-only overlay stroke-mask comparison against a concatenated oracle reel."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import numpy as np

from .config import Config
from .decode import decode_window
from .oracle import load_oracle_session
from .render.compositor import draw_faded_path as draw_v2_path, fit_track
from .render.styles import get_style


def stroke_mask(rendered: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Isolate yellow tracer strokes while suppressing unchanged source colour."""
    image = rendered.astype(np.int16)
    base = source.astype(np.int16)
    delta = np.max(np.abs(image - base), axis=2)
    blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    tracer_hue = (
        (red >= 115) & (green >= 100) & (red - blue >= 28) & (green - blue >= 28)
    )
    mask = (tracer_hue & (delta >= 10)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0


def golden_iou_report(
    manifest: Mapping[str, Any],
    rendered_reel: str | Path,
    output_path: str | Path,
    config: Config = Config(),
    *,
    sample_fps: float = 5.0,
) -> dict[str, Any]:
    """Compare corresponding concatenated clip windows and write an honest report."""
    session = load_oracle_session(manifest, config)
    oracle_reel = Path(str(manifest["combined"]["oracle_reel"]))
    rendered_reel = Path(rendered_reel)
    offset = 0.0
    swings: list[dict[str, Any]] = []
    for swing in session.swings:
        duration = swing.window_end - swing.window_start
        source_frames, _ = decode_window(
            session.video, swing.window_start, duration, fps=sample_fps, gray=False
        )
        v2_frames, _ = decode_window(rendered_reel, offset, duration, fps=sample_fps, gray=False)
        v1_frames, _ = decode_window(oracle_reel, offset, duration, fps=sample_fps, gray=False)
        count = min(len(source_frames), len(v2_frames), len(v1_frames))
        intersection = 0
        union = 0
        for index in range(count):
            v2_mask = stroke_mask(v2_frames[index], source_frames[index])
            v1_mask = stroke_mask(v1_frames[index], source_frames[index])
            intersection += int(np.count_nonzero(v2_mask & v1_mask))
            union += int(np.count_nonzero(v2_mask | v1_mask))
        iou = 1.0 if union == 0 else intersection / union
        swings.append({
            "swing_id": swing.id,
            "frames": count,
            "intersection_px": intersection,
            "union_px": union,
            "iou": round(iou, 6),
        })
        offset += duration
    values = [float(item["iou"]) for item in swings]
    report = {
        "sample_fps": sample_fps,
        "target_iou": 0.95,
        "minimum_iou": min(values) if values else None,
        "mean_iou": sum(values) / len(values) if values else None,
        "swings": swings,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _load_v1_renderer(repo: Path):
    script = repo / "scripts" / "render_v4.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("golftracer_v1_render_v4", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def overlay_only_iou_report(
    manifest: Mapping[str, Any], output_path: str | Path,
    config: Config = Config(), *, sample_count: int = 24,
) -> dict[str, Any]:
    """Render identical oracle tracks on black through v1/v2 and compare masks."""
    v1 = _load_v1_renderer(Path(str(manifest["v1_repo"])))
    club_document = json.loads(Path(str(manifest["club"]["oracle_arcs"])).read_text(encoding="utf-8"))
    ball_rows = json.loads(Path(str(manifest["ball"]["oracle_tracks"])).read_text(encoding="utf-8"))
    session = load_oracle_session(manifest, config)
    club_by_impact = {round(float(item["t_impact"]), 3): item for item in club_document["swings"]}
    ball_by_impact = {round(float(item["t"]), 3): item for item in ball_rows if item.get("ok")}
    width, height = (int(value) for value in manifest["decoded_size"])
    swings = []
    for swing in session.swings:
        key = round(float(swing.impact_t), 3)
        v1_club = v1.fit_club_track(club_by_impact[key])
        v2_club_track = next(item for item in swing.tracks if item.phase == "club")
        v2_club = fit_track(v2_club_track, config)
        assert v2_club is not None
        raw_ball = ball_by_impact.get(key)
        v1_ball = v1.fit_ball_track(raw_ball) if raw_ball is not None else None
        v2_ball_track = next((item for item in swing.tracks if item.phase == "ball"), None)
        v2_ball = fit_track(v2_ball_track, config) if v2_ball_track is not None else None
        end_t = max(v1_club.t1, swing.impact_t + (config.ball_fade_length_s if raw_ball else 0.0))
        frame_ious = []
        for timestamp in np.linspace(v1_club.t0, end_t, sample_count):
            first = np.zeros((height, width, 3), np.uint8)
            second = np.zeros_like(first)
            v1.draw_faded_path(first, v1_club.samples_until(float(timestamp), fps=240.0), kind="club")
            draw_v2_path(
                second,
                v2_club.samples_until(float(timestamp), fps=240.0, fade_length_s=config.club_fade_length_s),
                get_style("club", config), config,
            )
            if v1_ball is not None and v2_ball is not None:
                v1.draw_faded_path(first, v1_ball.samples_until(float(timestamp)), kind="ball")
                draw_v2_path(
                    second,
                    v2_ball.samples_until(float(timestamp), fps=60.0, fade_length_s=config.ball_fade_length_s),
                    get_style("ball", config), config, ball=True,
                )
            first_mask = np.any(first > 0, axis=2)
            second_mask = np.any(second > 0, axis=2)
            union = int(np.count_nonzero(first_mask | second_mask))
            intersection = int(np.count_nonzero(first_mask & second_mask))
            frame_ious.append(1.0 if union == 0 else intersection / union)
        swings.append({
            "swing_id": swing.id,
            "frames": len(frame_ious),
            "mean_iou": float(np.mean(frame_ious)),
            "min_iou": float(np.min(frame_ious)),
        })
    report = {
        "target_iou": 0.95,
        "mean_iou": float(np.mean([item["mean_iou"] for item in swings])),
        "minimum_iou": float(np.min([item["min_iou"] for item in swings])),
        "swings": swings,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
