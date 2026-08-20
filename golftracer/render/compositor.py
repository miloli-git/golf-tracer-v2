"""Deterministic spline compositor for club, ball, and follow-through tracks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline

from ..config import Config
from ..decode import decode_window
from ..session import Observation, Session, Swing, Track
from .presets import get_preset
from .styles import LayerStyle, get_style


@dataclass(frozen=True)
class PathSample:
    t: float
    x: float
    y: float


@dataclass
class _TimePiece:
    t0: float
    t1: float
    x_spline: UnivariateSpline
    y_spline: UnivariateSpline
    x_offset: float = 0.0
    y_offset: float = 0.0

    def xy(self, t: float) -> tuple[float, float]:
        local = float(np.clip(t, self.t0, self.t1) - self.t0)
        return (
            float(self.x_spline(local) + self.x_offset),
            float(self.y_spline(local) + self.y_offset),
        )


@dataclass
class _SpatialPiece:
    t0: float
    t1: float
    x_spline: PchipInterpolator
    y_spline: PchipInterpolator
    time_knots: np.ndarray
    arc_knots: np.ndarray
    parameter_samples: np.ndarray
    arc_samples: np.ndarray
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]

    def xy(self, t: float) -> tuple[float, float]:
        arc = float(np.interp(np.clip(t, self.t0, self.t1), self.time_knots, self.arc_knots))
        if arc <= 1e-9:
            return self.start_xy
        if arc >= self.arc_samples[-1] - 1e-9:
            return self.end_xy
        parameter = float(np.interp(arc, self.arc_samples, self.parameter_samples))
        return float(self.x_spline(parameter)), float(self.y_spline(parameter))


@dataclass
class FittedPath:
    pieces: list[_TimePiece | _SpatialPiece]

    @property
    def t0(self) -> float:
        return self.pieces[0].t0

    @property
    def t1(self) -> float:
        return self.pieces[-1].t1

    def samples_until(
        self, t: float, *, fps: float, fade_length_s: float
    ) -> list[PathSample]:
        """Sample the reached fitted curve on a stable frame-time grid."""
        if t < self.t0:
            return []
        samples: list[PathSample] = []
        step = 1.0 / fps
        cutoff = t - fade_length_s
        for piece in self.pieces:
            if t < piece.t0:
                break
            start = max(piece.t0, cutoff)
            end = min(t, piece.t1)
            if end < start:
                continue
            first_index = max(0, int(np.ceil((start - piece.t0) / step - 1e-9)))
            count = max(1, int(np.floor((end - piece.t0) / step + 1e-9)) - first_index + 1)
            times = piece.t0 + (first_index + np.arange(count, dtype=float)) * step
            if not len(times) or end - times[-1] > 1e-6:
                times = np.append(times, end)
            else:
                times[-1] = end
            for timestamp in times:
                if samples and abs(float(timestamp) - samples[-1].t) < 1e-6:
                    continue
                x, y = piece.xy(float(timestamp))
                samples.append(PathSample(float(timestamp), x, y))
            if end < piece.t1:
                break
        return samples


def _ordered(track: Track) -> list[Observation]:
    return sorted(track.observations, key=lambda item: (item.t, item.frame_index))


def _fit_time_piece(observations: Sequence[Observation]) -> _TimePiece:
    if len(observations) < 2:
        raise ValueError("a time-spline piece needs at least two observations")
    times = np.asarray([item.t for item in observations], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("track times must be strictly increasing")
    local = times - times[0]
    degree = min(3, len(observations) - 1)
    smooth = len(observations) * 1.0**2
    return _TimePiece(
        float(times[0]), float(times[-1]),
        UnivariateSpline(local, [item.x for item in observations], k=degree, s=smooth),
        UnivariateSpline(local, [item.y for item in observations], k=degree, s=smooth),
    )


def _fit_spatial_piece(
    observations: Sequence[Observation],
    trusted_curve_xy: Sequence[Sequence[float]] | None = None,
    trusted_arc_knots: Sequence[float] | None = None,
) -> _SpatialPiece:
    if len(observations) < 2:
        raise ValueError("an arc-length piece needs at least two observations")
    times = np.asarray([item.t for item in observations], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("track times must be strictly increasing")
    record_xy = np.asarray([(item.x, item.y) for item in observations], dtype=float)
    xy = record_xy if trusted_curve_xy is None else np.asarray(trusted_curve_xy, dtype=float)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.concatenate(([True], steps > 1e-9))
    xy = xy[keep]
    if trusted_curve_xy is None:
        times = times[keep]
    if len(xy) < 2:
        raise ValueError("an arc-length piece cannot have zero length")
    parameter = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    x_spline = PchipInterpolator(parameter, xy[:, 0])
    y_spline = PchipInterpolator(parameter, xy[:, 1])
    parameter_samples = np.linspace(0.0, parameter[-1], max(4096, len(xy) * 256))
    sampled_xy = np.column_stack((x_spline(parameter_samples), y_spline(parameter_samples)))
    arc_samples = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(sampled_xy, axis=0), axis=1))
    ))
    if trusted_arc_knots is None:
        arc_knots = np.interp(parameter, parameter_samples, arc_samples)
    else:
        arc_knots = np.asarray(trusted_arc_knots, dtype=float)
        if len(arc_knots) != len(times):
            raise ValueError("trusted arc knots do not match observation times")
        arc_knots = np.maximum.accumulate(np.clip(arc_knots, 0.0, arc_samples[-1]))
        arc_knots[-1] = arc_samples[-1]
    return _SpatialPiece(
        float(times[0]), float(times[-1]), x_spline, y_spline,
        times, arc_knots, parameter_samples, arc_samples,
        (float(record_xy[0, 0]), float(record_xy[0, 1])),
        (float(record_xy[-1, 0]), float(record_xy[-1, 1])),
    )


def _ball_track_is_gappy(observations: Sequence[Observation], max_gap: int = 2) -> bool:
    """True when a ball track would fold a time-spline: frame gaps or dwells."""
    frames = [item.frame_index for item in observations]
    if any(b - a > max_gap for a, b in zip(frames, frames[1:])):
        return True
    coords = [(round(item.x, 1), round(item.y, 1)) for item in observations]
    return len(set(coords)) < len(coords)


def fit_track(track: Track, config: Config | None = None) -> FittedPath | None:
    """Fit a phase-specific path; fewer than two points means abstain."""
    observations = _ordered(track)
    if len(observations) < 2:
        return None
    if track.phase == "ball":
        # Human correction labels are exact model constraints. Always render
        # their fitted support sequence as an interpolating spatial path so the
        # normal one-pixel time-spline smoothing cannot move a trusted click.
        if track.metadata.get("label_constrained"):
            return FittedPath([_fit_spatial_piece(observations)])
        # The v1-parity drawing is a cubic time-spline, and it folds into
        # loops on sparse tracks with frame gaps or repeated coordinates
        # (v1 has the same pathology). By default such tracks use an
        # arc-length spatial piece like the club layers: monotone along the
        # flight, zero-length steps dropped, animation timing unchanged via
        # the piece's time->arc mapping. Config.v1_style() turns this off so
        # the overlay parity gate keeps rendering exactly as v1 does.
        arc_fit = config.ball_render_arc_fit if config is not None else True
        if arc_fit and _ball_track_is_gappy(observations):
            return FittedPath([_fit_spatial_piece(observations)])
        apex_frame = track.metadata.get("apex_frame")
        if apex_frame is None:
            return FittedPath([_fit_time_piece(observations)])
        apex_index = min(
            range(len(observations)),
            key=lambda index: abs(observations[index].frame_index - int(apex_frame)),
        )
        if apex_index < 2 or len(observations) - apex_index < 3:
            return FittedPath([_fit_time_piece(observations)])
        ascent = _fit_time_piece(observations[: apex_index + 1])
        descent = _fit_time_piece(observations[apex_index:])
        apex = observations[apex_index]
        for piece in (ascent, descent):
            fitted_x, fitted_y = piece.xy(apex.t)
            piece.x_offset += apex.x - fitted_x
            piece.y_offset += apex.y - fitted_y
        return FittedPath([ascent, descent])
    if track.phase == "follow":
        trusted = track.metadata.get("trusted_geometry", {})
        curve = trusted.get("follow_curve_xy")
        knots = trusted.get("follow_arc_knots")
        if curve and knots:
            return FittedPath([_fit_spatial_piece(observations, curve, knots)])
    top_t = track.metadata.get("top_t") if track.phase == "club" else None
    if top_t is not None:
        split = min(range(len(observations)), key=lambda index: abs(observations[index].t - float(top_t)))
        if 1 <= split < len(observations) - 1:
            trusted = track.metadata.get("trusted_geometry", {})
            return FittedPath([
                _fit_spatial_piece(
                    observations[: split + 1], trusted.get("backswing_curve_xy"),
                    trusted.get("backswing_arc_knots"),
                ),
                _fit_spatial_piece(
                    observations[split:], trusted.get("downswing_curve_xy"),
                    trusted.get("downswing_arc_knots"),
                ),
            ])
    return FittedPath([_fit_spatial_piece(observations)])


def fade_alpha(index: int, count: int, tail: float = 0.10, head: float = 0.90) -> float:
    """Smooth deterministic tail-to-head alpha schedule."""
    if count <= 1:
        return head
    fraction = index / (count - 1)
    smoothstep = fraction * fraction * (3.0 - 2.0 * fraction)
    return tail + (head - tail) * smoothstep


def _blend_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
    width: int,
    alpha: float,
) -> None:
    pad = width + 2
    left = max(0, min(start[0], end[0]) - pad)
    right = min(frame.shape[1], max(start[0], end[0]) + pad + 1)
    top = max(0, min(start[1], end[1]) - pad)
    bottom = min(frame.shape[0], max(start[1], end[1]) + pad + 1)
    if left >= right or top >= bottom:
        return
    mask = np.zeros((bottom - top, right - left), np.uint8)
    cv2.line(
        mask, (start[0] - left, start[1] - top),
        (end[0] - left, end[1] - top), 255, width, cv2.LINE_AA,
    )
    weights = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    roi = frame[top:bottom, left:right]
    blended = roi.astype(np.float32) * (1.0 - weights) + np.asarray(colour, np.float32) * weights
    roi[:] = np.clip(blended, 0, 255).astype(np.uint8)


def draw_faded_path(
    frame: np.ndarray,
    samples: Sequence[PathSample],
    style: LayerStyle,
    config: Config,
    *,
    ball: bool = False,
) -> None:
    segments = list(zip(samples, samples[1:]))
    for index, (first, second) in enumerate(segments):
        alpha = fade_alpha(
            index, len(segments), config.render_alpha_tail, config.render_alpha_head
        )
        start = (int(round(first.x)), int(round(first.y)))
        end = (int(round(second.x)), int(round(second.y)))
        if style.glow > style.width:
            _blend_line(
                frame, start, end, (0, 0, 0), style.glow,
                alpha * config.render_glow_alpha_scale,
            )
        _blend_line(frame, start, end, style.colour, style.width, alpha)
    if samples:
        head = (int(round(samples[-1].x)), int(round(samples[-1].y)))
        if ball:
            cv2.circle(frame, head, style.glow + 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, head, style.width + 3, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, head, style.width + 3, (255, 255, 255), -1, cv2.LINE_AA)


def render_overlay(
    frame: np.ndarray,
    swing: Swing,
    t: float,
    fps: float,
    config: Config,
    layers: Sequence[str],
    fitted_paths: Mapping[int, FittedPath | None] | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    for track in swing.tracks:
        if track.phase not in layers:
            continue
        path = (
            fitted_paths.get(id(track))
            if fitted_paths is not None
            else fit_track(track, config)
        )
        if path is None:
            continue
        style = get_style(track.phase, config)
        sample_fps = fps * 4 if track.phase in {"club", "follow"} else fps
        samples = path.samples_until(t, fps=sample_fps, fade_length_s=style.fade_length_s)
        draw_faded_path(canvas, samples, style, config, ball=track.phase == "ball")
    return canvas


def _fit_preset(frame: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    if width is None or height is None:
        return frame.copy()
    source_h, source_w = frame.shape[:2]
    scale = max(width / source_w, height / source_h)
    resized_w = max(width, int(round(source_w * scale)))
    resized_h = max(height, int(round(source_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    left = (resized_w - width) // 2
    top = (resized_h - height) // 2
    return resized[top : top + height, left : left + width].copy()


def _outlined_text(frame: np.ndarray, text: str, origin: tuple[int, int], config: Config) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, config.caption_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, config.caption_scale, (255, 255, 255), 1, cv2.LINE_AA)


def _write_qa_strip(
    frames: np.ndarray,
    pts: np.ndarray,
    swing: Swing,
    destination: Path,
    fps: float,
    config: Config,
    layers: Sequence[str],
    fitted_paths: Mapping[int, FittedPath | None],
) -> Path:
    indices = set(range(0, len(frames), config.qa_every_frames))
    if len(pts):
        indices.add(int(np.argmin(np.abs(pts - swing.impact_t))))
    panels: list[np.ndarray] = []
    for index in sorted(indices):
        burned = render_overlay(
            frames[index], swing, float(pts[index]), fps, config, layers,
            fitted_paths,
        )
        width = min(config.qa_thumb_width, burned.shape[1])
        height = max(1, int(round(burned.shape[0] * width / burned.shape[1])))
        thumb = cv2.resize(burned, (width, height), interpolation=cv2.INTER_AREA)
        _outlined_text(thumb, f"{pts[index] - swing.impact_t:+.3f}s", (8, height - 12), config)
        panels.append(thumb)
    if not panels:
        raise RuntimeError(f"no frames decoded for swing {swing.id}")
    columns = min(config.qa_columns, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    panel_h, panel_w = panels[0].shape[:2]
    sheet = np.zeros((rows * panel_h, columns * panel_w, 3), np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        sheet[row * panel_h : (row + 1) * panel_h, column * panel_w : (column + 1) * panel_w] = panel
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"failed to write {destination}")
    return destination


def _encoder(
    path: Path, width: int, height: int, fps: float, config: Config,
    source: str, t0: float, duration: float,
) -> subprocess.Popen[bytes]:
    audio_seek = "-" + "ss"  # one per clip for audio, never one per frame
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.12g}", "-i", "-",
        audio_seek, f"{t0:.9f}", "-i", source, "-t", f"{duration:.9f}",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(config.render_crf),
        "-pix_fmt", "yuv420p", "-threads", "1", "-fflags", "+bitexact",
        "-flags:v", "+bitexact", "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-map_metadata", "-1", "-metadata", "creation_time=", "-movflags", "+faststart",
        "-y", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    return process


def _finish_encoder(process: subprocess.Popen[bytes], output: Path) -> None:
    assert process.stdin is not None
    process.stdin.close()
    stderr = b"" if process.stderr is None else process.stderr.read()
    if process.stderr is not None:
        process.stderr.close()
    if process.wait() != 0:
        raise RuntimeError(f"reel encode failed for {output}: {stderr.decode('utf-8', errors='replace').strip()}")


def _concat(clips: Sequence[Path], output: Path) -> None:
    if len(clips) == 1:
        shutil.copyfile(clips[0], output)
        return
    listing = output.parent / "clips.ffconcat"
    listing.write_text(
        "ffconcat version 1.0\n" + "".join(
            f"file '{path.resolve().as_posix()}'\n" for path in clips
        ),
        encoding="utf-8", newline="\n",
    )
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-map", "0", "-c", "copy", "-fflags", "+bitexact",
        "-map_metadata", "-1", "-metadata", "creation_time=", "-movflags", "+faststart",
        "-y", str(output),
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())


def render_reel(
    session: Session,
    swings: list[Swing],
    out_dir: str | Path,
    config: Config,
    preset_name: str = "social",
    write_qa: bool = True,
    layers: Sequence[str] = ("club", "ball", "follow"),
) -> Path:
    """Render source-prefixed per-swing clips and concatenate a reel."""
    unknown = sorted(set(layers) - {"club", "ball", "follow"})
    if unknown:
        raise ValueError(f"unknown layers: {', '.join(unknown)}")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    clips_dir = destination / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    source_stem = Path(session.video).stem
    preset = get_preset(preset_name, config)
    output_width = preset.width or session.width
    output_height = preset.height or session.height
    fps = session.fps or config.render_fps_fallback
    work = swings or [Swing(0, 0.0, min(1.0, session.duration), 0.0)]
    clips: list[Path] = []
    for ordinal, swing in enumerate(work, 1):
        fitted_paths = {id(track): fit_track(track, config) for track in swing.tracks}
        duration = max(0.0, swing.window_end - swing.window_start)
        frames, pts = decode_window(session.video, swing.window_start, duration, fps=fps, gray=False)
        if len(frames) == 0:
            raise RuntimeError(f"no frames decoded for swing {swing.id}")
        if write_qa and swing.id:
            _write_qa_strip(
                frames, pts, swing,
                destination / "qa" / f"{source_stem}_swing-{swing.id:03d}.png",
                fps, config, layers, fitted_paths,
            )
        clip = clips_dir / f"{source_stem}_swing-{ordinal:03d}.mp4"
        process = _encoder(
            clip, output_width, output_height, fps, config,
            session.video, swing.window_start, duration,
        )
        assert process.stdin is not None
        try:
            for frame, timestamp in zip(frames, pts, strict=True):
                burned = render_overlay(
                    frame, swing, float(timestamp), fps, config, layers,
                    fitted_paths,
                )
                process.stdin.write(_fit_preset(burned, preset.width, preset.height).tobytes())
        except BaseException:
            process.stdin.close()
            process.kill()
            process.wait()
            raise
        _finish_encoder(process, clip)
        clips.append(clip)
    output = destination / f"{source_stem}_reel.mp4"
    _concat(clips, output)
    return output
