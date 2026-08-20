from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from golftracer.config import Config
from golftracer.decode import probe
from golftracer.render.compositor import (
    PathSample, draw_faded_path, fade_alpha, fit_track, render_reel,
)
from golftracer.render.styles import get_style
from golftracer.session import Observation, Session, Swing, Track


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.mp4"


def _valid_video(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return any(stream.get("codec_type") == "video" for stream in json.loads(result.stdout)["streams"])


def _stream_types(path: Path) -> set[str]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return {stream["codec_type"] for stream in json.loads(result.stdout)["streams"]}


def test_stub_tracks_render_valid_deterministic_mp4(tmp_path: Path) -> None:
    meta = probe(FIXTURE)
    swing = Swing(1, 0.2, 0.9, 0.5)
    session = Session(str(FIXTURE), meta.width, meta.height, meta.fps, meta.duration, meta.rotation, swings=[swing])
    config = Config().with_overrides(qa_every_frames=5)
    first = render_reel(session, [swing], tmp_path / "first", config, "source")
    second = render_reel(session, [swing], tmp_path / "second", config, "source")
    assert _valid_video(first)
    assert _stream_types(first) == {"video", "audio"}
    assert first.read_bytes() == second.read_bytes()
    assert first.name == "synthetic_reel.mp4"
    assert (tmp_path / "first" / "clips" / "synthetic_swing-001.mp4").is_file()
    assert (tmp_path / "first" / "qa" / "synthetic_swing-001.png").is_file()


def test_arc_length_spline_sampling_is_deterministic() -> None:
    track = Track("club", [
        Observation(0, 0.0, 10.0, 30.0),
        Observation(1, 0.1, 20.0, 10.0),
        Observation(2, 0.2, 40.0, 30.0),
    ])
    path = fit_track(track)
    assert path is not None
    first = path.samples_until(0.2, fps=60.0, fade_length_s=1.0)
    second = path.samples_until(0.2, fps=60.0, fade_length_s=1.0)
    assert first == second
    assert (first[0].x, first[0].y) == (10.0, 30.0)
    assert (first[-1].x, first[-1].y) == (40.0, 30.0)


def test_club_arc_is_clipped_to_takeaway_through_impact() -> None:
    track = Track("club", [
        Observation(0, 1.0, 10.0, 30.0),
        Observation(1, 1.1, 20.0, 10.0),
        Observation(2, 1.2, 40.0, 30.0),
    ])
    path = fit_track(track)
    assert path is not None
    assert path.samples_until(0.999, fps=60.0, fade_length_s=10.0) == []
    at_impact = path.samples_until(1.2, fps=60.0, fade_length_s=10.0)
    after_impact = path.samples_until(2.0, fps=60.0, fade_length_s=10.0)
    assert at_impact == after_impact
    assert at_impact[-1].t == pytest.approx(1.2)


def test_fade_alpha_schedule_is_10_to_90_percent() -> None:
    values = [fade_alpha(index, 5) for index in range(5)]
    assert values[0] == 0.10
    assert values[-1] == 0.90
    assert values == sorted(values)
    assert fade_alpha(0, 1) == 0.90


def test_default_strokes_have_wider_dark_outline() -> None:
    config = Config()
    assert (config.club_width_px, config.club_glow_px) == (3, 5)
    assert (config.ball_width_px, config.ball_glow_px) == (4, 6)
    assert (config.follow_width_px, config.follow_glow_px) == (3, 5)

    frame = np.full((50, 50, 3), 200, dtype=np.uint8)
    samples = [
        PathSample(0.0, 10, 25),
        PathSample(0.1, 40, 25),
    ]
    draw_faded_path(frame, samples, get_style("club", config), config)
    assert tuple(frame[22, 25]) != (200, 200, 200)  # five-pixel dark outline


def test_v1_style_preset_keeps_parity_strokes_separate_from_defaults() -> None:
    config = Config.v1_style()
    assert (config.club_width_px, config.club_glow_px) == (2, 0)
    assert (config.ball_width_px, config.ball_glow_px) == (3, 5)
    assert (config.follow_width_px, config.follow_glow_px) == (2, 0)
    assert config.club_colour_bgr == (60, 220, 255)
    assert config.ball_colour_bgr == (0, 210, 255)
    assert config.follow_colour_bgr == config.club_colour_bgr


def test_default_layer_palette_is_red_yellow_blue_and_overridable() -> None:
    config = Config()
    assert get_style("ball", config).colour == (0, 0, 255)
    assert get_style("club", config).colour == (60, 220, 255)
    assert get_style("follow", config).colour == (255, 180, 60)

    custom = config.with_overrides(ball_colour_bgr=(1, 2, 3))
    assert get_style("ball", custom).colour == (1, 2, 3)


def test_gappy_ball_track_uses_spatial_piece() -> None:
    from golftracer.render.compositor import _ball_track_is_gappy
    from golftracer.session import Observation

    def obs(frame: int, x: float, y: float) -> Observation:
        return Observation(frame, frame / 60.0, x, y, source="observed")

    dense = [obs(i, 10.0 * i, 1000.0 - 20.0 * i) for i in range(30)]
    assert _ball_track_is_gappy(dense) is False
    gappy = [obs(1, 664.7, 1346.8), obs(2, 642.6, 1220.1), obs(15, 549.4, 685.7),
             obs(17, 549.4, 685.7), obs(31, 436.7, 84.0)]
    assert _ball_track_is_gappy(gappy) is True
    dwell_only = [obs(1, 10.0, 100.0), obs(2, 10.0, 100.0), obs(3, 20.0, 90.0)]
    assert _ball_track_is_gappy(dwell_only) is True


def test_v1_style_keeps_time_spline_for_gappy_ball_tracks() -> None:
    from golftracer.config import Config
    from golftracer.render.compositor import _fit_spatial_piece, fit_track
    from golftracer.session import Observation, Track

    def obs(frame: int, x: float, y: float) -> Observation:
        return Observation(frame, frame / 60.0, x, y, source="observed")

    gappy = Track("ball", [
        obs(1, 664.7, 1346.8), obs(2, 642.6, 1220.1), obs(3, 624.9, 1115.7),
        obs(15, 549.4, 685.7), obs(17, 549.4, 685.7), obs(31, 436.7, 84.0),
    ], metadata={})
    default_fit = fit_track(gappy, Config())
    assert type(default_fit.pieces[0]) is type(_fit_spatial_piece(gappy.observations))
    v1_fit = fit_track(gappy, Config.v1_style())
    assert type(v1_fit.pieces[0]) is not type(default_fit.pieces[0])


def test_glow_underlay_is_scaled_down() -> None:
    from dataclasses import replace
    from golftracer.render.styles import LayerStyle as LS
    style = LS(colour=(60, 220, 255), width=3, fade_length_s=10.0, glow=9)
    samples = [PathSample(x=20.0 + i * 40.0, y=30.0, t=float(i)) for i in range(4)]

    def glow_pixel(scale: float) -> int:
        config = replace(Config(), render_glow_alpha_scale=scale)
        frame = np.full((60, 200, 3), 255, np.uint8)
        draw_faded_path(frame, samples, style, config)
        return int(frame[34, 80].min())  # inside glow band, outside the core line

    assert Config().render_glow_alpha_scale == 0.40
    assert glow_pixel(0.40) > glow_pixel(1.0) + 60  # scaled glow is visibly lighter
    assert glow_pixel(0.40) >= 150
