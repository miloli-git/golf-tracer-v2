"""Per-layer compositor styles derived from central configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config


@dataclass(frozen=True)
class LayerStyle:
    colour: tuple[int, int, int]
    width: int
    fade_length_s: float
    glow: int


def get_style(layer: str, config: Config) -> LayerStyle:
    if layer == "club":
        return LayerStyle(
            config.club_colour_bgr, config.club_width_px,
            config.club_fade_length_s, config.club_glow_px,
        )
    if layer == "ball":
        return LayerStyle(
            config.ball_colour_bgr, config.ball_width_px,
            config.ball_fade_length_s, config.ball_glow_px,
        )
    if layer == "follow":
        return LayerStyle(
            config.follow_colour_bgr, config.follow_width_px,
            config.follow_fade_length_s, config.follow_glow_px,
        )
    raise ValueError(f"unknown render layer: {layer}")
