"""Output geometry presets."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config


@dataclass(frozen=True)
class RenderPreset:
    name: str
    width: int | None
    height: int | None


def get_preset(name: str, config: Config) -> RenderPreset:
    if name == "social":
        return RenderPreset("social", config.social_width, config.social_height)
    if name == "source":
        return RenderPreset("source", None, None)
    if name == "qa":
        return RenderPreset("qa", None, None)
    raise ValueError(f"unknown preset: {name}")
