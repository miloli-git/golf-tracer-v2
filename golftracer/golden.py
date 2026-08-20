"""Opt-in loader for a private, external golden manifest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def golden_path(path: str | Path | None = None) -> Path | None:
    value = path or os.environ.get("GOLFTRACER_GOLDEN")
    return Path(value).expanduser().resolve() if value else None


def load_golden(path: str | Path | None = None) -> dict[str, Any] | None:
    manifest_path = golden_path(path)
    if manifest_path is None:
        return None
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or int(data.get("schema", 0)) != 1:
        raise ValueError("unsupported golden manifest schema")
    data["_manifest_path"] = str(manifest_path)
    return data


def impact_seed_times(manifest: dict[str, Any] | None) -> list[float] | None:
    """Read optional, external visual-candidate times used by golden parity."""
    if manifest is None:
        return None
    value = manifest.get("impacts", {}).get("file")
    if not value:
        return None
    import json

    rows = json.loads(Path(value).read_text(encoding="utf-8"))
    return [
        float(item["t_impact_s"])
        for item in rows
        if item.get("accepted", True) and "t_impact_s" in item
    ]
