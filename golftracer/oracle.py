"""Adapters from external v1 oracle JSON into the public v2 session schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Config
from .decode import probe
from .session import Observation, Session, Swing, Track


def club_tracks_from_data(document: Mapping[str, Any]) -> list[tuple[float, Track]]:
    """Convert v1 ``clubtrack_*_final.json`` data without retaining v1 paths."""
    converted: list[tuple[float, Track]] = []
    for raw_swing in document.get("swings", []):
        observations = [
            Observation(
                frame_index=int(item.get("rel_frame", index)),
                t=float(item.get("t_s", item.get("time"))),
                x=float(item.get("x", item.get("u"))),
                y=float(item.get("y", item.get("v"))),
                confidence=1.0,
                source=str(item.get("source", "oracle")),
            )
            for index, item in enumerate(raw_swing.get("points", []))
        ]
        metadata = {
            "oracle": True,
            "top_t": float(raw_swing["top_t"]) if "top_t" in raw_swing else None,
        }
        retiming = raw_swing.get("retiming", {})
        if retiming.get("backswing_curve_xy") and retiming.get("downswing_curve_xy"):
            back_rows = [item for item in raw_swing.get("points", []) if item.get("phase") == "backswing"]
            down_rows = [back_rows[-1], *[
                item for item in raw_swing.get("points", [])
                if item.get("phase") in {"downswing", "impact"}
            ]]
            back_arc = [float(item.get("provenance", {}).get("arc_length_px", 0.0)) for item in back_rows]
            down_arc = [float(item.get("provenance", {}).get("arc_length_px", 0.0)) for item in down_rows]
            if down_arc:
                down_arc[0] = 0.0
            metadata["trusted_geometry"] = {
                "backswing_curve_xy": retiming["backswing_curve_xy"],
                "downswing_curve_xy": retiming["downswing_curve_xy"],
                "backswing_arc_knots": back_arc,
                "downswing_arc_knots": down_arc,
            }
        converted.append((
            float(raw_swing.get("t_impact", raw_swing.get("impact_t"))),
            Track("club", observations, metadata=metadata),
        ))
    return converted


def ball_tracks_from_data(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[float, Track | None]]:
    """Convert v1 ``retrack_*_v2.json`` rows, preserving abstention as None."""
    converted: list[tuple[float, Track | None]] = []
    for raw_track in rows:
        impact = float(raw_track["t"])
        if not bool(raw_track.get("ok")):
            converted.append((impact, None))
            continue
        observations = [
            Observation(
                frame_index=int(item.get("rel_frame", index)),
                t=float(item.get("t_s", impact + int(item.get("rel_frame", index)) / 60.0)),
                x=float(item["u"]),
                y=float(item["v"]),
                confidence=1.0,
                source=str(item.get("source", "oracle")),
            )
            for index, item in enumerate(raw_track.get("points", []))
        ]
        metadata = {
            "oracle": True,
            "ok": True,
            "apex_frame": raw_track.get("apex_frame"),
            "n_descent": int(raw_track.get("n_descent", 0)),
        }
        converted.append((impact, Track("ball", observations, metadata=metadata)))
    return converted


def load_oracle_session(
    manifest: Mapping[str, Any], config: Config = Config()
) -> Session:
    """Load the manifest's club and ball oracle files into seven v2 swings."""
    club_path = Path(str(manifest["club"]["oracle_arcs"]))
    ball_path = Path(str(manifest["ball"]["oracle_tracks"]))
    club_document = json.loads(club_path.read_text(encoding="utf-8"))
    ball_document = json.loads(ball_path.read_text(encoding="utf-8"))
    clubs = club_tracks_from_data(club_document)
    balls = ball_tracks_from_data(ball_document)
    swings: list[Swing] = []
    for index, (impact, club) in enumerate(clubs, 1):
        matching = [item for timestamp, item in balls if abs(timestamp - impact) <= 0.01]
        tracks = [club]
        if matching and matching[0] is not None:
            tracks.append(matching[0])
        start = min(item.t for item in club.observations) - config.render_lead_s
        swings.append(Swing(
            id=index,
            window_start=max(0.0, start),
            window_end=impact + config.render_post_s,
            impact_t=impact,
            tracks=tracks,
        ))
    meta = probe(str(manifest["video"]))
    return Session(
        video=str(manifest["video"]),
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        duration=meta.duration,
        rotation=meta.rotation,
        swings=swings,
    )
