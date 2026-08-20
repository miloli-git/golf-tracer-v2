"""Small JSON schema shared by phases and the compositor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


LABELLED = "LABELLED"
OBSERVED = "OBSERVED"
INTERPOLATED = "INTERPOLATED"
UNCONSTRAINED = "UNCONSTRAINED"


@dataclass
class Observation:
    frame_index: int
    t: float
    x: float
    y: float
    confidence: float = 1.0
    source: str = "stub"


@dataclass
class AuditFrame:
    frame_index: int
    t: float
    status: str
    distance_to_label_px: float | None = None


@dataclass
class AuditReport:
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    frames: list[AuditFrame] = field(default_factory=list)


@dataclass
class Track:
    phase: str
    observations: list[Observation] = field(default_factory=list)
    audit: AuditReport = field(default_factory=AuditReport)
    metadata: dict[str, Any] = field(default_factory=dict)
    abstained: bool = False
    reason: str | None = None


@dataclass
class Swing:
    id: int
    window_start: float
    window_end: float
    impact_t: float
    tracks: list[Track] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    video: str
    width: int
    height: int
    fps: float
    duration: float
    rotation: int
    impacts: list[dict[str, float]] = field(default_factory=list)
    swings: list[Swing] = field(default_factory=list)
    schema: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        swings: list[Swing] = []
        for raw_swing in data.get("swings", []):
            tracks: list[Track] = []
            for raw_track in raw_swing.get("tracks", []):
                raw_audit = dict(raw_track.get("audit", {}))
                raw_audit["frames"] = [
                    AuditFrame(**item) for item in raw_audit.get("frames", [])
                ]
                audit = AuditReport(**raw_audit)
                observations = [Observation(**item) for item in raw_track.get("observations", [])]
                tracks.append(Track(
                    phase=raw_track["phase"], observations=observations, audit=audit,
                    metadata=dict(raw_track.get("metadata", {})),
                    abstained=bool(raw_track.get("abstained", False)),
                    reason=raw_track.get("reason"),
                ))
            values = {k: raw_swing[k] for k in ("id", "window_start", "window_end", "impact_t")}
            swings.append(Swing(**values, tracks=tracks, metadata=dict(raw_swing.get("metadata", {}))))
        values = {k: data[k] for k in ("video", "width", "height", "fps", "duration", "rotation")}
        return cls(**values, impacts=list(data.get("impacts", [])), swings=swings, schema=int(data.get("schema", 1)))

    @classmethod
    def from_json(cls, path: str | Path) -> "Session":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
