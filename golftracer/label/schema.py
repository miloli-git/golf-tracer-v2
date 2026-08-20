"""Portable, count-indexed v2 ground-truth label schema and v1 converter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCES = frozenset({"human", "accepted", "corrected"})
CONVENTION = "head_com"


@dataclass(frozen=True)
class Label:
    frame_index: int
    t: float
    x: float
    y: float
    phase: str
    source: str = "human"
    convention: str = CONVENTION
    delta_px: float | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be count-indexed from zero")
        if self.source not in SOURCES:
            raise ValueError(f"invalid label source: {self.source}")
        if self.convention != CONVENTION:
            raise ValueError(f"unsupported label convention: {self.convention}")
        if self.delta_px is not None and self.delta_px < 0:
            raise ValueError("delta_px cannot be negative")


@dataclass
class LabelDocument:
    video: str
    window_start: float
    fps: float
    phase: str
    labels: list[Label] = field(default_factory=list)
    time_on_task_s: float = 0.0
    schema_version: int = 1
    correction_mode: bool = False
    missing_frames: list[int] = field(default_factory=list)
    skipped_frames: list[int] = field(default_factory=list)
    proposal_source: str | None = None

    def merged_labels(self) -> list[Label]:
        """Human wins a collision; otherwise the later record wins."""
        by_identity: dict[tuple[str, int], Label] = {}
        for item in self.labels:
            key = (item.phase, item.frame_index)
            incumbent = by_identity.get(key)
            if incumbent is None or item.source == "human" or incumbent.source != "human":
                by_identity[key] = item
        return [by_identity[key] for key in sorted(by_identity)]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["labels"] = [asdict(item) for item in self.merged_labels()]
        payload["coordinate_convention"] = CONVENTION
        payload["frame_index_basis"] = "0-based decoded frame count from window_start"
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LabelDocument":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported v2 label schema")
        labels = [Label(**item) for item in payload.get("labels", [])]
        return cls(
            video=str(payload.get("video", "")),
            window_start=float(payload.get("window_start", 0.0)),
            fps=float(payload.get("fps", 60.0)),
            phase=str(payload.get("phase", "all")),
            labels=labels,
            time_on_task_s=float(payload.get("time_on_task_s", 0.0)),
            schema_version=1,
            correction_mode=bool(payload.get("correction_mode", False)),
            missing_frames=[int(item) for item in payload.get("missing_frames", [])],
            skipped_frames=[int(item) for item in payload.get("skipped_frames", [])],
            proposal_source=(
                None if payload.get("proposal_source") is None
                else str(payload["proposal_source"])
            ),
        )


def save_labels(path: str | Path, document: LabelDocument) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = document.to_dict()
    payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_labels(path: str | Path) -> LabelDocument:
    return LabelDocument.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _clicked(sample: Mapping[str, Any]) -> tuple[float, float]:
    value = sample["clicked"]
    if isinstance(value, Mapping):
        return float(value["u"]), float(value["v"])
    return float(value[0]), float(value[1])


def _v2_phase(value: str) -> str:
    if value in {"takeaway", "mid-backswing", "top", "backswing"}:
        return "backswing"
    if value in {"downswing", "delivery", "impact"}:
        return "downswing"
    return value


def convert_v1_label_documents(
    documents: Sequence[Mapping[str, Any]],
    *,
    swing_id: int | None = None,
    video: str = "",
    window_start: float | None = None,
    decode_offsets_frames: Mapping[int, int] | None = None,
) -> LabelDocument:
    """Convert/merge label_club.py documents; later docs override, human wins."""
    if not documents:
        raise ValueError("at least one v1 label document is required")
    by_identity: dict[tuple[str, int], Label] = {}
    fps = float(documents[0].get("fps", 60.0))
    selected_starts: list[float] = []
    for document in documents:
        if int(document.get("schema_version", 0)) != 1:
            raise ValueError("unsupported v1 label document")
        legacy = (
            not document.get("decode_offsets_applied")
            and str(document.get("sampler", {}).get("mode", "standard")) == "standard"
        )
        for sample in document.get("samples", []):
            if swing_id is not None and int(sample["swing_id"]) != swing_id:
                continue
            if not sample.get("completed") or sample.get("skipped") or sample.get("clicked") is None:
                continue
            x, y = _clicked(sample)
            frame = int(sample["rel_frame"])
            timestamp = float(sample["frame_time_s"])
            sid = int(sample["swing_id"])
            is_impact = abs(timestamp - float(sample.get("impact_time_s", timestamp + 1.0))) < 1e-7
            if legacy and not is_impact:
                offset = int((decode_offsets_frames or {}).get(sid, 0))
                frame += offset
                timestamp += offset / fps
            phase = _v2_phase(str(sample.get("phase", "backswing")))
            label = Label(
                frame_index=frame,
                t=timestamp,
                x=x,
                y=y,
                phase=phase,
                source="human",
            )
            by_identity[(phase, frame)] = label
            if sample.get("decode_window_start_s") is not None:
                selected_starts.append(float(sample["decode_window_start_s"]))
    start = float(window_start if window_start is not None else min(selected_starts, default=0.0))
    return LabelDocument(
        video=video or str(documents[0].get("video_path", "")),
        window_start=start,
        fps=fps,
        phase="all",
        labels=[by_identity[key] for key in sorted(by_identity)],
    )


def load_v1_labels(paths: Iterable[str | Path], **kwargs: Any) -> LabelDocument:
    documents = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    return convert_v1_label_documents(documents, **kwargs)
