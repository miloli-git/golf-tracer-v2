"""Build a private YOLO dataset from portable labels and a golden manifest."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np
import yaml

from ..config import Config
from ..decode import decode_window
from ..golden import load_golden
from ..label.schema import LabelDocument, load_labels, load_v1_labels
from .clubhead import roi_bounds


@dataclass
class DetectorSample:
    swing_key: str
    video: str
    t: float
    x: float
    y: float
    phases: set[str] = field(default_factory=set)
    source: str = "local"


@dataclass(frozen=True)
class DatasetBuild:
    root: Path
    yaml_path: Path
    metadata_path: Path
    holdout: str
    raw_labels: int
    unique_frames: int
    phase_counts: dict[str, int]
    swing_counts: dict[str, int]
    train_images: int
    val_images: int
    skipped_outside_roi: int
    train_all: bool = False


def _swing_id(path: Path) -> int:
    match = re.match(r"(\d+)\.", path.name)
    if not match:
        raise ValueError(f"label filename must start with a swing id: {path.name}")
    return int(match.group(1))


def _add_document(
    samples: list[DetectorSample], document: LabelDocument,
    swing_key: str, source: str,
) -> int:
    for item in document.labels:
        samples.append(DetectorSample(
            swing_key, document.video, float(item.t), float(item.x), float(item.y),
            {item.phase}, source,
        ))
    return len(document.labels)


def collect_samples(
    label_dirs: Iterable[str | Path],
    *,
    golden: str | Path | None = None,
) -> tuple[list[DetectorSample], dict[str, object]]:
    """Collect local v2 documents first, then dedupe converted v1 labels by frame."""
    raw: list[DetectorSample] = []
    raw_labels = 0
    for value in label_dirs:
        root = Path(value)
        if not root.is_dir():
            raise FileNotFoundError(root)
        session_name = root.parent.name if root.name.lower() == "labels" else root.name
        for path in sorted(root.glob("*.json")):
            try:
                document = load_labels(path)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            raw_labels += _add_document(
                raw, document, f"{session_name}:{_swing_id(path)}", "local",
            )

    manifest = load_golden(golden)
    if manifest is not None:
        merged = manifest.get("club", {}).get("labels_merged")
        paths = [merged] if merged else manifest.get("club", {}).get("labels", [])
        for spec in manifest.get("club", {}).get("swings", []):
            document = load_v1_labels(
                paths, swing_id=int(spec["id"]), video=str(manifest["video"]),
                window_start=float(spec["window_start"]),
            )
            raw_labels += _add_document(
                raw, document, f"golden:{int(spec['id'])}", "golden",
            )

    # Decode identity is video + exact media time. Local labels precede golden
    # labels so local phase additions win when converted golden files overlap.
    by_frame: dict[tuple[str, int], DetectorSample] = {}
    for item in raw:
        identity = (str(Path(item.video).resolve()).lower(), int(round(item.t * 1_000_000)))
        incumbent = by_frame.get(identity)
        if incumbent is None:
            by_frame[identity] = item
        else:
            incumbent.phases.update(item.phases)
    samples = list(by_frame.values())
    phase_counts = Counter(phase for item in samples for phase in item.phases)
    swing_counts = Counter(item.swing_key for item in samples)
    return samples, {
        "raw_labels": raw_labels,
        "unique_frames": len(samples),
        "phase_counts": dict(sorted(phase_counts.items())),
        "swing_counts": dict(sorted(swing_counts.items())),
    }


def _augmentations(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    blur = cv2.GaussianBlur(image, (7, 7), 1.8)
    kernel = np.zeros((13, 13), np.float32)
    np.fill_diagonal(kernel, 1.0 / 13.0)
    streak = cv2.filter2D(image, -1, kernel)
    dark = np.clip(image.astype(np.float32) * 0.68 - 8.0, 0, 255).astype(np.uint8)
    bright = np.clip(image.astype(np.float32) * 1.28 + 5.0, 0, 255).astype(np.uint8)
    return [("base", image), ("blur", blur), ("streak", streak), ("dark", dark), ("bright", bright)]


def _safe_name(sample: DetectorSample) -> str:
    digest = hashlib.sha1(
        f"{sample.video}|{sample.t:.9f}".encode("utf-8")
    ).hexdigest()[:14]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", sample.swing_key) + "-" + digest


def build_dataset(
    label_dirs: Iterable[str | Path],
    output: str | Path,
    *,
    golden: str | Path | None = None,
    holdout: str | None = None,
    train_all: bool = False,
    config: Config | None = None,
) -> DatasetBuild:
    cfg = config or Config()
    samples, counts = collect_samples(label_dirs, golden=golden)
    if not samples:
        raise ValueError("no portable clubhead labels found")
    follow_swings = sorted({
        item.swing_key for item in samples if "followthrough" in item.phases
    })
    held = holdout or (follow_swings[0] if follow_swings else sorted({item.swing_key for item in samples})[0])
    if held not in {item.swing_key for item in samples}:
        raise ValueError(f"unknown holdout swing: {held}")

    root = Path(output)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[DetectorSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.video, sample.swing_key), []).append(sample)

    metadata: list[dict[str, object]] = []
    train_images = val_images = skipped = 0
    for (video, swing_key), group in sorted(grouped.items()):
        fps = 60.0
        start = max(0.0, min(item.t for item in group) - 1.0 / fps)
        end = max(item.t for item in group) + 1.5 / fps
        frames, timestamps = decode_window(video, start, end - start, fps=fps, gray=False)
        if not len(frames):
            raise RuntimeError(f"no frames decoded for detector swing {swing_key}")
        for sample in group:
            index = int(np.argmin(np.abs(timestamps - sample.t)))
            frame = frames[index]
            left, top, right, bottom = roi_bounds(frame, cfg)
            if not (left <= sample.x < right and top <= sample.y < bottom):
                skipped += 1
                continue
            crop = frame[top:bottom, left:right]
            scale = cfg.detect_input_size_px / crop.shape[1]
            resized = cv2.resize(
                crop,
                (cfg.detect_input_size_px, max(1, int(round(crop.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            cx = (sample.x - left) / (right - left)
            cy = (sample.y - top) / (bottom - top)
            bw = min(1.0, cfg.detect_box_size_px / (right - left))
            bh = min(1.0, cfg.detect_box_size_px / (bottom - top))
            stem = _safe_name(sample)
            outputs: list[tuple[str, str, np.ndarray]] = []
            if train_all:
                outputs.extend(("train", suffix, image) for suffix, image in _augmentations(resized))
                if swing_key == held:
                    outputs.append(("val", "base", resized))
            else:
                split = "val" if swing_key == held else "train"
                variants = [("base", resized)] if split == "val" else _augmentations(resized)
                outputs.extend((split, suffix, image) for suffix, image in variants)
            for split, suffix, image in outputs:
                name = f"{stem}-{suffix}"
                image_path = root / "images" / split / f"{name}.jpg"
                label_path = root / "labels" / split / f"{name}.txt"
                if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"failed to write {image_path}")
                label_path.write_text(
                    f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8",
                )
                if split == "train":
                    train_images += 1
                else:
                    val_images += 1
                metadata.append({
                    "image": str(image_path.resolve()), "split": split,
                    "variant": suffix, "swing": swing_key, "video": video,
                    "t": sample.t, "x": sample.x, "y": sample.y,
                    "phases": sorted(sample.phases), "roi": [left, top, right, bottom],
                })
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images/train", "val": "images/val",
        "names": {0: "clubhead"},
    }, sort_keys=False), encoding="utf-8")
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps({
        "holdout": held, "counts": counts, "frames": metadata,
    }, indent=2) + "\n", encoding="utf-8")
    return DatasetBuild(
        root, yaml_path, metadata_path, held,
        int(counts["raw_labels"]), int(counts["unique_frames"]),
        dict(counts["phase_counts"]), dict(counts["swing_counts"]),
        train_images, val_images, skipped, train_all,
    )
