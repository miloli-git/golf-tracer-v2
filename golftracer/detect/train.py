"""Bounded YOLO26n training and task-specific held-out evaluation."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import time
from typing import Iterable

import cv2
import numpy as np

from ..config import Config
from ..decode import decode_window
from ..label.schema import load_labels
from ..phases.followthrough import FollowthroughPhase, detect_finish
from ..session import Observation, Swing
from ..tracking import _club_handoff_impact_t
from .clubhead import _load_model, propose_frames
from .dataset import DatasetBuild, build_dataset


def _phase_metrics(
    rows: list[dict], phase: str | None, radius_px: float = 15.0,
) -> dict[str, object]:
    selected = rows if phase is None else [item for item in rows if phase in item["phases"]]
    errors = [float(item["error_px"]) for item in selected]
    detected = [
        float(item["error_px"]) for item in selected if item["proposal"] is not None
    ]
    return {
        "frames": len(selected),
        "proposals": len(detected),
        "within_15_px_pct": 0.0 if not selected else 100.0 * sum(value <= radius_px for value in errors) / len(selected),
        "median_correction_px": None if not errors else float(np.median(errors)),
        "median_detected_only_px": None if not detected else float(np.median(detected)),
    }


def evaluate_detector(
    weights: str | Path, metadata_path: str | Path, config: Config,
) -> dict[str, object]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    held = [item for item in metadata["frames"] if item["split"] == "val" and item["variant"] == "base"]
    model = _load_model(str(Path(weights).resolve()))
    images = [cv2.imread(item["image"]) for item in held]
    if any(image is None for image in images):
        raise RuntimeError("failed to read held-out detector frames")
    results = model.predict(
        source=images, imgsz=config.detect_input_size_px, conf=config.detect_confidence,
        device="cpu", verbose=False, stream=False,
    )
    rows: list[dict] = []
    for item, image, result in zip(held, images, results, strict=True):
        left, top, right, bottom = [int(value) for value in item["roi"]]
        error = float(math.hypot(right - left, bottom - top))
        proposal = None
        boxes = result.boxes
        if boxes is not None and len(boxes):
            confidence = boxes.conf.detach().cpu().numpy()
            winner = int(np.argmax(confidence))
            box = boxes.xyxy[winner].detach().cpu().numpy()
            px = float(left + 0.5 * (box[0] + box[2]) * (right - left) / image.shape[1])
            py = float(top + 0.5 * (box[1] + box[3]) * (bottom - top) / image.shape[0])
            error = float(math.hypot(px - float(item["x"]), py - float(item["y"])))
            proposal = {"x": px, "y": py, "confidence": float(confidence[winner])}
        rows.append({
            "t": item["t"], "phases": item["phases"], "label_x": item["x"],
            "label_y": item["y"], "proposal": proposal, "error_px": error,
        })

    bench = images or [np.zeros((640, 416, 3), np.uint8)]
    repeated = (bench * (max(32, len(bench)) // len(bench) + 1))[:max(32, len(bench))]
    model.predict(source=repeated[:1], imgsz=config.detect_input_size_px, conf=config.detect_confidence, device="cpu", verbose=False)
    started = time.perf_counter()
    model.predict(source=repeated, imgsz=config.detect_input_size_px, conf=config.detect_confidence, device="cpu", verbose=False, stream=False)
    elapsed = time.perf_counter() - started
    return {
        "holdout": metadata["holdout"],
        "downswing": _phase_metrics(rows, "downswing", config.detect_acceptance_radius_px),
        "followthrough": _phase_metrics(rows, "followthrough", config.detect_acceptance_radius_px),
        "overall": _phase_metrics(rows, None, config.detect_acceptance_radius_px),
        "cpu_inference_fps": len(repeated) / elapsed,
        "per_frame": rows,
    }


def _holdout_documents(label_dirs: Iterable[str | Path], holdout: str):
    session_name, raw_id = holdout.rsplit(":", 1)
    swing_id = int(raw_id)
    for value in label_dirs:
        root = Path(value)
        name = root.parent.name if root.name.lower() == "labels" else root.name
        if name != session_name:
            continue
        docs = {}
        for phase in ("backswing", "downswing", "followthrough"):
            path = root / f"{swing_id}.{phase}.json"
            if path.is_file():
                docs[phase] = load_labels(path)
        return docs
    return {}


def evaluate_followthrough_fit(
    weights: str | Path,
    label_dirs: Iterable[str | Path],
    holdout: str,
    config: Config,
) -> dict[str, object] | None:
    docs = _holdout_documents(label_dirs, holdout)
    follow = docs.get("followthrough")
    down = docs.get("downswing")
    if follow is None or down is None or not follow.labels or not down.labels:
        return None
    impact_t = _club_handoff_impact_t(
        float(follow.window_start), down.labels, follow, 60.0,
    )
    anchor = min(down.labels, key=lambda item: abs(item.t - impact_t))
    impact_xy = (anchor.x, anchor.y)
    start = max(0.0, impact_t - config.backswing_pre_s)
    duration = impact_t - start + config.follow_through_post_s + 1.0 / 60.0
    frames, _ = decode_window(follow.video, start, duration, fps=60.0, gray=False)
    impact_frame = int(round((impact_t - start) * 60.0))
    swing = Swing(1, start, start + duration, impact_t)
    phase = FollowthroughPhase(config, impact_point=impact_xy)
    phase.track(frames, swing, config)  # initialise shared background/pose evidence only
    post = frames[impact_frame:]
    proposals = propose_frames(post, weights=weights, config=config, device="cpu")
    observations = [
        Observation(index, impact_t + index / 60.0, item.x, item.y, item.confidence, "detector")
        for index, item in sorted(proposals.items())
    ]
    finish = detect_finish(
        observations, frame_count=len(post), frame_shape=post.shape[1:3],
        fps=60.0, config=config,
    )
    phase.finish_t = None if finish is None else finish.t
    phase.finish_reason = None if finish is None else finish.reason
    if finish is not None:
        observations = [item for item in observations if item.frame_index <= finish.frame_index]
    spline = phase.fit(observations, [])
    if spline is None:
        return {
            "passed_fit": False, "reason": "insufficient detector proposals",
            "proposals": len(observations),
            "trajectory_acceptance": {
                "passed": False, "reasons": ["insufficient detector proposals"],
            },
        }
    positions = phase.retime(spline, post, fps=60.0, start_t=impact_t)
    by_frame = {item.frame_index: item for item in positions}
    residuals = []
    for label in sorted(follow.labels, key=lambda item: item.frame_index):
        point = by_frame.get(label.frame_index)
        residuals.append({
            "frame_index": label.frame_index,
            "residual_px": None if point is None else float(math.hypot(point.x - label.x, point.y - label.y)),
        })
    join_gap = None if not positions else float(math.hypot(positions[0].x - impact_xy[0], positions[0].y - impact_xy[1]))
    measured = [
        float(item["residual_px"]) for item in residuals
        if item["residual_px"] is not None
    ]
    radius = config.detect_acceptance_radius_px
    within_pct = 100.0 * sum(value <= radius for value in measured) / len(residuals)
    reasons: list[str] = []
    if len(measured) != len(residuals):
        reasons.append(
            f"trajectory covers {len(measured)}/{len(residuals)} labelled frames"
        )
    if join_gap is None or join_gap > config.detect_trajectory_max_join_gap_px:
        reasons.append(
            f"impact join exceeds {config.detect_trajectory_max_join_gap_px:.1f} px"
        )
    if within_pct < config.detect_trajectory_min_within_pct:
        reasons.append(
            f"{within_pct:.1f}% within {radius:.1f} px is below "
            f"{config.detect_trajectory_min_within_pct:.1f}%"
        )
    return {
        "passed_fit": True, "proposals": len(observations), "positions": len(positions),
        "finish_reason": phase.finish_reason, "join_gap_px": join_gap,
        "frame_residuals": residuals,
        "labelled_frames": len(residuals), "evaluated_frames": len(measured),
        "within_15_px_pct": within_pct,
        "median_residual_px": None if not measured else float(np.median(measured)),
        "max_residual_px": None if not measured else float(max(measured)),
        "trajectory_acceptance": {"passed": not reasons, "reasons": reasons},
    }


def train_detector(
    label_dirs: Iterable[str | Path],
    output: str | Path,
    *,
    golden: str | Path | None = None,
    holdout: str | None = None,
    epochs: int = 25,
    batch: int = 16,
    model_name: str = "yolo26n.pt",
    weights_out: str | Path | None = None,
    train_all: bool = False,
    config: Config | None = None,
) -> dict[str, object]:
    cfg = config or Config()
    output = Path(output)
    dataset: DatasetBuild = build_dataset(
        label_dirs, output / "dataset", golden=golden, holdout=holdout,
        train_all=train_all, config=cfg,
    )
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('training requires: pip install -e ".[detect]"') from exc
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_name)
    started = time.perf_counter()
    model.train(
        data=str(dataset.yaml_path), epochs=epochs, imgsz=cfg.detect_input_size_px,
        batch=batch, device=device, workers=0, project=str(output), name="train",
        exist_ok=True, seed=0, deterministic=True, patience=max(5, epochs // 3),
        fliplr=0.0, flipud=0.0, hsv_h=0.015, hsv_s=0.35, hsv_v=0.35,
        degrees=4.0, translate=0.08, scale=0.25, perspective=0.0002,
        mosaic=0.5, mixup=0.05, plots=False, verbose=False,
    )
    wall_s = time.perf_counter() - started
    best = Path(model.trainer.best)
    destination = Path(weights_out) if weights_out is not None else Path("weights") / "clubhead-yolo26n.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    _load_model.cache_clear()
    metrics = evaluate_detector(destination, dataset.metadata_path, cfg)
    follow_fit = evaluate_followthrough_fit(destination, label_dirs, dataset.holdout, cfg)
    report = {
        "model": model_name, "weights": str(destination.resolve()),
        "weights_bytes": destination.stat().st_size, "device": str(device),
        "training_wall_s": wall_s, "epochs_requested": epochs,
        "dataset": {
            "raw_labels": dataset.raw_labels, "unique_frames": dataset.unique_frames,
            "phase_counts": dataset.phase_counts, "swing_counts": dataset.swing_counts,
            "holdout": dataset.holdout, "train_images_with_augmentation": dataset.train_images,
            "val_images": dataset.val_images, "skipped_outside_roi": dataset.skipped_outside_roi,
            "train_all": dataset.train_all,
        },
        "metrics": metrics, "followthrough_proposals_only": follow_fit,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(
        report, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report
