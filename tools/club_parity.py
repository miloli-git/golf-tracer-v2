"""Dump a stage-by-stage v1/v2 club-chain parity run for one or all swings.

All footage, labels, oracle, v1-repository, model, and output locations are
arguments. The tool contains no private paths or data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from golftracer.config import Config
from golftracer.decode import decode_window
from golftracer.label.schema import convert_v1_label_documents
from golftracer.session import Swing
from golftracer.phases.backswing import club_observations
from golftracer.tracking import (
    fit_club_spatial_v1, phase_biases_from_v1_samples,
    retime_club_spatial_v1,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")


def _rms_by_time(actual: Sequence[dict], expected: Sequence[dict]) -> float:
    if not actual or not expected:
        return float("inf")
    actual_t = np.asarray([float(item.get("t_s", item.get("t"))) for item in actual])
    actual_xy = np.asarray([[float(item["x"]), float(item["y"])] for item in actual])
    errors = []
    for item in expected:
        time = float(item.get("t_s", item.get("t")))
        point = actual_xy[int(np.argmin(np.abs(actual_t - time)))]
        errors.append(float(np.linalg.norm(point - (float(item["x"]), float(item["y"])))))
    return float(np.sqrt(np.mean(np.square(errors))))


def _import_v1(v1_repo: Path):
    scripts = v1_repo / "scripts"
    for path in (str(scripts), str(v1_repo)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("club_refit"), importlib.import_module("club_retime")


def _run_v1_refit(v1_repo: Path, raw_arc: Path, labels: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(v1_repo / "scripts" / "club_refit.py"),
        "--arc", str(raw_arc), "--labels", str(labels), "--output", str(output),
        "--report", str(output.with_suffix(".md")),
        "--qa-dir", str(output.parent / "refit-qa"),
    ], cwd=v1_repo, check=True)


def _run_v1_retime(
    module: Any, calibrated: dict, labels: dict, video: Path,
    model: Path | None, swing_ids: Sequence[int], output: Path,
) -> dict:
    module.VIDEO = video
    if model is not None:
        module.MODEL = model
    grouped = {
        swing_id: [item for item in labels["samples"] if int(item["swing_id"]) == swing_id]
        for swing_id in swing_ids
    }
    offsets = {
        swing_id: int(labels.get("decode_offsets_frames", {}).get(str(swing_id), module.MEASURED_DECODE_OFFSETS[swing_id]))
        for swing_id in swing_ids
    }
    applied = bool(labels.get("decode_offsets_applied"))
    result = {"swings": []}
    for swing_id in swing_ids:
        swing = calibrated["swings"][swing_id - 1]
        retimed, _ = module.retime_swing(
            swing, grouped[swing_id], offsets[swing_id], 0 if applied else offsets[swing_id],
        )
        result["swings"].append(retimed)
    _write(output, result)
    return result


def _constraints(rows: Sequence[Any]) -> list[dict]:
    return [
        {
            "frame_index": item.frame_index, "t": item.t, "fit_time": item.fit_time,
            "x": item.x, "y": item.y, "source": item.source,
            "phase": item.calibration_phase, "weight": item.weight,
        }
        for item in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-arc", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="already-merged v1 label document")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--swing", default="all", help="1-based swing id or 'all'")
    parser.add_argument("--pose-model", type=Path)
    parser.add_argument("--reuse-v1-refit", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    v1_repo = Path(manifest["v1_repo"])
    video = Path(manifest["video"])
    raw_document = json.loads(args.raw_arc.read_text(encoding="utf-8"))
    label_document = json.loads(args.labels.read_text(encoding="utf-8"))
    swing_ids = list(range(1, len(raw_document["swings"]) + 1)) if args.swing == "all" else [int(args.swing)]
    args.out.mkdir(parents=True, exist_ok=True)

    v1_refit_path = args.reuse_v1_refit or args.out / "v1-refit.json"
    if args.reuse_v1_refit is None:
        _run_v1_refit(v1_repo, args.raw_arc, args.labels, v1_refit_path)
    v1_refit = json.loads(v1_refit_path.read_text(encoding="utf-8"))
    _, v1_retime_module = _import_v1(v1_repo)
    v1_retimed = _run_v1_retime(
        v1_retime_module, v1_refit, label_document, video,
        args.pose_model, swing_ids, args.out / "v1-retimed.json",
    )

    config = Config()
    biases = phase_biases_from_v1_samples(label_document["samples"])
    smoothing = float(v1_refit.get("calibration", {}).get("label_smoothing_px", config.club_label_smoothing_px))
    reports = []
    for swing_id, v1_final in zip(swing_ids, v1_retimed["swings"], strict=True):
        raw_swing = raw_document["swings"][swing_id - 1]
        spec = manifest["club"]["swings"][swing_id - 1]
        takeaway_t = float(raw_swing["takeaway_t"])
        # The raw v1 tracker carries count-indexed times at full precision. The
        # manifest's three-decimal display values are not fit/retime inputs.
        top_t = float(raw_swing["top_t"])
        impact_t = float(raw_swing["t_impact"])
        converted = convert_v1_label_documents(
            [label_document], swing_id=swing_id,
            window_start=float(spec["window_start"]),
        )
        fit = fit_club_spatial_v1(
            raw_swing["points"], converted.labels,
            takeaway_t=takeaway_t, top_t=top_t, impact_t=impact_t,
            config=config, biases=biases, label_smoothing_px=smoothing,
            fps=float(manifest["fps"]),
        )
        window_start = min(float(item["t_s"]) for item in v1_refit["swings"][swing_id - 1]["points"]) - config.club_render_lead_s
        duration = impact_t + 1.0 / float(manifest["fps"]) - window_start
        frames, _ = decode_window(video, window_start, duration, fps=float(manifest["fps"]))
        v2_points, trusted = retime_club_spatial_v1(
            fit, frames, window_start=window_start, takeaway_t=takeaway_t,
            top_t=top_t, impact_t=impact_t, labels=converted.labels,
            fps=float(manifest["fps"]), config=config,
        )
        v2_rows = [{"frame_index": item.frame_index, "t_s": item.t, "x": item.x, "y": item.y, "source": item.source} for item in v2_points]

        local = args.out / f"swing-{swing_id:03d}"
        _write(local / "raw-tracker.v1.json", raw_swing["points"])
        local_swing = Swing(swing_id, window_start, impact_t + 1 / float(manifest["fps"]), impact_t)
        v2_raw, _ = club_observations(frames, local_swing, config, local_swing.metadata)
        _write(local / "raw-tracker.v2.json", [asdict(item) for item in v2_raw])
        _write(local / "constraints.v2.json", {
            "labels": _constraints(fit.backswing.labels + fit.downswing.labels),
            "accepted_tracker": _constraints(fit.accepted),
            "rejected_tracker": _constraints(fit.rejected),
            "backswing_control_points": _constraints(fit.backswing.constraints),
            "downswing_control_points": _constraints(fit.downswing.constraints),
        })
        _write(local / "refit-records.v1.json", v1_refit["swings"][swing_id - 1]["points"])
        _write(local / "refit-records.v2.json", [*fit.backswing_records, *fit.downswing_records[1:]])
        _write(local / "spline-samples.v2.json", {
            "backswing": fit.backswing.xy_samples,
            "downswing": fit.downswing.xy_samples,
        })
        _write(local / "retime.v1.json", {
            "backswing_curve_xy": v1_final["retiming"]["backswing_curve_xy"],
            "downswing_curve_xy": v1_final["retiming"]["downswing_curve_xy"],
            "assignments": [item.get("provenance", {}).get("arc_length_px") for item in v1_final["points"]],
            "positions": v1_final["points"],
        })
        _write(local / "retime.v2.json", {
            "backswing_curve_xy": trusted["backswing_curve_xy"],
            "downswing_curve_xy": trusted["downswing_curve_xy"],
            "backswing_assignments": trusted["backswing_arc_knots"],
            "downswing_assignments": trusted["downswing_arc_knots"],
            "backswing_emissions": trusted["backswing_emissions"],
            "downswing_emissions": trusted["downswing_emissions"],
            "positions": v2_rows,
        })
        report = {
            "swing": swing_id,
            "refit_rms_px": _rms_by_time([*fit.backswing_records, *fit.downswing_records[1:]], v1_refit["swings"][swing_id - 1]["points"]),
            "final_rms_px": _rms_by_time(v2_rows, v1_final["points"]),
            "accepted_tracker": len(fit.accepted),
            "rejected_tracker": len(fit.rejected),
            "downswing_unconstrained_frames": sum(
                1 for actual, expected in zip(v2_rows, v1_final["points"], strict=True)
                if expected["phase"] == "downswing" and actual["source"] != "labelled"
            ),
        }
        reports.append(report)
        print(f"swing {swing_id}: refit={report['refit_rms_px']:.6f}px final={report['final_rms_px']:.6f}px")
    _write(args.out / "report.json", {"label_smoothing_px": smoothing, "biases": biases, "swings": reports})


if __name__ == "__main__":
    main()
