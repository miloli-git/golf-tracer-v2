"""Stage-by-stage v1/v2 ball-chain parity harness for one impact.

Example:
    py -3 tools/ball_parity.py --manifest PATH --impact 76.786 --out DIR

All media, oracle, repository, and output paths are arguments or manifest data.
No private path is embedded in this tracked tool.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from golftracer.candidates import (  # noqa: E402
    CandidateConfig as V2CandidateConfig,
    extract_candidate_observations as v2_candidates,
)
from golftracer.config import Config  # noqa: E402
from golftracer.decode import read_window_pts  # noqa: E402
from golftracer.phases.ball import (  # noqa: E402
    BallPhase,
    _descent_extension as v2_descent,
    _longest_rising as v2_longest_rising,
    estimate_session_tees,
    gate_track as v2_gate,
    launch_vote as v2_vote,
)
from golftracer.session import Swing  # noqa: E402
from golftracer.stabilize import stabilize_frames as v2_stabilize  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--impact", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tee-method", choices=("v1", "tophat"), default="v1")
    parser.add_argument(
        "--tee-roi", help="optional v0,v1,u0,u1 override in display pixels",
    )
    return parser


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _hash_frames(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def _compact_candidates(records: Sequence[dict], impact_index: int) -> list[dict]:
    output = []
    for record in records:
        frame = int(record["frame_index"]) - impact_index
        for item in record.get("observations", record.get("candidates", [])):
            output.append({
                "frame": frame,
                "u": float(item["u"]),
                "v": float(item["v"]),
                "area": float(item.get("area", 0.0)),
                "polarity": item.get("polarity"),
            })
    return output


def _observations(records: Sequence[dict], impact_index: int) -> np.ndarray | None:
    compact = _compact_candidates(records, impact_index)
    if not compact:
        return None
    return np.asarray([
        (item["frame"], item["u"], item["v"], item["area"])
        for item in compact
    ], dtype=float)


def _gate_rows(points: list[dict], tee: Sequence[float], cfg: Any, metrics: dict | None) -> list[dict]:
    if metrics is None:
        return [{"gate": "min_inliers", "passed": False}]
    checks = [
        ("launch_delay", metrics["f"][0] <= cfg.max_launch_delay_frames),
        ("net_rise", metrics["rise"] > 0),
        ("min_rise", metrics["rise"] >= cfg.min_rise_px),
        ("lateral", metrics["lateral"] <= cfg.max_lateral_ratio * metrics["rise"]),
        ("median_step", len(metrics["step"]) >= 3 and float(np.median(metrics["step"])) >= cfg.min_median_step_px),
        ("moving_steps", int((metrics["step"] >= cfg.moving_step_px).sum()) >= cfg.min_moving_steps),
        ("early_gt_late", metrics["early"] > metrics["late"]),
        ("speed_decay_correlation", np.isfinite(metrics["speed_decay_correlation"]) and metrics["speed_decay_correlation"] <= cfg.min_speed_decay_correlation),
        ("local_speed", metrics["local_speed_violations"] <= cfg.max_local_speed_violations),
        ("launch_step", metrics["launch_step"] >= cfg.min_launch_step_px),
        ("global_speed", metrics["speed_violation"] <= cfg.max_speed_violation_frac),
        ("median_area", not (0 < cfg.max_median_area_px < metrics["median_area"])),
        ("max_area", not (0 < cfg.max_blob_area_px < metrics["max_area"])),
    ]
    offset = np.asarray(tee, float) - metrics["origin"]
    checks.append(("tee_not_ahead", float(offset @ metrics["direction"]) <= 0))
    gap = float(abs(
        offset[0] * metrics["direction"][1]
        - offset[1] * metrics["direction"][0]
    ))
    checks.append(("origin_gap", gap <= cfg.origin_tolerance_px))
    return [{"gate": name, "passed": bool(passed)} for name, passed in checks]


def _trace_v1_vote(module: Any, observations: np.ndarray | None, tee: Sequence[float], impact: float, cfg: Any) -> dict:
    trace: dict[str, Any] = {"winner": None, "windows": []}
    if observations is None or len(observations) < cfg.min_inliers:
        trace["reason"] = "too_few_candidates"
        return trace
    values = np.asarray(observations, float)
    if values.shape[1] < 4:
        values = np.column_stack((values, np.zeros(len(values))))
    tee_array = np.asarray(tee, float)
    above = (values[:, 2] < tee_array[1] - cfg.min_above_tee_px) & (values[:, 0] >= 1)
    trace["above"] = above.tolist()
    values = values[above]
    if len(values) < cfg.min_inliers:
        trace["reason"] = "too_few_above_tee"
        return trace
    theta = np.degrees(np.arctan2(values[:, 1] - tee_array[0], tee_array[1] - values[:, 2]))
    physical = np.abs(theta) <= cfg.max_launch_angle_deg
    values, theta = values[physical], theta[physical]
    trace["physical_observations"] = values.tolist()
    trace["physical_angles_deg"] = theta.tolist()
    centres = np.arange(
        -cfg.max_launch_angle_deg,
        cfg.max_launch_angle_deg + 0.5 * cfg.centre_step_deg,
        cfg.centre_step_deg,
    )
    seen: set[tuple[Any, ...]] = set()
    best = None
    for half in cfg.bin_widths_deg:
        for centre in centres:
            selected = np.abs(theta - centre) <= half
            if int(selected.sum()) < cfg.min_inliers:
                continue
            key = (round(half, 2), tuple(np.flatnonzero(selected)))
            if key in seen:
                continue
            seen.add(key)
            candidates, angles = values[selected], theta[selected]
            chosen: dict[int, int] = {}
            for index in range(len(candidates)):
                frame = int(candidates[index, 0])
                if frame not in chosen or abs(angles[index] - centre) < abs(angles[chosen[frame]] - centre):
                    chosen[frame] = index
            chosen_indices = np.asarray([chosen[key] for key in sorted(chosen)], int)
            candidates = candidates[chosen_indices]
            if len(candidates) < cfg.min_inliers:
                continue
            rising = module._longest_rising(candidates[:, 0], candidates[:, 2], cfg.max_frame_gap)
            if len(rising) < cfg.min_inliers:
                continue
            candidates = candidates[rising]
            points = [{
                "rel_frame": int(row[0]), "u": float(row[1]), "v": float(row[2]),
                "t_s": float(impact + row[0] / cfg.fps), "area": float(row[3]),
            } for row in candidates]
            ok, reason, metrics = module.gate_track(points, tee_array, cfg)
            row = {
                "centre_deg": float(centre), "half_width_deg": float(half),
                "selected_indices": np.flatnonzero(selected).tolist(),
                "chosen_frames": candidates[:, 0].astype(int).tolist(),
                "rising_indices": rising.tolist(), "gate_ok": bool(ok),
                "gate_reason": reason,
                "gates": _gate_rows(points, tee, cfg, metrics),
            }
            trace["windows"].append(row)
            if not ok:
                continue
            score = (
                len(points) / 3.0 + metrics["rise"] / 60.0
                - metrics["start_gap"] / 150.0
                - 4.0 * metrics["lateral"] / max(metrics["rise"], 1.0)
            )
            if best is None or score > best[0]:
                best = (score, points, metrics, centre, half, rising)
    if best is None:
        trace["reason"] = "no_accepted_window"
        return trace
    trace["winner"] = {
        "score": float(best[0]), "centre_deg": float(best[3]),
        "half_width_deg": float(best[4]),
        "chosen_frames": [point["rel_frame"] for point in best[1]],
        "points": best[1], "metrics": best[2],
    }
    return trace


class _V2GateConfig:
    """Expose v1 field names for the harness gate table."""

    def __init__(self, config: Config):
        mapping = {
            "max_launch_delay_frames": "ball_max_launch_delay_frames",
            "min_rise_px": "ball_min_rise_px",
            "max_lateral_ratio": "ball_max_lateral_ratio",
            "min_median_step_px": "ball_min_median_step_px",
            "moving_step_px": "ball_moving_step_px",
            "min_moving_steps": "ball_min_moving_steps",
            "min_speed_decay_correlation": "ball_min_speed_decay_correlation",
            "max_local_speed_violations": "ball_max_local_speed_violations",
            "min_launch_step_px": "ball_min_launch_step_px",
            "max_speed_violation_frac": "ball_max_speed_violation_frac",
            "max_median_area_px": "ball_max_median_area_px",
            "max_blob_area_px": "ball_max_blob_area_px",
            "origin_tolerance_px": "ball_origin_tolerance_px",
        }
        for target, source in mapping.items():
            setattr(self, target, getattr(config, source))


def _point_rms(first: Sequence[dict], second: Sequence[dict]) -> float | None:
    by_frame = {int(item["rel_frame"]): item for item in second}
    errors = []
    for item in first:
        other = by_frame.get(int(item["rel_frame"]))
        if other is not None:
            errors.append(np.hypot(float(item["u"]) - float(other["u"]), float(item["v"]) - float(other["v"])))
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


def main() -> int:
    args = _parser().parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    video = Path(manifest["video"])
    v1_repo = Path(manifest["v1_repo"])
    if str(v1_repo) not in sys.path:
        sys.path.insert(0, str(v1_repo))
    v1_ingest = importlib.import_module("tracer.ingest")
    v1_candidates_module = importlib.import_module("tracer.candidates")
    v1_stabilize_module = importlib.import_module("tracer.stabilize")
    v1_retrack = importlib.import_module("tracer.retrack")

    impacts = json.loads(Path(manifest["impacts"]["file"]).read_text(encoding="utf-8"))
    impact_times = [float(item["t_impact_s"]) for item in impacts if item.get("accepted", True)]
    impact = min(impact_times, key=lambda value: abs(value - args.impact))
    if abs(impact - args.impact) > 0.01:
        raise ValueError("requested impact is not in the accepted impact list")
    roi = tuple(int(value) for value in args.tee_roi.split(",")) if args.tee_roi else None
    if roi is None:
        configured = manifest.get("ball", {}).get("tee_roi")
        if configured is not None:
            roi = tuple(int(value) for value in configured)

    args.out.mkdir(parents=True, exist_ok=True)
    v1_cfg = v1_retrack.RetrackConfig()
    if roi is not None:
        v1_cfg = replace(v1_cfg, tee_roi=roi)
    v2_cfg = Config().with_overrides(
        tee_method=args.tee_method,
        tee_roi=roi,
    )
    if args.tee_method == "tophat":
        v1_tophat = importlib.import_module("tracer.tee_tophat")
        original_estimator = v1_retrack.estimate_tee
        v1_retrack.estimate_tee = v1_tophat.estimate_tee_tophat
    else:
        original_estimator = None
    try:
        v1_tee_cache = args.out / "session-tees.v1.json"
        tees_v1 = v1_retrack.session_tee_table(
            video, impact_times, v1_cfg, cache_path=v1_tee_cache,
        )
        v2_tee_cache = args.out / "session-tees.v2.json"
        v2_provenance_cache = args.out / "session-tee-provenance.v2.json"
        if v2_tee_cache.is_file() and v2_provenance_cache.is_file():
            raw = json.loads(v2_tee_cache.read_text(encoding="utf-8"))
            tees_v2 = {float(key): tuple(value) if value else None for key, value in raw.items()}
            raw_provenance = json.loads(v2_provenance_cache.read_text(encoding="utf-8"))
            tee_provenance = {float(key): value for key, value in raw_provenance.items()}
        else:
            tees_v2, tee_provenance = estimate_session_tees(
                str(video), impact_times, v2_cfg, roi=roi,
                return_provenance=True,
            )
            _write(v2_tee_cache, tees_v2)
            _write(v2_provenance_cache, tee_provenance)
    finally:
        if original_estimator is not None:
            v1_retrack.estimate_tee = original_estimator

    tee_v1, tee_v2 = tees_v1[impact], tees_v2[impact]
    _write(args.out / "tee.v1.json", {
        "tee_xy": tee_v1, "method": args.tee_method,
        "source": "v1_session_tee_table", "roi": roi or list(v1_cfg.tee_roi),
    })
    _write(args.out / "tee.v2.json", {
        "tee_xy": tee_v2, **tee_provenance[impact],
    })

    start = impact - v1_cfg.pre_s
    duration = v1_cfg.pre_s + v1_cfg.post_s
    frames_v1, pts_v1 = v1_ingest.read_window(video, start, duration, fps=v1_cfg.fps, gray=True)
    frames_v2, pts_v2 = read_window_pts(video, start, duration, fps=v2_cfg.ball_fps, gray=True)
    decode_v1 = {"shape": list(frames_v1.shape), "pts_s": pts_v1.tolist(), "sha256": _hash_frames(frames_v1)}
    decode_v2 = {"shape": list(frames_v2.shape), "pts_s": pts_v2.tolist(), "sha256": _hash_frames(frames_v2)}
    _write(args.out / "decoded.v1.json", decode_v1)
    _write(args.out / "decoded.v2.json", decode_v2)

    impact_index = int(np.clip(np.searchsorted(pts_v1, impact, side="left"), 1, len(frames_v1) - 1))
    registered_v1, registration_v1 = v1_stabilize_module.stabilize_frames(frames_v1, range(impact_index))
    registered_v2, registration_v2 = v2_stabilize(frames_v1, range(impact_index))
    _write(args.out / "stabilization.v1.json", [item.to_dict() for item in registration_v1])
    _write(args.out / "stabilization.v2.json", [item.to_dict() for item in registration_v2])

    records_v1, _, _ = v1_candidates_module.extract_candidate_observations(
        registered_v1, impact_index, v1_cfg.fps,
        start_time_s=start, timestamps_s=pts_v1, config=v1_retrack.LOOSE,
    )
    records_v2, _, _ = v2_candidates(
        registered_v2, impact_index, v2_cfg.ball_fps,
        start_time_s=start, timestamps_s=pts_v1, config=V2CandidateConfig(),
    )
    compact_v1 = _compact_candidates(records_v1, impact_index)
    compact_v2 = _compact_candidates(records_v2, impact_index)
    _write(args.out / "candidates.v1.json", compact_v1)
    _write(args.out / "candidates.v2.json", compact_v2)
    observations_v1 = _observations(records_v1, impact_index)
    observations_v2 = _observations(records_v2, impact_index)

    vote_v1 = _trace_v1_vote(v1_retrack, observations_v1, tee_v1, impact, v1_cfg)
    vote_v2_debug: dict[str, Any] = {}
    points_v2, reason_v2, metrics_v2 = v2_vote(
        observations_v2, tee_v2, impact, v2_cfg, debug=vote_v2_debug,
    )
    vote_v2_debug["reason"] = reason_v2
    if points_v2 is not None:
        vote_v2_debug["winner"]["points"] = points_v2
        vote_v2_debug["winner"]["metrics"] = metrics_v2
        gate_cfg = _V2GateConfig(v2_cfg)
        vote_v2_debug["winner"]["gates"] = _gate_rows(points_v2, tee_v2, gate_cfg, metrics_v2)
    _write(args.out / "vote.v1.json", vote_v1)
    _write(args.out / "vote.v2.json", vote_v2_debug)

    long_duration = v1_cfg.pre_s + v1_cfg.descent_post_s
    long_frames, long_pts = v1_ingest.read_window(video, start, long_duration, fps=v1_cfg.fps, gray=True)
    long_impact = int(np.clip(np.searchsorted(long_pts, impact, side="left"), 1, len(long_frames) - 1))
    long_registered, _ = v1_stabilize_module.stabilize_frames(long_frames, range(long_impact))
    long_records, _, _ = v1_candidates_module.extract_candidate_observations(
        long_registered, long_impact, v1_cfg.fps,
        start_time_s=start, timestamps_s=long_pts, config=v1_retrack.LOOSE,
    )
    long_observations = _observations(long_records, long_impact)
    ascent_v1 = (vote_v1.get("winner") or {}).get("points")
    descent_v1 = v1_retrack._descent_extension(ascent_v1, long_observations, v1_cfg) if ascent_v1 else None
    descent_v2_result = v2_descent(points_v2, long_observations, v2_cfg) if points_v2 else None
    _write(args.out / "descent.v1.json", descent_v1)
    _write(args.out / "descent.v2.json", descent_v2_result)

    actual_v1 = v1_retrack.retrack_combined(
        video, impact, v1_cfg, tee_xy=tee_v1,
    )
    phase = BallPhase(v2_cfg, tee_xy=tee_v2)
    actual_points_v2 = phase.track_video(
        str(video), Swing(1, start, impact + v2_cfg.ball_descent_post_s, impact), v2_cfg,
    )
    actual_v2 = {
        "t": round(impact, 3), "ok": not phase.abstained,
        "source": "vote" if not phase.abstained else None,
        "reject_reason": phase.reason, "tee_xy_used": phase.tee_xy,
        **phase.metrics,
        "points": [{
            "rel_frame": point.frame_index, "t_s": point.t,
            "u": point.x, "v": point.y,
        } for point in actual_points_v2],
    }
    _write(args.out / "final.v1.json", actual_v1)
    _write(args.out / "final.v2.json", actual_v2)

    checks = [
        ("decode", decode_v1 == decode_v2),
        ("stabilization", np.array_equal(registered_v1, registered_v2) and _safe([item.to_dict() for item in registration_v1]) == _safe([item.to_dict() for item in registration_v2])),
        ("candidates", compact_v1 == compact_v2),
        ("tee", bool(tee_v1 is None and tee_v2 is None) or (tee_v1 is not None and tee_v2 is not None and np.allclose(tee_v1, tee_v2, atol=1e-9))),
        ("vote", (vote_v1.get("winner") or {}).get("chosen_frames") == (vote_v2_debug.get("winner") or {}).get("chosen_frames")),
        ("final_ok", bool(actual_v1.get("ok")) == bool(actual_v2.get("ok"))),
    ]
    first_divergence = next((stage for stage, equal in checks if not equal), None)
    report = {
        "impact": impact,
        "checks": [{"stage": stage, "equal": equal} for stage, equal in checks],
        "first_divergence": first_divergence,
        "v1_ok": bool(actual_v1.get("ok")),
        "v2_ok": bool(actual_v2.get("ok")),
        "final_shared_frame_rms_px": _point_rms(actual_v1.get("points", []), actual_v2.get("points", [])),
    }
    _write(args.out / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if first_divergence is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
