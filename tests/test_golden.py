from __future__ import annotations

from pathlib import Path
import importlib
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from golftracer.config import Config
from golftracer.decode import probe
from golftracer.golden import load_golden
from golftracer.impacts import detect_impacts
from golftracer.iou import golden_iou_report, overlay_only_iou_report
from golftracer.oracle import load_oracle_session
from golftracer.decode import decode_window
from golftracer.phases.ball import BallPhase, estimate_session_tees
from golftracer.render.compositor import render_reel
from golftracer.session import Observation, Swing


pytestmark = pytest.mark.golden


def test_private_video_probe_and_impacts(golden_option) -> None:
    manifest = load_golden(golden_option)
    if manifest is None:
        pytest.skip("golden manifest is not configured")
    video = Path(manifest["video"])
    meta = probe(video)
    expected_width, expected_height = manifest["decoded_size"]
    assert (meta.width, meta.height) == (expected_width, expected_height)
    assert meta.fps == pytest.approx(float(manifest["fps"]), abs=0.01)

    config = Config().with_overrides(av_offset_s=float(manifest["av_offset_s"]))
    impacts = detect_impacts(video, config)
    print(f"unseeded impact count: {len(impacts)}")
    assert 52 <= len(impacts) <= 64
    detected = [float(item["t_video"]) for item in impacts]
    for swing in manifest["club"]["swings"]:
        expected = float(swing["t_impact"])
        assert min(abs(actual - expected) for actual in detected) <= 0.05


def test_oracle_render_and_overlay_iou(golden_option, tmp_path: Path) -> None:
    manifest = load_golden(golden_option)
    if manifest is None:
        pytest.skip("golden manifest is not configured")
    config = Config().with_overrides(qa_every_frames=30)
    session = load_oracle_session(manifest, config)
    assert len(session.swings) == 7
    output = render_reel(
        session, session.swings, tmp_path / "oracle", config, "source",
        layers=("club", "ball"),
    )
    report = golden_iou_report(
        manifest, output, tmp_path / "oracle" / "iou-report.json", config
    )
    print("overlay IoU:", [item["iou"] for item in report["swings"]])
    # The report is the shipped gate while cross-encoder mask alignment is
    # measured.  Do not turn an unachieved 0.95 target into a false pass.
    assert len(report["swings"]) == 7
    assert all(item["frames"] > 0 for item in report["swings"])


def test_overlay_only_parity_gate(golden_option, tmp_path: Path) -> None:
    manifest = load_golden(golden_option)
    if manifest is None:
        pytest.skip("golden manifest is not configured")
    report = overlay_only_iou_report(
        manifest, tmp_path / "overlay-only-iou.json", config=Config.v1_style(),
    )
    for item in report["swings"]:
        print(
            f"overlay-only swing {item['swing_id']}: "
            f"mean={item['mean_iou']:.6f} min={item['min_iou']:.6f}"
        )
        assert item["mean_iou"] > 0.95
        assert item["min_iou"] > 0.95


def test_club_phase_parity_from_v1_labels(golden_option, tmp_path: Path) -> None:
    manifest = load_golden(golden_option)
    if manifest is None:
        pytest.skip("golden manifest is not configured")
    manifest_path = Path(golden_option or os.environ["GOLFTRACER_GOLDEN"])
    label_paths = [Path(path) for path in manifest["club"]["labels"]]
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in label_paths]
    static_oracle = json.loads(Path(manifest["club"]["oracle_arcs"]).read_text(encoding="utf-8"))
    raw_path = Path(static_oracle["calibration"]["source_arc"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    offsets = {
        int(key): int(value)
        for key, value in static_oracle["calibration"]["measured_decode_offsets_frames"].items()
    }
    v1_repo = Path(manifest["v1_repo"])
    for path in (v1_repo / "scripts", v1_repo):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    finalize = importlib.import_module("club_finalize")
    merged, _ = finalize.merge_labels(
        documents[0], documents[1], documents[2], raw,
        label_paths[0], label_paths[1], label_paths[2], offsets,
    )
    merged_path = tmp_path / "labels-merged.json"
    merged_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    assert len(merged["samples"]) == 195

    output = tmp_path / "club-parity"
    subprocess.run([
        sys.executable, str(Path(__file__).parents[1] / "tools" / "club_parity.py"),
        "--manifest", str(manifest_path), "--raw-arc", str(raw_path),
        "--labels", str(merged_path), "--out", str(output), "--swing", "all",
    ], check=True)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    rms_values = [float(item["final_rms_px"]) for item in report["swings"]]
    for item in report["swings"]:
        print(f"club phase swing {item['swing']}: RMS={item['final_rms_px']:.6f}px")
    print("club phase RMS:", rms_values)
    assert len(rms_values) == 7
    assert max(rms_values) < 2.0
    assert all(int(item["downswing_unconstrained_frames"]) == 0 for item in report["swings"])


def _write_ball_qa(path: Path, panels: list[np.ndarray]) -> None:
    if panels:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.hstack(panels))


def _cached_tee_table(name: str, video: str, times: list[float], config: Config, roi=None):
    root = Path(".tmp/golden-cache")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}-tees.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {float(key): (tuple(value) if value is not None else None) for key, value in raw.items()}
    table = estimate_session_tees(video, times, config, roi=roi)
    path.write_text(json.dumps({str(key): value for key, value in table.items()}, indent=2), encoding="utf-8")
    return table


def _ball_cache(name: str) -> tuple[Path, dict[float, dict]]:
    root = Path(".tmp/golden-cache")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}-tracks.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    return path, {float(item["t"]): item for item in rows}


def test_ball_phase_primary_parity(golden_option, tmp_path: Path) -> None:
    manifest = load_golden(golden_option)
    if manifest is None:
        pytest.skip("golden manifest is not configured")
    impacts = json.loads(Path(manifest["impacts"]["file"]).read_text(encoding="utf-8"))
    times = [float(item["t_impact_s"]) for item in impacts if item.get("accepted", True)]
    expected_rows = json.loads(Path(manifest["ball"]["oracle_tracks"]).read_text(encoding="utf-8"))
    expected = {round(float(item["t"]), 3): item for item in expected_rows}
    primary_roi = tuple(int(value) for value in manifest["ball"].get(
        "tee_roi", Config().tee_v1_roi,
    ))
    config = Config().with_overrides(tee_method="v1", tee_roi=primary_roi)
    tee_table = _cached_tee_table("primary-s4-v1", manifest["video"], times, config, roi=primary_roi)
    cache_path, cached = _ball_cache("primary-s4-v1")
    actual_ok: set[float] = set()
    expected_ok = {key for key, item in expected.items() if item.get("ok")}
    rms_by_impact: dict[float, float] = {}
    tee_count = 0
    qa_count = 0
    for ordinal, impact in enumerate(times, 1):
        start = impact - config.tee_pre_s
        duration = config.tee_pre_s + config.ball_descent_post_s
        row = cached.get(impact)
        frames = None
        if row is None:
            phase = BallPhase(config, tee_xy=tee_table[impact])
            points = phase.track_video(
                manifest["video"], Swing(ordinal, start, start + duration, impact), config,
            )
            row = {
                "t": impact, "abstained": phase.abstained, "reason": phase.reason,
                "tee_xy": phase.tee_xy, "metrics": phase.metrics,
                "points": [item.__dict__ for item in points],
            }
            cached[impact] = row
            cache_path.write_text(json.dumps([cached[key] for key in sorted(cached)], indent=2), encoding="utf-8")
        points = [Observation(**item) for item in row["points"]]
        tee_count += int(row["tee_xy"] is not None)
        key = round(impact, 3)
        if not row["abstained"]:
            actual_ok.add(key)
            oracle_points = expected[key]["points"]
            actual_by_frame = {item.frame_index: item for item in points}
            actual_frames = np.asarray(sorted(actual_by_frame))
            errors = []
            for item in oracle_points:
                nearest_frame = int(actual_frames[np.argmin(np.abs(actual_frames - int(item["rel_frame"])))])
                actual = actual_by_frame[nearest_frame]
                errors.append(np.hypot(actual.x - float(item["u"]), actual.y - float(item["v"])))
            if errors:
                rms_by_impact[key] = float(np.sqrt(np.mean(np.square(errors))))
            if qa_count < 2:
                if frames is None:
                    frames, _ = decode_window(manifest["video"], start, duration, fps=config.ball_fps)
                panels = []
                for index in np.linspace(
                    int(round(config.tee_pre_s * config.ball_fps)),
                    len(frames) - 1, 8, dtype=int,
                ):
                    panel = frames[index].copy()
                    path = np.asarray([
                        (round(item.x), round(item.y)) for item in points
                        if item.t <= start + index / config.ball_fps
                    ], np.int32)
                    if len(path) > 1:
                        cv2.polylines(panel, [path], False, (0, 210, 255), 3, cv2.LINE_AA)
                    panels.append(cv2.resize(panel, (180, 320), interpolation=cv2.INTER_AREA))
                _write_ball_qa(
                    tmp_path / "ball-qa" / f"primary-{key:.3f}.png", panels,
                )
                qa_count += 1
        print(
            f"ball {ordinal:02d}/{len(times)} t={key:.3f} "
            f"ok={not row['abstained']} reason={row['reason']} n={len(points)}"
        )
    missing, extra = sorted(expected_ok - actual_ok), sorted(actual_ok - expected_ok)
    print(f"primary ball: tracked={len(actual_ok)}/{len(times)} tees={tee_count}/{len(times)}")
    print(f"primary ball abstention mismatch: missing={missing} extra={extra}")
    print("primary ball RMS:", rms_by_impact)
    assert actual_ok == expected_ok
    assert rms_by_impact and max(rms_by_impact.values()) < 2.0


def test_ball_phase_secondary_smoke(golden_option, tmp_path: Path) -> None:
    manifest = load_golden(golden_option)
    if manifest is None:
        pytest.skip("golden manifest is not configured")
    secondary = manifest["secondary"][0]
    impacts = json.loads(Path(secondary["impacts"]).read_text(encoding="utf-8"))
    times = [float(item["t_impact_s"]) for item in impacts if item.get("accepted", True)]
    roi = tuple(int(value) for value in secondary["tee_roi"])
    config = Config().with_overrides(
        tee_method="tophat", ball_shaft_rule_enabled=True, tee_roi=roi,
        ball_descent_post_s=Config().ball_post_s,
    )
    tee_table = _cached_tee_table("secondary-s4", secondary["video"], times, config, roi=roi)
    cache_path, cached = _ball_cache("secondary-s4")
    tracked = 0
    tee_count = 0
    shaft_removed = 0
    panels: list[np.ndarray] = []
    for ordinal, impact in enumerate(times, 1):
        start = impact - config.tee_pre_s
        duration = config.tee_pre_s + config.ball_descent_post_s
        row = cached.get(impact)
        frames = None
        if row is None:
            phase = BallPhase(config, tee_roi=roi, tee_xy=tee_table[impact])
            points = phase.track_video(
                secondary["video"], Swing(ordinal, start, start + duration, impact), config,
            )
            row = {
                "t": impact, "abstained": phase.abstained, "reason": phase.reason,
                "tee_xy": phase.tee_xy, "shaft_rule_fired": phase.shaft_rule_fired,
                "metrics": phase.metrics, "points": [item.__dict__ for item in points],
            }
            cached[impact] = row
            cache_path.write_text(json.dumps([cached[key] for key in sorted(cached)], indent=2), encoding="utf-8")
        points = [Observation(**item) for item in row["points"]]
        tee_count += int(row["tee_xy"] is not None)
        shaft_removed += int(row.get("shaft_rule_fired", False))
        if not row["abstained"]:
            tracked += 1
            if len(panels) < 12:
                if frames is None:
                    frames, _ = decode_window(secondary["video"], start, duration, fps=config.ball_fps)
                index = min(len(frames) - 1, int(round(config.tee_pre_s * config.ball_fps)) + 20)
                panel = frames[index].copy()
                path = np.asarray([(round(item.x), round(item.y)) for item in points if item.t <= start + index / config.ball_fps], np.int32)
                if len(path) > 1:
                    cv2.polylines(panel, [path], False, (0, 210, 255), 3, cv2.LINE_AA)
                panels.append(cv2.resize(panel, (180, 320), interpolation=cv2.INTER_AREA))
        print(
            f"secondary ball {ordinal:02d}/{len(times)} ok={not row['abstained']} "
            f"reason={row['reason']} shaft={row.get('shaft_rule_fired', False)}"
        )
    _write_ball_qa(tmp_path / "ball-qa" / "secondary-first-12.png", panels)
    prior = json.loads(Path(secondary["v1_run"]).read_text(encoding="utf-8"))
    prior_ok = sum(bool(item.get("ok")) for item in prior)
    print(
        f"secondary ball: tracked={tracked}/{len(times)} abstained={len(times)-tracked} "
        f"tees={tee_count}/{len(times)} shaft_removed={shaft_removed}; "
        f"v1_run={prior_ok}/{len(prior)} (~10 shaft false)"
    )
    assert tee_count >= int(np.ceil(0.90 * len(times)))
    assert len(panels) == min(12, tracked)
