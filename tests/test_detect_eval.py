from __future__ import annotations

import json
from pathlib import Path

from golftracer.detect.eval import render_markdown, run_retrain_eval
from golftracer.label.schema import Label, LabelDocument, save_labels


def test_retrain_eval_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    labels = tmp_path / "session"
    save_labels(labels / "1.downswing.json", LabelDocument(
        "video.mp4", 1.0, 60.0, "downswing",
        [Label(0, 1.0, 10.0, 20.0, "downswing")],
    ))
    save_labels(labels / "1.followthrough.json", LabelDocument(
        "video.mp4", 2.0, 60.0, "followthrough",
        [Label(0, 2.0, 10.0, 20.0, "followthrough")],
    ))
    calls = []

    def trainer(label_dirs, output, **kwargs):
        calls.append(kwargs)
        return {
            "weights": str(Path(kwargs["weights_out"]).resolve()),
            "followthrough_proposals_only": {
                "labelled_frames": 1, "proposals": 1,
                "within_15_px_pct": 100.0, "median_residual_px": 2.0,
                "max_residual_px": 2.0, "join_gap_px": 0.0,
                "trajectory_acceptance": {"passed": True, "reasons": []},
                "frame_residuals": [{"frame_index": 0, "residual_px": 2.0}],
            },
        }

    output = tmp_path / "report"
    report = run_retrain_eval(
        [labels], output, golden=None, epochs=1,
        weights_out=tmp_path / "final.pt", trainer=trainer,
    )
    assert report["holdouts"] == ["session:1"]
    assert len(calls) == 2
    assert calls[0].get("train_all", False) is False
    assert calls[1]["train_all"] is True
    loaded = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert loaded["folds"][0]["followthrough_proposals_only"]["frame_residuals"][0]["residual_px"] == 2.0
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "session:1" in markdown
    assert "Trajectory verdict: **1/1 passed**" in markdown
    assert render_markdown(loaded).startswith("# Detector retrain")


def test_retrain_eval_resumes_completed_folds_unless_fresh_or_stale(tmp_path: Path) -> None:
    labels = tmp_path / "session"
    for swing in (1, 2):
        save_labels(labels / f"{swing}.downswing.json", LabelDocument(
            "video.mp4", 1.0, 60.0, "downswing",
            [Label(0, 1.0, 10.0, 20.0, "downswing")],
        ))
        save_labels(labels / f"{swing}.followthrough.json", LabelDocument(
            "video.mp4", 2.0, 60.0, "followthrough",
            [Label(0, 2.0, 10.0, 20.0, "followthrough")],
        ))
    calls: list[str] = []

    def trainer(label_dirs, output, **kwargs):
        calls.append(str(kwargs["holdout"]) + (":all" if kwargs.get("train_all") else ""))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        weights = Path(kwargs["weights_out"])
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"w")
        report = {
            "weights": str(weights.resolve()),
            "followthrough_proposals_only": {
                "labelled_frames": 1, "proposals": 1,
                "within_15_px_pct": 100.0, "median_residual_px": 2.0,
                "max_residual_px": 2.0, "join_gap_px": 0.0,
                "trajectory_acceptance": {"passed": True, "reasons": []},
                "frame_residuals": [],
            },
        }
        (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    output = tmp_path / "report"
    run_retrain_eval([labels], output, golden=None, weights_out=tmp_path / "final.pt", trainer=trainer)
    assert calls == ["session:1", "session:2", "session:1:all"]

    # Simulate an interrupted second fold: remove its weights so only fold 1 is complete.
    (output / "folds" / "session-2" / "weights.pt").unlink()
    calls.clear()
    report = run_retrain_eval([labels], output, golden=None, weights_out=tmp_path / "final.pt", trainer=trainer)
    assert calls == ["session:2", "session:1:all"]
    assert report["resumed_folds"] == ["session:1"]
    assert report["folds"][0]["resumed"] is True

    # --fresh retrains everything.
    calls.clear()
    run_retrain_eval([labels], output, golden=None, weights_out=tmp_path / "final.pt", trainer=trainer, fresh=True)
    assert calls == ["session:1", "session:2", "session:1:all"]

    # A label edited after the fold report invalidates the resume.
    import os, time
    future = time.time() + 60
    os.utime(labels / "1.followthrough.json", (future, future))
    calls.clear()
    report = run_retrain_eval([labels], output, golden=None, weights_out=tmp_path / "final.pt", trainer=trainer)
    assert calls == ["session:1", "session:2", "session:1:all"]
    assert report["resumed_folds"] == []


def test_retrain_eval_reeval_rescores_without_retraining(tmp_path: Path) -> None:
    labels = tmp_path / "session"
    save_labels(labels / "1.downswing.json", LabelDocument(
        "video.mp4", 1.0, 60.0, "downswing", [Label(0, 1.0, 10.0, 20.0, "downswing")],
    ))
    save_labels(labels / "1.followthrough.json", LabelDocument(
        "video.mp4", 2.0, 60.0, "followthrough", [Label(0, 2.0, 10.0, 20.0, "followthrough")],
    ))
    calls: list[str] = []

    def trainer(label_dirs, output, **kwargs):
        calls.append("train")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        weights = Path(kwargs["weights_out"])
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"w")
        report = {
            "weights": str(weights.resolve()),
            "followthrough_proposals_only": {
                "labelled_frames": 1, "proposals": 1, "within_15_px_pct": 0.0,
                "median_residual_px": 300.0, "max_residual_px": 300.0, "join_gap_px": 0.0,
                "trajectory_acceptance": {"passed": False, "reasons": ["old"]},
                "frame_residuals": [],
            },
        }
        (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    def evaluator(weights, label_dirs, holdout, config):
        calls.append("eval")
        return {
            "labelled_frames": 1, "proposals": 1, "within_15_px_pct": 100.0,
            "median_residual_px": 2.0, "max_residual_px": 2.0, "join_gap_px": 0.0,
            "trajectory_acceptance": {"passed": True, "reasons": []}, "frame_residuals": [],
        }

    output = tmp_path / "report"
    run_retrain_eval([labels], output, golden=None, weights_out=tmp_path / "final.pt", trainer=trainer)
    assert calls == ["train", "train"]
    calls.clear()
    report = run_retrain_eval(
        [labels], output, golden=None, weights_out=tmp_path / "final.pt",
        trainer=trainer, evaluator=evaluator, reeval=True,
    )
    assert calls == ["eval"]
    assert report["folds"][0]["reevaluated"] is True
    assert report["folds"][0]["followthrough_proposals_only"]["trajectory_acceptance"]["passed"] is True
    assert "Trajectory verdict: **1/1 passed**" in (output / "report.md").read_text(encoding="utf-8")
