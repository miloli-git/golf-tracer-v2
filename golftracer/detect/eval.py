"""Leave-one-swing-out detector retraining and trajectory evaluation."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence

from ..config import Config
from ..label.schema import load_labels
from .train import evaluate_followthrough_fit, train_detector


Trainer = Callable[..., dict[str, object]]


def _session_name(root: Path) -> str:
    return root.parent.name if root.name.lower() == "labels" else root.name


def eligible_holdouts(label_dirs: Iterable[str | Path]) -> list[str]:
    holdouts: set[str] = set()
    for value in label_dirs:
        root = Path(value)
        session = _session_name(root)
        for follow_path in root.glob("*.followthrough.json"):
            match = re.match(r"(\d+)\.followthrough\.json$", follow_path.name)
            if match is None:
                continue
            swing_id = int(match.group(1))
            down_path = root / f"{swing_id}.downswing.json"
            if not down_path.is_file():
                continue
            follow = load_labels(follow_path)
            down = load_labels(down_path)
            if follow.labels and down.labels:
                holdouts.add(f"{session}:{swing_id}")
    return sorted(holdouts)


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def _trajectory(fold: dict[str, object]) -> dict[str, object]:
    value = fold.get("followthrough_proposals_only")
    if isinstance(value, dict):
        return value
    return {
        "passed_fit": False,
        "trajectory_acceptance": {
            "passed": False, "reasons": ["no follow-through evaluation"],
        },
    }


def render_markdown(report: dict[str, object]) -> str:
    gates = report["gates"]
    lines = [
        "# Detector retrain and follow-through evaluation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Each evaluation row is from a leave-one-swing-out model. The final model is then trained on all labels.",
        "",
        "| Holdout | Label frames | Proposals | Within 15 px | Median px | Max px | Join px | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for fold in report["folds"]:
        trajectory = _trajectory(fold)
        verdict = trajectory.get("trajectory_acceptance", {})
        within = trajectory.get("within_15_px_pct")
        median = trajectory.get("median_residual_px")
        maximum = trajectory.get("max_residual_px")
        join = trajectory.get("join_gap_px")
        lines.append(
            f"| {fold['holdout']} | {trajectory.get('labelled_frames', 0)} | "
            f"{trajectory.get('proposals', 0)} | "
            f"{'--' if within is None else f'{within:.1f}%'} | "
            f"{'--' if median is None else f'{median:.2f}'} | "
            f"{'--' if maximum is None else f'{maximum:.2f}'} | "
            f"{'--' if join is None else f'{join:.3f}'} | "
            f"{'PASS' if verdict.get('passed') else 'FAIL'} |"
        )
    passed = sum(
        bool(_trajectory(fold).get("trajectory_acceptance", {}).get("passed"))
        for fold in report["folds"]
    )
    lines.extend([
        "",
        f"Trajectory verdict: **{passed}/{len(report['folds'])} passed**.",
        "",
        "Gate: every labelled frame covered, impact join at most "
        f"{gates['max_join_gap_px']:.1f} px, and at least "
        f"{gates['min_within_pct']:.1f}% within {gates['radius_px']:.1f} px.",
        "",
        f"Final all-label weights: `{report['final_model']['weights']}`",
        "",
    ])
    return "\n".join(lines)


def _latest_label_mtime(roots: Sequence[Path]) -> float:
    latest = 0.0
    for root in roots:
        for path in root.glob("*.json"):
            latest = max(latest, path.stat().st_mtime)
    return latest


def completed_fold(fold_root: Path, label_mtime: float) -> dict[str, object] | None:
    """Return a prior fold report if it is complete and newer than every label file."""
    report_path = fold_root / "report.json"
    weights_path = fold_root / "weights.pt"
    if not (report_path.is_file() and weights_path.is_file()):
        return None
    if report_path.stat().st_mtime < label_mtime:
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(report, dict) or "followthrough_proposals_only" not in report:
        return None
    report["weights"] = str(weights_path.resolve())
    report["report"] = str(report_path.resolve())
    report["resumed"] = True
    return report


def run_retrain_eval(
    label_dirs: Sequence[str | Path],
    output: str | Path,
    *,
    golden: str | Path | None,
    holdouts: Sequence[str] | None = None,
    epochs: int = 25,
    batch: int = 16,
    model_name: str = "yolo26n.pt",
    weights_out: str | Path = Path("weights") / "clubhead-yolo26n.pt",
    config: Config | None = None,
    trainer: Trainer = train_detector,
    fresh: bool = False,
    log: Callable[[str], None] | None = None,
    reeval: bool = False,
    evaluator: Callable[..., dict[str, object] | None] = evaluate_followthrough_fit,
) -> dict[str, object]:
    cfg = config or Config()
    roots = [Path(value).resolve() for value in label_dirs]
    selected = list(holdouts or eligible_holdouts(roots))
    if not selected:
        raise ValueError("no swings have both downswing and follow-through labels")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    label_mtime = _latest_label_mtime(roots)
    folds: list[dict[str, object]] = []
    for holdout in selected:
        fold_root = destination / "folds" / _safe_key(holdout)
        fold_weights = fold_root / "weights.pt"
        prior = None if fresh else completed_fold(fold_root, label_mtime)
        if prior is not None:
            if reeval:
                if log is not None:
                    log(f"reeval: fold {holdout} trajectory from existing weights")
                prior["followthrough_proposals_only"] = evaluator(
                    fold_weights, roots, holdout, cfg,
                )
                prior["reevaluated"] = True
                (fold_root / "report.json").write_text(
                    json.dumps(prior, indent=2, default=str) + "\n", encoding="utf-8",
                )
            elif log is not None:
                log(f"resume: fold {holdout} already complete, skipping")
            folds.append({"holdout": holdout, **prior})
            continue
        trained = trainer(
            roots, fold_root, golden=golden, holdout=holdout,
            epochs=epochs, batch=batch, model_name=model_name,
            weights_out=fold_weights, config=cfg,
        )
        folds.append({"holdout": holdout, **trained})
    final_root = destination / "final-all-labels"
    final_report_path = final_root / "report.json"
    final: dict[str, object] | None = None
    if reeval and not fresh and final_report_path.is_file():
        try:
            loaded = json.loads(final_report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, dict) and Path(str(loaded.get("weights", ""))).is_file():
            final = loaded
            if log is not None:
                log("reeval: final all-label model kept from the prior run")
    if final is None:
        final = trainer(
            roots, final_root, golden=golden, holdout=selected[0],
            epochs=epochs, batch=batch, model_name=model_name,
            weights_out=weights_out, train_all=True, config=cfg,
        )
    report: dict[str, object] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": "leave-one-swing-out folds, then final training on all labels",
        "resumed_folds": [fold["holdout"] for fold in folds if fold.get("resumed")],
        "labels": [str(value) for value in roots],
        "golden": None if golden is None else str(Path(golden).resolve()),
        "holdouts": selected,
        "gates": {
            "radius_px": cfg.detect_acceptance_radius_px,
            "min_within_pct": cfg.detect_trajectory_min_within_pct,
            "max_join_gap_px": cfg.detect_trajectory_max_join_gap_px,
        },
        "folds": folds,
        "final_model": final,
    }
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    report["report_json"] = str(json_path.resolve())
    report["report_markdown"] = str(markdown_path.resolve())
    return report
