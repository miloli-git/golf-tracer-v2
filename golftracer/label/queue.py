"""Manifest-driven, resume-safe queue for detector-assisted labelling."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Callable, Iterable

from ..config import Config
from ..decode import decode_window, probe
from .app import phase_window
from .app import LabelApp, _append_time_log, load_proposals
from .schema import LabelDocument, load_labels


PHASES = ("backswing", "downswing", "followthrough")


@dataclass(frozen=True)
class QueueSwing:
    key: str
    name: str
    video: Path
    swing_id: int
    impact_t: float
    labels_dir: Path
    phases: tuple[str, ...]
    frame_counts: dict[str, int]
    reason: str = ""

    def label_path(self, phase: str) -> Path:
        return self.labels_dir / f"{self.swing_id}.{phase}.json"


@dataclass(frozen=True)
class LabelQueue:
    manifest: Path
    weights: Path
    swings: tuple[QueueSwing, ...]
    config_path: Path | None = None
    proposals_dir: Path | None = None

    def proposal_path(self, swing: QueueSwing, phase: str) -> Path:
        root = self.proposals_dir or (self.manifest.parent / "proposals")
        safe = swing.key.replace(":", "-").replace("/", "-").replace("\\", "-")
        return root / f"{safe}.{phase}.json"


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_queue(path: str | Path) -> LabelQueue:
    manifest = Path(path).expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("schema", 0)) != 1:
        raise ValueError("unsupported label queue schema")
    base = manifest.parent
    weights = _resolve(base, payload["weights"])
    config_path = (
        None if payload.get("config") is None else _resolve(base, payload["config"])
    )
    proposals_dir = _resolve(base, payload.get("proposals_dir", "proposals"))
    swings: list[QueueSwing] = []
    seen: set[str] = set()
    for raw in payload.get("swings", []):
        key = str(raw["key"])
        if key in seen:
            raise ValueError(f"duplicate queue swing key: {key}")
        seen.add(key)
        phases = tuple(str(value) for value in raw.get("phases", ["followthrough"]))
        unknown = sorted(set(phases) - set(PHASES))
        if unknown:
            raise ValueError(f"unknown queue phases for {key}: {', '.join(unknown)}")
        if "followthrough" not in phases:
            raise ValueError(f"queue swing {key} must include followthrough")
        swings.append(QueueSwing(
            key=key,
            name=str(raw.get("name", key)),
            video=_resolve(base, raw["video"]),
            swing_id=int(raw["swing_id"]),
            impact_t=float(raw["impact_t"]),
            labels_dir=_resolve(base, raw["labels_dir"]),
            phases=phases,
            frame_counts={
                str(phase): int(count)
                for phase, count in raw.get("frame_counts", {}).items()
            },
            reason=str(raw.get("reason", "")),
        ))
    if not swings:
        raise ValueError("label queue contains no swings")
    return LabelQueue(manifest, weights, tuple(swings), config_path, proposals_dir)


def phase_frame_count(swing: QueueSwing, phase: str, config: Config) -> int:
    cached = swing.frame_counts.get(phase)
    if cached is not None:
        if cached < 1:
            raise ValueError(f"invalid {phase} frame count for {swing.key}: {cached}")
        return cached
    start, duration = phase_window(phase, swing.impact_t, config)
    fps = probe(swing.video).fps
    frames, _ = decode_window(swing.video, start, duration, fps=fps)
    return len(frames)


def _document(path: Path) -> LabelDocument | None:
    return load_labels(path) if path.is_file() else None


def _covered_frames(document: LabelDocument | None, total: int) -> set[int]:
    if document is None:
        return set()
    covered = {item.frame_index for item in document.merged_labels()}
    covered.update(document.missing_frames)
    covered.update(document.skipped_frames)
    return {index for index in covered if 0 <= index < total}


def phase_status(
    swing: QueueSwing, phase: str, config: Config,
) -> dict[str, object]:
    total = phase_frame_count(swing, phase, config)
    path = swing.label_path(phase)
    document = _document(path)
    covered = _covered_frames(document, total)
    labels = [] if document is None else document.merged_labels()
    proposal_actions = [
        item for item in labels if item.source in {"accepted", "corrected"}
    ]
    corrections = [
        float(item.delta_px) for item in proposal_actions if item.delta_px is not None
    ]
    accepted = sum(item.source == "accepted" for item in proposal_actions)
    return {
        "phase": phase,
        "path": str(path),
        "frames": total,
        "done": len(covered),
        "remaining": total - len(covered),
        "complete": len(covered) == total,
        "accepted": accepted,
        "proposal_actions": len(proposal_actions),
        "accepted_pct": (
            None if not proposal_actions else 100.0 * accepted / len(proposal_actions)
        ),
        "corrections_px": corrections,
        "minutes": 0.0 if document is None else document.time_on_task_s / 60.0,
    }


def swing_status(swing: QueueSwing, config: Config) -> dict[str, object]:
    phases = [phase_status(swing, phase, config) for phase in swing.phases]
    proposal_actions = sum(int(item["proposal_actions"]) for item in phases)
    accepted = sum(int(item["accepted"]) for item in phases)
    corrections = [
        value for item in phases for value in item["corrections_px"]  # type: ignore[union-attr]
    ]
    return {
        "key": swing.key,
        "name": swing.name,
        "impact_t": swing.impact_t,
        "frames": sum(int(item["frames"]) for item in phases),
        "done": sum(int(item["done"]) for item in phases),
        "remaining": sum(int(item["remaining"]) for item in phases),
        "complete": all(bool(item["complete"]) for item in phases),
        "followthrough_complete": next(
            bool(item["complete"]) for item in phases
            if item["phase"] == "followthrough"
        ),
        "accepted_pct": (
            None if not proposal_actions else 100.0 * accepted / proposal_actions
        ),
        "median_correction_px": (
            None if not corrections else float(statistics.median(corrections))
        ),
        "minutes": sum(float(item["minutes"]) for item in phases),
        "phases": phases,
    }


def queue_status(queue: LabelQueue, config: Config) -> dict[str, object]:
    swings = [swing_status(swing, config) for swing in queue.swings]
    return {
        "swings": swings,
        "frames": sum(int(item["frames"]) for item in swings),
        "done": sum(int(item["done"]) for item in swings),
        "remaining": sum(int(item["remaining"]) for item in swings),
        "minutes": sum(float(item["minutes"]) for item in swings),
        "complete": all(bool(item["complete"]) for item in swings),
    }


def format_status(status: dict[str, object]) -> str:
    lines = [
        "Swing                 frames done  remaining  accepted  median correction  minutes",
        "--------------------  -----------  ---------  --------  -----------------  -------",
    ]
    for item in status["swings"]:  # type: ignore[union-attr]
        accepted = item["accepted_pct"]
        correction = item["median_correction_px"]
        lines.append(
            f"{item['name'][:20]:20}  {item['done']:4}/{item['frames']:<4}  "
            f"{item['remaining']:9}  "
            f"{'--' if accepted is None else f'{accepted:.1f}%':>8}  "
            f"{'--' if correction is None else f'{correction:.2f} px':>17}  "
            f"{item['minutes']:7.1f}"
        )
    lines.append(
        f"TOTAL {status['done']}/{status['frames']} frames; "
        f"{status['remaining']} remaining; {status['minutes']:.1f} minutes"
    )
    return "\n".join(lines)


def label_command(
    queue: LabelQueue, swing: QueueSwing, phase: str,
) -> list[str]:
    proposal_path = queue.proposal_path(swing, phase)
    proposal_source = str(proposal_path) if proposal_path.is_file() else "detector"
    command = [
        sys.executable, "-m", "golftracer.cli", "label", str(swing.video),
        "--phase", phase, "--impact", f"{swing.impact_t:.9f}", "--full",
        "--out", str(swing.label_path(phase)), "--propose", proposal_source,
        "--weights", str(queue.weights), "--swing-id", str(swing.swing_id),
    ]
    if queue.config_path is not None:
        command.extend(["--config", str(queue.config_path)])
    return command


def prepare_proposals(
    queue: LabelQueue, config: Config, *, force: bool = False,
) -> list[dict[str, object]]:
    """Run detector inference over every queued frame and cache proposal JSON."""
    from ..detect.clubhead import propose_frames

    rows: list[dict[str, object]] = []
    for swing in queue.swings:
        fps = probe(swing.video).fps
        for phase in swing.phases:
            destination = queue.proposal_path(swing, phase)
            if destination.is_file() and not force:
                payload = json.loads(destination.read_text(encoding="utf-8"))
                rows.append({
                    "key": swing.key, "phase": phase,
                    "frames": int(payload["frames_evaluated"]),
                    "proposals": len(payload.get("proposals", [])),
                    "path": str(destination), "cached": True,
                })
                continue
            start, duration = phase_window(phase, swing.impact_t, config)
            frames, _ = decode_window(swing.video, start, duration, fps=fps)
            proposal_config = config.with_overrides(
                detect_confidence=config.label_proposal_confidence,
            )
            proposals = propose_frames(
                frames, weights=queue.weights, config=proposal_config, device="cpu",
            )
            payload = {
                "video": str(swing.video), "phase": phase,
                "window_start": start, "fps": fps,
                "frames_evaluated": len(frames),
                "weights": str(queue.weights),
                "confidence_floor": config.label_proposal_confidence,
                "proposals": [
                    {
                        "frame_index": index, "x": item.x, "y": item.y,
                        "confidence": item.confidence,
                    }
                    for index, item in sorted(proposals.items())
                ],
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            rows.append({
                "key": swing.key, "phase": phase, "frames": len(frames),
                "proposals": len(proposals), "path": str(destination), "cached": False,
            })
    return rows


def queue_selftest(queue: LabelQueue, config: Config) -> dict[str, object]:
    """Open one cached-proposal frame per swing and script correction controls."""
    import cv2
    import numpy as np

    opened: list[str] = []
    with tempfile.TemporaryDirectory(prefix="golftracer-queue-selftest-") as directory:
        temporary = Path(directory)
        for swing in queue.swings:
            phase = swing.phases[0]
            proposal_path = queue.proposal_path(swing, phase)
            if not proposal_path.is_file():
                raise FileNotFoundError(
                    f"proposal cache missing for {swing.key} {phase}; run --prepare"
                )
            start, duration = phase_window(phase, swing.impact_t, config)
            fps = probe(swing.video).fps
            frames, timestamps = decode_window(
                swing.video, start, duration, fps=fps,
            )
            proposals = load_proposals(proposal_path, phase, start, fps)
            if 0 not in proposals:
                raise AssertionError(f"{swing.key} {phase} has no frame-zero proposal")
            output = temporary / f"{swing.key.replace(':', '-')}.{phase}.json"
            document = LabelDocument(str(swing.video), start, fps, phase, correction_mode=True)
            app = LabelApp(frames[:1], timestamps[:1], document, output, [0], proposals)
            view = app._view()
            if not np.any(np.all(view == np.asarray([255, 0, 255]), axis=2)):
                raise AssertionError(f"proposal marker is not visible for {swing.key}")
            try:
                cv2.namedWindow(f"golftracer selftest {swing.name}", cv2.WINDOW_AUTOSIZE)
                cv2.imshow(f"golftracer selftest {swing.name}", view)
                cv2.waitKey(30)
                cv2.destroyWindow(f"golftracer selftest {swing.name}")
            except cv2.error as exc:
                raise RuntimeError(f"OpenCV labeller window failed for {swing.key}") from exc
            opened.append(swing.key)

        frames = np.zeros((4, 100, 100, 3), np.uint8)
        timestamps = np.arange(4, dtype=float) / 60.0
        from .app import Proposal
        proposal = Proposal(0, 50.0, 85.0, 1.0)
        output = temporary / "controls.followthrough.json"
        document = LabelDocument("synthetic.mp4", 0.0, 60.0, "followthrough", correction_mode=True)
        controls = LabelApp(frames, timestamps, document, output, range(4), {0: proposal})
        controls._handle_key(ord("l"))
        controls._handle_key(ord("a"))
        controls._handle_key(ord("s"))
        controls._handle_key(ord("m"))
        controls._handle_key(ord("f"))
        _append_time_log(output, document, 1.0)
        resumed = LabelApp(
            frames, timestamps, load_labels(output), output, range(4), {0: proposal},
        )
        if resumed.indices:
            raise AssertionError("scripted queue labeller did not resume complete")
        if not output.with_suffix(output.suffix + ".time.jsonl").is_file():
            raise AssertionError("scripted queue labeller did not write its time log")
    return {"swings_opened": opened, "controls": "accept/nudge/skip/missing/finish/resume/time-log"}


def _launch_detached(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        kwargs: dict[str, object] = {
            "cwd": str(cwd), "stdin": subprocess.DEVNULL,
            "stdout": log, "stderr": subprocess.STDOUT,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:  # pragma: no cover - the production launcher is Windows
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        return process.wait()


def run_queue(
    queue: LabelQueue,
    config: Config,
    *,
    launcher: Callable[[list[str], Path, Path], int] = _launch_detached,
    dry_run: bool = False,
) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    for number, swing in enumerate(queue.swings, 1):
        status = swing_status(swing, config)
        if status["complete"]:
            print(f"[{number}/{len(queue.swings)}] {swing.name}: complete, skipping")
            continue
        print(
            f"[{number}/{len(queue.swings)}] {swing.name}: "
            f"{status['done']}/{status['frames']} frames, "
            f"{status['minutes']:.1f} minutes so far"
        )
        for phase in swing.phases:
            phase_progress = phase_status(swing, phase, config)
            if phase_progress["complete"]:
                continue
            command = label_command(queue, swing, phase)
            print(
                f"  launching {phase}: {phase_progress['done']}/"
                f"{phase_progress['frames']} frames complete"
            )
            if dry_run:
                print("  " + subprocess.list2cmdline(command))
                return 0
            log_path = swing.label_path(phase).with_suffix(f".{phase}.launcher.log")
            exit_code = launcher(command, log_path, repo_root)
            if exit_code:
                print(f"  labeller failed with exit {exit_code}; see {log_path}")
                return exit_code
            after = phase_status(swing, phase, config)
            if not after["complete"]:
                print(
                    f"  paused at {after['done']}/{after['frames']} frames; "
                    "run the same queue command to resume"
                )
                return 0
    final = queue_status(queue, config)
    print(format_status(final))
    return 0
