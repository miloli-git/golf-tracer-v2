"""Command-line entry point for the M1 skeleton."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .config import Config
from .decode import probe
from .golden import load_golden
from .impacts import (
    detect_impacts, make_swings, read_candidate_times, select_impacts,
    write_calibration_strips, write_impacts,
)
from .oracle import load_oracle_session
from .label.app import run_labeller, selftest as label_selftest
from .render.compositor import render_reel
from .session import Session
from .tracking import track_session


LOG = logging.getLogger("golftracer")


def _common(
    command: argparse.ArgumentParser,
    *,
    video: bool = True,
    out_required: bool = True,
) -> None:
    if video:
        command.add_argument("video", type=Path)
    command.add_argument("--out", type=Path, required=out_required)
    command.add_argument("--config", type=Path)
    command.add_argument("--golden", type=Path)


def _av_offset_arg(value: str) -> float | str:
    if value.strip().lower() == "auto":
        return "auto"
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="golftracer", description=__doc__)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    commands = parser.add_subparsers(dest="command", required=True)

    impacts = commands.add_parser("impacts", help="detect audio impact times")
    _common(impacts)
    impacts.add_argument("--av-offset", type=_av_offset_arg, help="seconds, or 'auto' to measure per clip from tee departure")
    impacts.add_argument("--only")
    impacts.add_argument("--calibrate", type=int, nargs="?", const=3, metavar="N")
    impacts.add_argument("--candidates", type=Path, help="explicit visual candidate JSON")

    track = commands.add_parser("track", help="run one or all tracking phases")
    _common(track)
    track.add_argument("--phase", choices=("backswing", "downswing", "followthrough", "ball", "all"), default="all")
    track.add_argument("--labels", type=Path)
    track.add_argument("--debug", action="store_true")
    track.add_argument("--av-offset", type=_av_offset_arg)
    track.add_argument("--only")
    track.add_argument("--candidates", type=Path)

    label = commands.add_parser("label", help="launch the count-indexed OpenCV labeller")
    label.add_argument("video", type=Path, nargs="?")
    label.add_argument("--out", type=Path)
    label.add_argument("--phase", choices=("backswing", "downswing", "ball", "followthrough"), default="backswing")
    label.add_argument("--impact", type=float)
    label.add_argument("--full", action="store_true")
    label.add_argument("--propose", nargs="?", const="detector", metavar="SOURCE")
    label.add_argument("--weights", type=Path)
    label.add_argument("--swing-id", type=int)
    label.add_argument("--config", type=Path)
    label.add_argument("--selftest", action="store_true")

    detect = commands.add_parser("detect", help="train or inspect the clubhead detector")
    detect_commands = detect.add_subparsers(dest="detect_command", required=True)
    detect_train = detect_commands.add_parser("train", help="build data, train, and report held-out metrics")
    detect_train.add_argument("--labels", type=Path, action="append", required=True)
    detect_train.add_argument("--golden", type=Path)
    detect_train.add_argument("--out", type=Path, default=Path(".tmp/detect"))
    detect_train.add_argument("--holdout")
    detect_train.add_argument("--epochs", type=int, default=25)
    detect_train.add_argument("--batch", type=int, default=16)
    detect_train.add_argument("--model", default="yolo26n.pt")
    detect_train.add_argument("--weights-out", type=Path)
    detect_train.add_argument("--config", type=Path)

    render = commands.add_parser("render", help="render an existing session")
    render.add_argument("session", type=Path, nargs="?")
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--config", type=Path)
    render.add_argument("--golden", type=Path)
    render.add_argument("--golden-oracle", action="store_true")
    render.add_argument("--preset", choices=("social", "source", "qa"), default="social")
    render.add_argument("--layers", default="club,ball,follow")

    reel = commands.add_parser("reel", help="detect impacts and render a reel")
    _common(reel, out_required=False)
    reel.add_argument("--av-offset", type=_av_offset_arg)
    reel.add_argument("--only")
    reel.add_argument("--preset", choices=("social", "source", "qa"), default="social")
    reel.add_argument("--layers", default="club,ball,follow")
    reel.add_argument("--candidates", type=Path, help="explicit visual candidate JSON")
    reel.add_argument("--labels", type=Path)
    reel.add_argument("--detector-weights", type=Path)
    track.add_argument("--detector-weights", type=Path)
    return parser


def _configure_logging(verbose: int) -> None:
    level = logging.DEBUG if verbose > 1 else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def _session(video: Path, impacts: list[dict[str, float]], config: Config) -> Session:
    meta = probe(video)
    swings = make_swings(impacts, config)
    return Session(str(video.resolve()), meta.width, meta.height, meta.fps, meta.duration, meta.rotation, impacts, swings)


def _ensure_layout(out: Path, session: Session) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for swing in session.swings:
        root = out / "swings" / str(swing.id)
        for child in ("tracks", "labels", "audit"):
            (root / child).mkdir(parents=True, exist_ok=True)


def _measure_av_offset(video: Path, impacts: list[dict[str, float]], config: Config) -> tuple[list[dict[str, float]], Config]:
    from .avoffset import apply_measured_av_offset
    updated, measured_config, summary = apply_measured_av_offset(str(video), impacts, config)
    if summary["applied"]:
        LOG.info(
            "av-offset auto: %d/%d impacts measured, departure median %.3f s, applied %.3f s (configured %.3f s)",
            summary["measured"], summary["impacts"], summary["median_offset_s"],
            summary["impact_offset_estimate_s"], config.av_offset_s,
        )
    else:
        LOG.warning("av-offset auto: no impact measured, keeping configured %.3f s", config.av_offset_s)
    return updated, measured_config


def _load_or_detect(
    video: Path,
    out: Path,
    config: Config,
    candidate_times: list[float] | None = None,
    auto_av_offset: bool = False,
) -> tuple[list[dict[str, float]], Config]:
    path = out / f"{video.stem}_impacts.json"
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("impacts.json must contain a list")
        LOG.info("reusing editable %s", path)
        applied = [row.get("av_offset_applied_s") for row in loaded if isinstance(row, dict)]
        if applied and applied[0] is not None:
            config = config.with_overrides(av_offset_s=float(applied[0]))
        return loaded, config
    LOG.info("detecting impacts in %s", video.name)
    found = detect_impacts(video, config, candidate_times)
    if auto_av_offset:
        found, config = _measure_av_offset(video, found, config)
    write_impacts(path, found)
    return found, config


def _layers(value: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise ValueError("--layers must select at least one layer")
    return selected


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    if args.command == "label":
        if args.selftest:
            label_selftest()
            print("SELFTEST OK: v2 label schema round-trip")
            return 0
        if args.video is None or args.out is None or args.impact is None:
            raise ValueError("label requires VIDEO, --out and --impact unless --selftest is used")
        config = Config.load(args.config)
        document = run_labeller(
            args.video, args.out, args.phase, args.impact, args.full, config,
            propose=args.propose, weights=args.weights, swing_id=args.swing_id,
        )
        LOG.info("saved %d labels (%.1f minutes)", len(document.labels), document.time_on_task_s / 60.0)
        return 0
    if args.command == "detect":
        from .detect.train import train_detector
        report = train_detector(
            args.labels, args.out, golden=args.golden, holdout=args.holdout,
            epochs=args.epochs, batch=args.batch, model_name=args.model,
            weights_out=args.weights_out, config=Config.load(args.config),
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "reel" and args.out is None:
        args.out = args.video.with_name(f"{args.video.stem}_tracer")
    config = Config.load(args.config)
    auto_av_offset = getattr(args, "av_offset", None) == "auto"
    if hasattr(args, "av_offset") and not auto_av_offset:
        config = config.with_overrides(av_offset_s=args.av_offset)
    manifest = load_golden(getattr(args, "golden", None))
    candidate_path = getattr(args, "candidates", None)
    candidate_times = read_candidate_times(candidate_path) if candidate_path else None

    if args.command == "impacts":
        impacts = detect_impacts(args.video, config, candidate_times)
        if auto_av_offset:
            impacts, config = _measure_av_offset(args.video, impacts, config)
        impacts = select_impacts(impacts, args.only)
        write_impacts(args.out / f"{args.video.stem}_impacts.json", impacts)
        if args.calibrate:
            write_calibration_strips(args.video, impacts, args.out, config, args.calibrate)
        LOG.info("wrote %d impacts", len(impacts))
        return 0

    if args.command == "render":
        if args.golden_oracle:
            if manifest is None:
                raise ValueError("--golden-oracle requires --golden or GOLFTRACER_GOLDEN")
            session = load_oracle_session(manifest, config)
        else:
            if args.session is None:
                raise ValueError("render requires SESSION unless --golden-oracle is used")
            session = Session.from_json(args.session)
        render_reel(
            session, session.swings, args.out, config, args.preset,
            layers=_layers(args.layers),
        )
        return 0

    impacts, config = _load_or_detect(args.video, args.out, config, candidate_times, auto_av_offset)
    selected = select_impacts(impacts, args.only)
    # Keep the ORIGINAL one-based impact numbers as swing ids so `--only 2` and
    # label files named `2.<phase>.json` refer to the same swing.
    session = _session(args.video, impacts, config)
    if args.only:
        keep = {id(item) for item in selected}
        session.swings = [swing for swing, item in zip(session.swings, impacts) if id(item) in keep]
    _ensure_layout(args.out, session)
    stem = args.video.stem
    write_impacts(args.out / f"{stem}_impacts.json", impacts)
    if args.command in {"track", "reel"}:
        track_session(
            session, config,
            phase=getattr(args, "phase", "all"),
            labels_root=getattr(args, "labels", None),
            debug_dir=(args.out / "debug") if getattr(args, "debug", False) else None,
            detector_weights=getattr(args, "detector_weights", None),
        )
    session.to_json(args.out / f"{stem}_session.json")
    if args.command == "track":
        return 0
    render_reel(
        session, session.swings, args.out, config, args.preset, write_qa=True,
        layers=_layers(args.layers),
    )
    LOG.info("wrote reel with %d swings", len(session.swings))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
