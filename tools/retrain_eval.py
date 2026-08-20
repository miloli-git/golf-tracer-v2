"""Retrain detector folds, evaluate follow trajectories, and train final weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from golftracer.config import Config
from golftracer.detect.eval import run_retrain_eval
from golftracer.label.queue import load_queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True, help="private label queue JSON")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(".tmp/retrain-eval"))
    parser.add_argument("--holdout", action="append", help="session:swing; repeat to restrict folds")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--weights-out", type=Path, default=Path("weights/clubhead-yolo26n.pt"))
    parser.add_argument(
        "--fresh", action="store_true",
        help="retrain every fold even if a complete fold report and weights already exist under --out",
    )
    parser.add_argument(
        "--reeval", action="store_true",
        help="re-score follow-through trajectories from existing fold weights without retraining",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = load_queue(args.queue)
    label_dirs = sorted({swing.labels_dir for swing in queue.swings})
    report = run_retrain_eval(
        label_dirs, args.out, golden=args.golden, holdouts=args.holdout,
        epochs=args.epochs, batch=args.batch, model_name=args.model,
        weights_out=args.weights_out, config=Config.load(queue.config_path),
        fresh=args.fresh, reeval=args.reeval, log=lambda message: print(message, flush=True),
    )
    print(json.dumps({
        "holdouts": report["holdouts"],
        "resumed_folds": report["resumed_folds"],
        "report_json": report["report_json"],
        "report_markdown": report["report_markdown"],
        "final_weights": report["final_model"]["weights"],
    }, indent=2))


if __name__ == "__main__":
    main()
