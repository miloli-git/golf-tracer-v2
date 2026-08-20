"""Launch a private detector-assisted label queue or report its progress."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from golftracer.config import Config
from golftracer.label.queue import (
    format_status, load_queue, prepare_proposals, queue_status, run_queue,
    queue_selftest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True, help="private queue JSON")
    parser.add_argument("--status", action="store_true", help="print progress without launching")
    parser.add_argument("--prepare", action="store_true", help="cache detector proposals for every queued frame")
    parser.add_argument("--force", action="store_true", help="replace existing proposal caches")
    parser.add_argument("--selftest", action="store_true", help="open one proposal frame per swing and script controls")
    parser.add_argument("--dry-run", action="store_true", help="print the next labeller command")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = load_queue(args.queue)
    config = Config.load(queue.config_path)
    if args.status:
        print(format_status(queue_status(queue, config)))
        return
    if args.prepare:
        rows = prepare_proposals(queue, config, force=args.force)
        for item in rows:
            print(
                f"{item['key']} {item['phase']}: {item['proposals']}/"
                f"{item['frames']} frames have proposals"
                f"{' (cached)' if item['cached'] else ''}"
            )
        return
    if args.selftest:
        result = queue_selftest(queue, config)
        print(
            f"SELFTEST OK: opened {len(result['swings_opened'])} swings; "
            f"{result['controls']}"
        )
        return
    raise SystemExit(run_queue(queue, config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
