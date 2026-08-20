"""Measure a clip's audio/video offset from tee-patch departure vs audio onset.

Reads an existing ``<stem>_impacts.json`` (or detects impacts), measures the
offset per impact and prints a table plus the median. Does not change any
default; pass the median to ``--av-offset`` when it disagrees with
``Config.av_offset_s``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from golftracer.avoffset import measure_av_offset, summary_lines
from golftracer.config import Config
from golftracer.impacts import detect_impacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--impacts", type=Path, help="existing <stem>_impacts.json; detected if omitted")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--only", type=int, action="append", help="1-based impact index; repeatable")
    parser.add_argument("--tee-roi", help="v0,v1,u0,u1 pixel ROI for the tee estimator")
    parser.add_argument("--json", type=Path, help="write the full summary here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.load(args.config) if args.config else Config()
    if args.impacts is not None:
        rows = json.loads(args.impacts.read_text(encoding="utf-8"))
    else:
        rows = detect_impacts(str(args.video), config, None)
    if args.only:
        rows = [rows[index - 1] for index in args.only if 0 < index <= len(rows)]
    audio_times = []
    for row in rows:
        if "t_audio" in row:
            audio_times.append(float(row["t_audio"]))
        else:
            # v1 / track rows carry the corrected video time; undo the configured constant
            video_time = next(float(row[key]) for key in ("t_video", "t_impact_s", "time") if key in row)
            audio_times.append(video_time + config.av_offset_s)
    roi = None
    if args.tee_roi:
        roi = tuple(int(value) for value in args.tee_roi.split(","))
        if len(roi) != 4:
            raise SystemExit("--tee-roi needs four integers v0,v1,u0,u1")
    summary = measure_av_offset(str(args.video), audio_times, config, roi=roi)
    print("\n".join(summary_lines(summary)))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
