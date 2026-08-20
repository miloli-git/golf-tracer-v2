# golf-tracer-v2

Reel-making tool for golf swing video: fixed phone camera in, tracer reel out (club arc, ball trail, follow-through burned in). Phase-structured rebuild of a private v1 tracer.

**Agents:** read [AGENTS.md](AGENTS.md) first — repo map, commands, binding invariants, rejected approaches.

- **[docs/PLAN.md](docs/PLAN.md)** — how: workstreams, decisions, milestones, sequencing; PRDs per workstream in `docs/prds/`.
- **[V2-SCOPE.md](V2-SCOPE.md)** — purpose (reel-making only), the repo/local split, four phase modules, one compositor, and the core bet: detector-bootstrapped labelling that drives per-session human cost toward zero.
- **[LESSONS.md](LESSONS.md)** — the constraints this project is built on: everything the private v1 build proved the hard way about calibration, decode determinism, fitting, physics gates, and knowing when to stop.

Data combination (launch-monitor joins, per-shot stats, personal footage and label sets) stays local and never lands in this repo. See V2-SCOPE.md "Two repos, two jobs".

v1 stays the working system until v2 is validated against footage captured under the new capture contract (LESSONS.md §20).

## Quickstart

```powershell
pip install -e .
golftracer reel my.mp4
# Writes my_tracer/my_reel.mp4 next to the source video.

# An explicit output root keeps the clips/ and qa/ layout.
golftracer reel my.mp4 --out out/
# out/my_reel.mp4, out/my_impacts.json, out/my_session.json,
# out/clips/my_swing-NNN.mp4, out/qa/my_swing-NNN.png
```

Impact detection is unseeded by default: low-resolution golfer-ROI motion finds
impulsive swing candidates, an optional pose gate rejects candidates where no
wrist reaches shoulder height, and the source audio pins the final impact time.
The editable result is written to `out/<video-stem>_impacts.json`; `--only`, `--av-offset`,
and `--calibrate` retain their S1 behaviour. An explicit visual-candidate JSON
can be supplied with `--candidates` for parity/debug work.

The audio-to-video offset is a per-recording constant (LESSONS §9). The default
`av_offset_s` is the calibrated value for the reference session; on new footage
pass `--av-offset auto` to measure it from the tee-patch departure frame (rows
then carry `av_offset_applied_s`), or run `py -3 tools/av_offset.py VIDEO
--impacts out/<stem>_impacts.json` to see the per-impact table first.

```powershell
# Optional wrist/shoulder pose gate
pip install -e ".[pose]"

# Optional PRD-09 clubhead detector (base install stays light)
pip install -e ".[detect]"

golftracer impacts my.mp4 --out out/
golftracer render out/my_session.json --out rendered/ --preset social --layers club,ball
```

Correction mode accepts either an existing tracker/session JSON or the trained
detector. Proposals are pre-placed; `a`/Space accepts, arrow keys or `hjkl`
nudge, a click moves and saves, `m` marks a false/missing proposal, and `s`
skips. `f` marks the remaining frames skipped once the club finishes or exits;
`q` saves and pauses for a later resume. Each saved label records
`source: accepted|corrected|human` plus
`delta_px`; a per-run `*.time.jsonl` receipt is written beside the label file.

```powershell
# Tracker/session proposals (use --swing-id when the session has several swings)
golftracer label my.mp4 --phase downswing --impact 12.345 --full `
  --out labels/1.downswing.json --propose out/my_session.json --swing-id 1

# Build private data, train the nano detector, and write held-out task metrics.
golftracer detect train --labels labels/ --golden path/to/v2-golden.yaml

# Detector proposals in the labeller, or as follow-through fit evidence.
golftracer label my.mp4 --phase followthrough --impact 12.345 --full `
  --out labels/1.followthrough.json --propose
golftracer reel my.mp4 --labels labels/ --detector-weights weights/clubhead-yolo26n.pt
```

For a multi-swing dense pass, keep footage paths and label locations in a
private queue JSON (schema 1). The queue caches a detector candidate for every
decoded frame, launches each OpenCV labeller as a detached Windows process,
skips completed phases, and stops on an incomplete phase so the same command
resumes it safely.

```powershell
# One-time proposal cache and a read-only progress check.
py -3 tools/label_queue.py --queue path/to/private-queue.json --prepare
py -3 tools/label_queue.py --queue path/to/private-queue.json --status

# The human runs this command and works through the windows in order.
py -3 tools/label_queue.py --queue path/to/private-queue.json

# Afterwards: leave-one-swing-out trajectory gates, then final all-label weights.
py -3 tools/retrain_eval.py --queue path/to/private-queue.json `
  --golden path/to/v2-golden.yaml --out path/to/retrain-eval
```

The proposal cache uses a labeller-only confidence floor so correction mode
always has a marker. Autonomous detector evidence and its held-out gates retain
the production confidence threshold. `retrain_eval.py` writes `report.json` and
`report.md`, including every human-label residual, percent within 15 px, impact
join, and trajectory verdict for each held-out fold. It then trains the output
weights on every available label; duplicated validation images in that final
training run are trainer plumbing, not held-out evidence.

Detector inference is confined to the configured swing ROI. Training adds
blur, motion-streak and brightness variants, disables horizontal flips, uses
CUDA when available, and writes a JSON report with held-out 15 px accuracy,
correction distance, proposals-only follow-fit residuals, and CPU throughput.

MediaPipe's lite Pose Landmarker model downloads on first use to
`~/.cache/golftracer/pose_landmarker_lite.task`; it is never stored in the
repository. Model URL:
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`.
Without MediaPipe, motion+audio detection still runs and logs that the pose
gate is off.

The compositor fits and samples curves rather than drawing raw observation
polylines. It supports independent `club`, `ball`, and `follow` layers with
configured colour, width, fade length and glow. Presets are `social` (default
9:16), `source` (decoded source geometry), and `qa` (source geometry plus
burned-in frame strips). Per-swing clips retain source audio and are joined into
`<video-stem>_reel.mp4`. Every generated file is source-prefixed, so several
videos can safely share one explicit output root.

The deliberately wider full-resolution defaults use a 3 px club/follow core
with a 5 px dark outline and a 4 px ball core with a 6 px dark outline. V1 used
2 px with no outline for club and 3 + 5 px for ball. All are overridable through
the corresponding `*_width_px` and `*_glow_px` Config fields.
The default palette is a red ball trail, one yellow club arc across backswing
and downswing, and a blue follow-through; the `*_colour_bgr` Config fields can
override any layer.

Private parity data remains external. With a configured golden manifest,
`golftracer render --golden-oracle --out rendered/ --preset source` adapts the
v1 oracle JSON into v2 `Track` objects and renders the validation swings.
