# PRD-09 — Detector bootstrap (proposals for the labeller)

**Type:** new · **Milestone:** M4 · **Effort:** ~6–10 h + training time · **Owner:** `golftracer/detect/{clubhead,dataset,train}.py`, `weights/clubhead-*.pt`, `tests/test_detect.py`

## Goal
Drive per-session human labelling toward zero: a clubhead detector proposes a position on every swing frame; the human corrects instead of clicks. Success is measured in minutes saved and correction distance, not mAP.

## Inputs / Outputs
- In (train): label JSONs (local, never committed) + decoded frames; augmentation.
- In (infer): swing-window frames. Out: per-frame proposals `{x, y, conf}` consumed by PRD-07 and, as secondary evidence, by PRD-03/04/06 fits.
- Weights ship in-repo (small, < ~15 MB) or as a release asset.

## Requirements
1. Single-class clubhead detector, YOLO-family nano or equivalent, CPU-inferable at > 10 fps on 1080p crops; inference restricted to the swing window and a golfer-centred ROI.
2. `golftracer detect train --labels DIR` builds the dataset from label JSON + decoded frames (count-indexed), heavy augmentation (blur, motion streak, brightness; no horizontal flips, handedness matters), trains, writes weights + a report.
3. `golftracer label --propose` uses the detector; corrections write back with `source: corrected` and px delta.
4. Feedback loop: `train` accepts multiple sessions' labels; each session's corrections become next training data.
5. Metrics per session: proposals accepted %, median correction px, human minutes. Target < 30 min/session, stretch < 10 (V2-SCOPE).
6. Ball side unchanged (classical). A TrackNet-style ball model is a follow-on, only if new footage shows the classical ceiling still bites for reels.

## Acceptance
- On a held-out v1 swing: ≥ 70 % of downswing frames within 15 px of the label; labelling that swing in correction mode takes < 1/3 the click-from-zero time.
- On the first capture-contract session: whole session labelled in < 30 min, audit 0 failures after fit.

## Non-goals
Measurement-grade localisation; ball detection model (follow-on).

## Dependencies
PRD-04 (labels + audit), PRD-07 correction mode.

## Notes
LESSONS §6, §12 (verify visually), §24 (report failures plainly; the detector will be thin on ~200 labels, the report must say so).
