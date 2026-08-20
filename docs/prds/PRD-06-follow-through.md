# PRD-06 — Follow-through phase (impact → finish)

**Type:** new · **Milestone:** M3 · **Effort:** ~4–6 h · **Owner:** `golftracer/phases/followthrough.py`, `tests/test_followthrough.py`

## Goal
The missing phase. Track the clubhead from impact to the finish position so the reel shows the whole swing, not an arc that stops at the ball.

## Inputs / Outputs
- In: `Swing`, post-impact frames (~1.0 s), impact anchor from PRD-04 (position + frame), labels.
- Out: `Track` from impact to finish (rest frame or frame exit), audit.

## Requirements
1. Design: backswing tracker mirrored in time, anchored at the impact point. The first ~4 frames after impact are as fast as the last 4 before it (LESSONS §10), so retiming is DP, not linear.
2. Background is mostly sky: easier detection but new failure modes (shaft against sky, club exiting frame top). Frame exit is a valid finish.
3. Finish detection: velocity floor for N frames or frame exit; the arc stops there, no extrapolation.
4. Ground truth before trust: labeller (PRD-07) supports `--phase followthrough`; label 3–4 swings (~50 clicks) on existing local footage before tuning anything (LESSONS §1, §4).
5. Handoff: impact point shared with downswing so the arc is continuous through impact.

## Acceptance
- On 4 labelled swings: 0 unconstrained frames in the labelled region, tracker error vs labels reported per frame, no visible kink at impact in the `qa` strip.
- Combined reel (PRD-08) covers address → finish + ball trail on those swings.
- M4 new-footage validation with fresh labels: 0 audit failures.

## Non-goals
Body pose, hands, weight shift.

## Dependencies
PRD-03/04 (base interface + impact anchor), PRD-07 (labelling), PRD-08 (visual acceptance).

## Notes
Prototype now on existing footage; the design does not need the capture-contract session.
