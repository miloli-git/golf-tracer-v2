# PRD-03 — Backswing phase (address → top)

**Type:** port · **Milestone:** M2 · **Effort:** ~2 h · **Owner:** `golftracer/phases/base.py`, `golftracer/phases/backswing.py`, `tests/test_backswing.py`

## Goal
First concrete phase module and the `Phase` interface all others implement: track → label-constrained fit → gates → audit.

## Inputs / Outputs
- In: `Swing`, decoded frames for the backswing sub-window, optional labels for this phase.
- Out: `Track` with raw observations, fitted arc-length spline (head centre-of-mass convention), per-frame retimed positions, `top` cusp frame, audit report.

## Requirements
1. `phases/base.py`: `Phase.track(frames) -> observations`, `Phase.fit(observations, labels) -> spline`, `Phase.retime(spline, frames) -> per-frame arc positions`, `Phase.gates()`, `Phase.audit(...) -> AuditReport`. Labels are hard constraints; detections are secondary evidence with per-phase bias (LESSONS §2).
2. Port the address→top part of `tracer/clubtrack.py`, spatial refit from `scripts/club_refit.py`, DP retiming from `scripts/club_retime.py`, into the interface. The top is preserved as a cusp (LESSONS §10).
3. Interpolated points are flagged, never silently mixed with observations (LESSONS §3).
4. Audit emits per-frame status LABELLED / OBSERVED / INTERPOLATED / UNCONSTRAINED and fails on any unconstrained frame in the certified region.

## Acceptance
- M2 parity: on the 7 v1 test swings with v1 labels loaded by path, spline points differ from v1's refit output by < 2 px RMS.
- `qa` frame strip of the arc matches v1 clubhead_v6 for the backswing portion.

## Non-goals
Body/hands tracking.

## Dependencies
PRD-01, PRD-02. Ports: `tracer/clubtrack.py` (backswing part), `scripts/club_refit.py`, `scripts/club_retime.py`, `tracer/fit.py`.

## Notes
LESSONS §1–5, §10–12.
