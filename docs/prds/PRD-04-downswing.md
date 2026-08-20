# PRD-04 — Downswing phase (top → impact)

**Type:** port · **Milestone:** M2 · **Effort:** ~3 h · **Owner:** `golftracer/phases/downswing.py`, `tests/test_downswing.py`

## Goal
The fast, blurred phase, ported with its full calibration chain. Labels matter most here and the detector (PRD-09) pays off here first.

## Inputs / Outputs
- In: `Swing`, downswing sub-window frames anchored at the top cusp (PRD-03) and impact (PRD-02), labels.
- Out: `Track` (as PRD-03) ending at the impact frame, plus delivery-region audit.

## Requirements
1. Port top→impact tracking from `tracer/clubtrack.py`; label-constrained refit; dense per-frame DP retiming (monotone Viterbi over arc × frame). Linear time→arc is forbidden here (LESSONS §10: ~850 px of arc in the last 4 frames).
2. Delivery-phase interpolations are dropped, not trusted (LESSONS §3).
3. Audit: zero unconstrained downswing frames is the pass condition; report distance-to-nearest-label per frame.
4. Anchor handoff: the top cusp is shared with backswing so combined arcs meet at one point; impact point is exported for PRD-06.

## Acceptance
- M2 parity as PRD-03: < 2 px RMS vs v1 refit on the 7 swings; 0 audit failures.
- Combined backswing+downswing `qa` strip matches v1 clubhead_v6.

## Non-goals
Club speed or plane numbers.

## Dependencies
PRD-03. Ports: `tracer/clubtrack.py` (downswing), `scripts/club_refit.py`, `scripts/club_retime.py`, `scripts/club_calibration_report.py`.

## Notes
LESSONS §1–5, §10–12, §17.
