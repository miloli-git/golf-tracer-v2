# PRD-05 — Ball flight phase (impact → out of frame / landing)

**Type:** port+fix · **Milestone:** M2 · **Effort:** ~3 h · **Owner:** `golftracer/phases/ball.py`, `golftracer/candidates.py`, `golftracer/stabilize.py`, `tests/test_ball.py`

## Goal
Port the calibrated ball tracker with unchanged behaviour: streak candidates, per-shot tee measurement, launch-angle vote, physics gates, apex-split fit, abstention.

## Inputs / Outputs
- In: `Swing`, post-impact frames, config gates.
- Out: `Track` with observations, fitted trail, `abstained: bool` + reason code, measured tee point.

## Requirements
1. Port `tracer/{stabilize,candidates,retrack,fit,features}.py`. Dual-polarity streak candidates; per-shot tee measurement (LESSONS §16); launch-angle voting (§14); gates: net rise, min rise, lateral ≤ 0.35×rise, decaying speed, launch step (§13). No tee-proximity seeding (§15).
2. Abstention is a first-class result; the compositor renders nothing for abstained shots rather than a guess.
3. Gate values are config; defaults are the v1 measured values.
4. `golftracer track --phase ball --debug` writes candidate/vote overlays per frame.
5. Tee estimation is the top-hat present-then-absent method (v1 `tracer/tee_tophat.py`), with the ROI derived from the golfer/mat location, not hard-coded; the v1 bright-diff estimator is dropped (LESSONS §26).
6. Follow-through-shaft rejection: a track whose points lie within the club's post-impact arc corridor (from PRD-06 when available, else a coarse hands-height/shaft-angle heuristic) is rejected. Measured on the second outdoor corpus: launch ≤ −10° tracks ending at hand height are the shaft (LESSONS §26).

## Acceptance
- M2 parity: 33/58 tracked on the v1 corpus with the identical abstention set to v1 `retrack.py`; trail points < 2 px RMS.
- `qa` strip of 5 tracked shots matches v1 tracer_v4.
- Second corpus (golden manifest `secondary[0]`): tee measured on ≥ 90 % of real strikes; no shaft false tracks in the `qa` strip of all tracked shots.

## Non-goals
Recall improvements (ceiling is capture-side, LESSONS §19). Curve/shape measurement (§21; out of scope by V2-SCOPE).

## Dependencies
PRD-01, PRD-02. Ports: `tracer/stabilize.py`, `tracer/candidates.py`, `tracer/retrack.py`, `tracer/fit.py`, `tracer/features.py`.

## Notes
`tracer/recall.py` and the gate1 review packs are NOT ported (experiment tooling, stays local).
