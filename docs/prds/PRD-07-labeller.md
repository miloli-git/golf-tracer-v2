# PRD-07 — Labeller with correction mode

**Type:** port+extend · **Milestone:** M2 (port), M4 (correction mode) · **Effort:** ~3–4 h · **Owner:** `golftracer/label/{app,schema}.py`, `tests/test_labels.py`

## Goal
The tool that turns human minutes into ground truth. Port v1's click-through labeller, then add proposal-correction: it shows the tracker/detector's guess and the human moves it or accepts it.

## Inputs / Outputs
- In: `Swing`, phase, decoded frames, optional proposals (from tracker or PRD-09 detector).
- Out: `labels/<swing>.<phase>.json` (schema: frame index count-indexed from window start, x, y, `source: human|accepted|corrected`, `convention: head_com`).

## Requirements
1. Port `scripts/label_club.py` (resume-safe, magnifier, phase modes) onto the shared decoder; count-indexed frames only (LESSONS §7).
2. Phase-aware: `--phase backswing|downswing|ball|followthrough`; `--full` labels every frame in the phase (LESSONS §4: whole phases in one sitting).
3. Correction mode: proposals pre-placed; keys: accept, nudge, click-to-move, mark-missing. Records whether each label was accepted or corrected and by how many px (training signal for PRD-09 and its metric).
4. Time-on-task logged per session (minutes): the KPI for the v2 bet.
5. Label collisions: human wins, always.

## Acceptance
- Labels produced by v2 on a v1 swing load into PRD-03/04 fits and reproduce v1 results.
- Correction mode round-trip: propose → correct → save → reload → fit; timing log written.

## Non-goals
Web UI, multi-user.

## Dependencies
PRD-01, PRD-03/04. Ports: `scripts/label_club.py`, `tracer/truth.py`.

## Notes
LESSONS §2, §4, §5, §6.
