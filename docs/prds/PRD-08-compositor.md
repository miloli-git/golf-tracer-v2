# PRD-08 — Compositor and reel presets

**Type:** port+extend · **Milestone:** M1 (minimal), M2 (full) · **Effort:** ~3 h · **Owner:** `golftracer/render/{compositor,styles,presets}.py`, `tests/test_render.py`

## Goal
One renderer that takes any set of phase tracks and produces the reel: fitted curves sampled per frame, gradient-fade trails, per-layer styles, output presets for social.

## Inputs / Outputs
- In: `Session`, selected swings, phase tracks, style + preset config.
- Out: `reel.mp4` (+ optional per-swing clips, + `qa` frame-strip PNGs).

## Requirements
1. Port `scripts/render_v4.py`: fitted splines sampled per frame, never raw polylines (LESSONS §11); 90→10 % gradient fade; club arc, ball trail and follow-through as separate layers with per-layer style (colour, width, fade length, glow).
2. `--layers club,ball,follow` in any combination; abstained ball → no trail.
3. Presets: `social` (9:16 crop centred on tee/golfer, 1080×1920, per-swing clip + concatenation with cut/fade), `source` (native aspect), `qa` (frame strips every k frames with overlay burned in; the acceptance artefact for every phase PRD, LESSONS §12).
4. Deterministic: same inputs → byte-identical frames (test).
5. No text/stat overlays here; that composition lives in the local pipeline (V2-SCOPE).

## Acceptance
- M1: stub tracks render to a valid mp4.
- M2: overlay-mask diff vs v1 `combined_v5` on the 7 swings, stroke-mask IoU > 0.95.
- `social` preset opens on a phone and looks right; `qa` strips generated for every phase.

## Non-goals
Stats overlays, music, captions.

## Dependencies
PRD-01; full mode needs PRD-03–06. Ports: `scripts/render_v4.py`, `tracer/montage.py` (strip utilities).

## Notes
LESSONS §11, §12.

## S2a implementation note (2026-08-18)

Full mode, presets, layer styles/selection, oracle adaptation, source audio,
per-swing clips, concat, and burned-in QA strips are implemented. Seven oracle
swings render successfully and were visually checked on two QA strips. The
cross-encoder stroke-mask report currently measures 0.191-0.342 IoU, below the
0.95 target; the report ships as a golden test artefact and the target remains
open rather than being weakened.

Orchestrator note: the 0.95 IoU figure was written assuming overlay-only masks. Cross-codec stroke extraction from two encoded reels is the wrong oracle. Parity gate for M2 is redefined as: render v1's oracle tracks through BOTH pipelines onto black frames (v1 `render_v4.draw_faded_path` called directly, v2 compositor with `--background none`), then IoU on those pure overlay masks per sampled frame. Ship in S2b.
