# PRD-02 — Impact detection (audio) and swing windows

**Type:** port (audio in S1; visual candidate stage in S2a) · **Milestone:** M1 · **Effort:** ~1–2 h · **Owner:** `golftracer/impacts.py`, `tests/test_impacts.py`

## Goal
Find every strike in a range video from the audio click and turn each into a swing window with corrected video time, so phases know where to look.

## Inputs / Outputs
- In: video path, config (A/V offset, min gap between strikes, threshold).
- Out: `impacts.json`: list of `{t_audio, t_video, confidence}`; `Swing` windows `[t_video - pre, t_video + post]` with phase sub-windows (backswing pre ~0.6 s, downswing ~0.35 s, ball/follow-through post ~1.5 s; all from config).

## Requirements
1. Port `tracer/impacts.py` + `scripts/impacts.py` logic; keep the onset detector and dedupe.
2. A/V offset is a config value with `--av-offset` override and `golftracer impacts --calibrate`, which writes a 5-frame strip around the audio time so the user reads the offset once per phone (LESSONS §9).
3. Manual add/remove: `impacts.json` is editable and re-read; CLI accepts `--only k`.
4. Confidence and count summary; no crash on videos with zero detected strikes.

## Acceptance
- On a local range video the count matches a hand count within ±1 and every `t_video` lands on the impact frame ±1 in a burned-in strip.
- Synthetic click-train audio fixture returns exact times.

## S1 finding (2026-08-17)
Global audio onset scanning on a busy range returns ~7x too many onsets (441 vs 58 on the golden video): audio cannot tell the filmed golfer's strikes from neighbours'. v1 solved this with motion/pose gating of candidate windows (`impacts_*.json` carries `motion_score`, `pose_gated`). S1 shipped the audio detector + A/V correction + calibration strip; in golden mode it refines manifest-supplied candidate times. **Outstanding for S2:** port v1's visual candidate stage (swing-window motion score / pose gate) so `golftracer impacts` works unseeded on ordinary videos. Until then, unseeded impact detection is not share-ready.

## S2a completion (2026-08-18)

The visual-first stage is now the default and manifest candidates require an
explicit `--candidates` flag. The full unseeded golden run returns 62 impacts
and matches all seven calibrated club impacts within 0.05 s. MediaPipe remains
optional; when absent, the command runs motion+audio and warns that pose gating
is disabled. The model is downloaded to a user cache on first use.

## Non-goals
Visual impact *timing* (audio remains the timing source once a candidate window is chosen).

## Dependencies
PRD-01. Ports: `tracer/impacts.py`, `scripts/impacts.py`.

## Notes
LESSONS §9, §17 (windows sized to real swing timing, not 1.6 s guesses).
