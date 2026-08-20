# Agent guide

This file is the entry point for AI agents working in this repo. Read it before touching code. `CLAUDE.md` points here; humans should read it too.

## What this is

`golftracer` turns a fixed-phone-camera golf video into a tracer reel: unseeded impact detection → per-swing club/ball/follow-through tracking → deterministic overlay compositing. Input is a video path; output is `<stem>_reel.mp4` plus per-stage JSON artifacts. Nothing here reads personal data by path; inputs are always arguments.

## Repo map

| Path | What it is |
|---|---|
| `golftracer/cli.py` | CLI entry (`golftracer impacts\|track\|label\|detect\|render\|reel`) |
| `golftracer/impacts.py` | Unseeded impact detection: motion → optional pose gate → audio refine |
| `golftracer/avoffset.py`, `tools/av_offset.py` | Per-recording audio-to-video offset (`--av-offset auto`) |
| `golftracer/phases/` | Phase trackers: `backswing.py`, `downswing.py`, `ball.py`, `followthrough.py`; shared fitting in `base.py` |
| `golftracer/tracking.py`, `fit.py` | Label-constrained curve fitting; labels are hard constraints, detections secondary |
| `golftracer/render/` | Compositor (`compositor.py`), presets, layer styles; all knobs in `config.py` |
| `golftracer/detect/` | Clubhead detector: dataset build, train, eval (optional `[detect]` extra) |
| `golftracer/label/` | Correction-mode labeller UI and label schema |
| `golftracer/golden.py`, `tests/test_golden.py` | Parity gates against the private v1 oracle (skip without manifest) |
| `weights/clubhead-yolo26n.pt` | Current all-label clubhead detector weights |
| `tests/` | Unit suite incl. `test_hygiene.py` (private-reference scan) |
| `examples/clip.mp4` | Synthetic CI smoke input (plumbing only, not tracking quality) |
| `LESSONS.md` | Numbered build constraints. **Binding.** Re-deriving or contradicting one is a regression |
| `V2-SCOPE.md` | Repo/local data split ("Two repos, two jobs") |
| `docs/PLAN.md`, `docs/prds/` | Workstreams, milestones, one PRD per workstream |

## Commands

```
pip install -e ".[dev]"          # base; [pose] adds MediaPipe gate, [detect] adds torch/ultralytics
pytest -q                        # must be green before any commit; includes hygiene
golftracer reel VIDEO --out out/ # end-to-end
golftracer reel examples/clip.mp4 --out out/   # the CI smoke, runs anywhere
GOLFTRACER_GOLDEN=<manifest.yaml> pytest tests/test_golden.py   # parity vs v1 oracle (private manifest, ~20 min)
```

## Invariants — violating any of these is a bug, not a style choice

1. **Determinism.** Same input, same output. No wall-clock, no unseeded RNG, no per-frame ffmpeg seeks (LESSONS §7: one sequential decode per window). A single `"-ss"` exists in `decode.py`; the hygiene-adjacent test pins it to exactly one.
2. **No personal data.** Footage, label sets, launch-monitor exports, trained-run artifacts, private paths/names never land in the tree, in comments, in issues, or in test fixtures. `tests/test_hygiene.py` enforces a forbidden-term list — extend it when you scrub something new, never weaken it. Fixtures are synthetic (`tests/fixtures/make_fixture.py`) or consented.
3. **v1 parity is gated.** `Config.v1_style()` semantics and anything the golden tests cover must not change behaviour without the goldens re-run green (needs the private manifest — if you don't have it, say so and stop; do not assume).
4. **Human labels outrank everything** (LESSONS §1–§6). Never let a smoothing/tracking change dilute label constraints.
5. **Detection rate is not accuracy** (LESSONS §12). Any tracking claim needs per-stage evidence: proposals vs labels, spline(t), retimed — probe stages before blaming a detector or labels.
6. **Trust decoded frames only** (LESSONS §8): phones store landscape + rotation metadata; stored dimensions are garbage.
7. **Per-recording calibration:** A/V offset (`--av-offset auto`) and tee position are measured per clip/shot, never hard-coded (LESSONS §9, §16).

## Known rejected approaches — do not re-propose

Seed-cone near the tee, clubhead-detector track selection for the ball, velocity reweighting, pose masking (each failed with evidence — see LESSONS §15 and the physics-gates section). Ball-flight magnitude from one camera is not observable; start-line and curve sign are.

## Failure modes to expect on new footage

- Follow-through shaft against sky mimics a ball track (launch ≤ −10°, ends at hand height).
- The tee estimator can lock onto a static body/background feature and produce confident false tracks; identical tee across many swings is a red flag.
- Off-contract footage (close, face-on, low-res, no ball flight) degrades every stage; characterise the footage before blaming code. Capture contract: LESSONS §20.

## Working rules for agents

- Run `pytest -q` before and after changes; never commit red.
- Behavioural tracking/render changes need before/after numbers in the commit or PR body.
- Issues on GitHub are the task source of truth; keep session narrative out of them.
- Never commit videos except the synthetic example; never widen `.gitignore` exceptions without checking hygiene.
- If a change touches follow-through/backswing retiming, watch for unbounded emission geometry (issue #10 class): cap work against frame size, abstain instead of grinding.
