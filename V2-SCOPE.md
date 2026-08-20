# Tracer v2 — Scope

Status: scoped 2026-08-17, build not started. v1 (a private repo) remains the working system; v2 begins against the **next** filming session's footage, captured under the contract in LESSONS.md §20.

## Purpose

**v2 is a reel-making tool.** Input: a range or course video from a fixed phone camera. Output: a reel with club arc, ball trail and follow-through overlays burned in. That is the whole product.

It is not an analysis tool. Shot shape, curve sign, launch numbers, club kinematics and any fusion with launch-monitor data are out of scope here and live in the private local pipeline (see "What stays local").

## Two repos, two jobs

| | This repo (`golf-tracer-v2`) | Local / private |
|---|---|---|
| Job | Footage in, tracer reel out | Data combination: TrackMan/Garmin session data joined to video, per-shot stats, labels, hand-labelled ground truth, personal footage |
| Contents | Trackers, compositor, labeller UI, tests, synthetic or consented sample clips | the private v1 repo, local footage drives, label sets, private notes |
| Trend | Toward something another golfer can clone and run on their own video | Never published |

Rules that follow from this:

- No personal footage, label sets, TrackMan exports or session data are committed here. Fixtures are synthetic or short consented clips.
- Nothing in this repo reads from local footage drives, private notes, or v1's `data/` by path. Inputs are arguments.
- Stats-overlay reels (the golf-supercut flow) consume this tool's output plus local data; that composition step stays local.
- v1 stays private as-is. Anything ported into v2 comes across clean of paths, labels and footage.

## Architecture: four phase modules, one compositor

| Phase | Dynamics | v1 status | v2 plan |
|---|---|---|---|
| Backswing (address→top) | slow, trackable | works, calibrated | port as-is into phase framework |
| Downswing (top→impact) | 4x faster, blurred | works after full ground-truthing | port; detector-assisted labels |
| Ball flight (impact→landing) | ballistic | `retrack.py` calibrated, 33/58 ceiling on v1 footage | port unchanged; new footage lifts ceiling |
| Follow-through (impact→finish) | moderate, mostly sky | **absent** | new module — backswing tracker mirrored, anchored at impact |

Each phase: tracker → label-constrained fit → phase-specific gates → audit. One compositor renders any combination (club-only, ball-only, combined) with the v1 conventions (arc-length splines, DP retiming, gradient fade).

## The core v2 bet: near-zero per-session labelling

Bootstrap a clubhead detector from v1's ~200 ground-truth clicks (plus ball-track observations for the ball side, TrackNet-style). New session flow: detector proposes labels → human **corrects** in the labeller UI instead of clicking from zero → corrections feed back into training data. Target: <10 min human time per session, converging toward zero. For a shareable tool this is the difference between "works on Milo's swings" and "works on yours".

Detector weights trained on private labels can ship in the repo (weights are not footage); the training set itself does not.

## Non-goals

- Shot-shape or curve measurement, launch numbers, club speed/plane. Overlay fidelity only.
- Joining video to launch-monitor or watch data. Local.
- Rebuilding the ball tracker (calibrated, working; ceiling is capture-side).
- Real-time processing.

## Sequencing

1. v1 closes out: full-downswing labels → v6 reels (done, private).
2. Next range session filmed under the capture contract.
3. v2 phase framework + follow-through module, validated on the new footage. Ported code arrives path-clean.
4. Detector bootstrap; measure label-time reduction session over session.
5. Once it runs end-to-end from a CLI on a fresh video with no local dependencies, it is ready to share.

## Acceptance

- All four phases audited frame-by-frame with zero constraint failures on the new session.
- Combined reel covers full swing + ball flight + follow-through.
- Human labelling time for the session under 30 min (stretch: 10).
- `git clone` + one video + one command produces a reel on a machine that has never seen the local data.
