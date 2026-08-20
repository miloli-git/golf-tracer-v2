# v2 Plan

Companion to `../V2-SCOPE.md` (what) and `../LESSONS.md` (constraints). This is the how: workstreams, decisions, sequencing, risks. PRDs per workstream in `prds/`.

## Summary

v2 is a clean, shareable reel-making tool built by porting v1's calibrated trackers into a phase-structured package, adding the missing follow-through phase, and replacing click-from-zero labelling with detector-proposed labels the human corrects. Ten workstreams, four milestones. Everything except the detector and the new-footage validation can start now against local v1 footage passed in as arguments. Effort is in focused hours, not weeks: framework and ports are mostly mechanical, follow-through and detector are the two pieces of real work.

## Workstreams → PRDs

| # | Workstream | Type | PRD | Rough effort |
|---|---|---|---|---|
| 01 | Package + CLI + deterministic decode | port/refactor | `prds/PRD-01-core.md` | 3–4 h |
| 02 | Impact detection (audio) | port | `prds/PRD-02-impacts.md` | 1–2 h |
| 03 | Backswing phase | port | `prds/PRD-03-backswing.md` | 2 h |
| 04 | Downswing phase | port | `prds/PRD-04-downswing.md` | 3 h |
| 05 | Ball flight phase | port | `prds/PRD-05-ball-flight.md` | 3 h |
| 06 | Follow-through phase | new | `prds/PRD-06-follow-through.md` | 4–6 h |
| 07 | Labeller (correction mode) | port + extend | `prds/PRD-07-labeller.md` | 3–4 h |
| 08 | Compositor + reel presets | port + extend | `prds/PRD-08-compositor.md` | 3 h |
| 09 | Detector bootstrap | new | `prds/PRD-09-detector.md` | 6–10 h + training |
| 10 | Share-readiness | new | `prds/PRD-10-share.md` | 2–3 h |

Total ≈ 30–40 focused hours across 5–8 sessions. Ports are cheap because the calibrated algorithms already exist; the cost is scrubbing paths, unifying interfaces and re-verifying visually (LESSONS §12).

## Architecture in one picture

```
video ──► decode (single-seek, count-indexed) ──► impacts (audio, A/V-corrected)
                                                        │ per-swing windows
              ┌──────────────┬──────────────┬───────────┴───┬──────────────────┐
              ▼              ▼              ▼               ▼                  │
         backswing      downswing      ball flight    follow-through           │
         tracker        tracker        tracker        tracker                  │
              │              │              │               │                  │
              └──── label-constrained fit + phase gates + audit (per phase) ───┘
                                          │  phase tracks (json)
                                          ▼
                       compositor (splines, DP retiming, fade, presets) ──► reel.mp4
                                          ▲
              labeller: detector proposes ──► human corrects ──► labels ──► (feeds detector)
```

Data model: one `Session` (video + decoded frame index + impacts) → many `Swing` (window) → per-phase `Track` (observations, fitted spline, retiming, audit) → `Reel` spec (which phases, style preset, output size).

## Decisions (made, unless you overturn)

1. **Start now, don't wait for new footage.** PRDs 01–08 and 10 run against local v1 footage passed by path. Only the *validation* milestone needs the capture-contract session. Waiting would idle 80% of the work.
2. **Python package `golftracer`, single CLI `golftracer`** with subcommands (`impacts`, `track`, `label`, `render`, `reel`). `reel` is the one-command path: `golftracer reel video.mp4 --out reel.mp4`.
3. **Follow-through prototypes on the original v1 footage now.** Mostly-sky background is the easy case; the module gets its real test on new footage but the mirrored-backswing design can be proven today.
4. **Detector: small fine-tuned single-class object detector for the clubhead** (YOLO-family nano, CPU-inferable) trained on v1's ~200 clicks + augmentation, run only inside the swing window; ball side stays classical (streak candidates + angle vote) until a TrackNet-style temporal model earns its place. Weights ship, training data doesn't.
5. **Labels stay per-session JSON in the user's own output dir**, never in the repo. Repo carries the schema and a synthetic fixture.
6. **License at share time: MIT.** Decide nothing else about "public" until PRD-10 acceptance passes.
7. **Confirmed 2026-08-17:** package name `golftracer`; follow-through is clubhead-only for now (body/hands trace via MediaPipe pose is a later layer, only if it helps the reel); `social` 9:16 is the default preset, `source` via flag.

## Milestones

- **M1 Skeleton (PRDs 01, 02, 08 minimal):** `golftracer reel` runs end-to-end on a local video and produces a reel from *unfitted* raw tracks or even a stub, proving decode → windows → compositor plumbing. Commit the baseline before fanning out (LESSONS §25).
- **M2 Parity (PRDs 03, 04, 05, 07):** v2 reproduces v1's `combined_v5` output on the original footage frame-for-frame (pixel diff on overlay masks within tolerance) using v1's labels loaded from a local path. This is the regression gate for the port.
- **M3 Full swing (PRD 06):** follow-through tracked and audited; combined reel covers address → finish + ball.
- **M4 Shareable (PRDs 09, 10):** detector-proposed labels cut human time; clone + one video + one command works on a clean machine; new-footage validation passes with zero audit failures.

## Sequencing (dependency graph)

```
01 core ──┬── 02 impacts ──┬── 03 backswing ──┬── 06 follow-through
          │                ├── 04 downswing ──┤
          │                └── 05 ball ───────┤
          ├── 08 compositor (min) ── M1        ├── 08 compositor (full) ── M2 ── M3
          └── 07 labeller (port) ──────────────┘
                     └── 09 detector (needs 07 correction mode + 04 labels) ── M4
10 share runs continuously; final pass after 09
```

Parallelisable waves (for multi-agent builds, LESSONS §22–25):
- **Wave A** (after 01 baseline commit): 02, 07-port, 08-min in parallel; distinct file ownership.
- **Wave B**: 03, 04, 05 in parallel (each owns `golftracer/phases/<name>.py` + its tests).
- **Wave C**: 06, 08-full, 07-correction-mode.
- **Wave D**: 09, 10.

Session plan (each 2–4 h): S1 = 01 + M1; S2 = wave A + wave B kickoff; S3 = wave B → M2; S4 = 06 → M3; S5–S6 = 09; S7 = 10 + new-footage validation → M4.

## Risks and flags

- **Port drift.** Refactoring 13k lines invites silent behaviour change. Mitigation: M2 parity gate is a pixel-mask diff against v1's `combined_v5`, not a "looks fine". Keep v1 untouched as the oracle.
- **Follow-through has no ground truth yet.** Budget a labelling pass on 3–4 swings before trusting anything (LESSONS §1, §4). Cheap: ~50 clicks.
- **Detector on 200 labels is thin.** Heavy augmentation, and the correction loop is the point: it only needs to be good enough that correcting beats clicking. Measure minutes-per-session, not mAP.
- **Shareability erodes silently.** A single hard-coded personal path or fixture from private footage undoes it. PRD-10 includes a private-path CI grep and a clean-VM smoke run.
- **Scope creep back into analysis.** Any request for numbers (speed, curve, launch) is out of scope by V2-SCOPE; route to the local pipeline.
- **Capture contract still unfilmed.** M4 validation is gated on it; nothing else is.

## Golden tests (parity gates)

Golden inputs and oracles live in the **private v1 repo** as a manifest (`golden/v2-golden.yaml` in that repo): the source video, 58 audio impacts, the 7 calibrated club swings (impact/top/window times, 195 merged labels, oracle fitted arcs `clubtrack_*_final.json`, oracle reel `clubhead_v6`), the ball oracle (`retrack_*_v2.json`, 33/58 with the abstention set), and the combined oracle reel `combined_v5`. Secondary corpora (2026-07-02 outdoor grass, sim) are listed for follow-through prototyping and detector diversity, not as gates.

v2 never embeds any of this. `pytest -m golden` reads `--golden PATH` / `GOLFTRACER_GOLDEN`, skips cleanly when unset (CI), and runs the M2 parity checks when set (local). Repo tests use the synthetic fixture only.
