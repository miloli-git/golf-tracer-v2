# Golf Shot Tracer — Build Lessons

Distilled from the private v1 build, primarily the 2026-08-15 → 2026-08-17 sprint that took the pilot outdoor-range footage from raw video to fully calibrated club-arc and ball-flight reels. These are the constraints v2 is built on. Session-level detail lives in private notes, not in this repo.

## Calibration and ground truth

1. **Ground truth beats smoothing, always.** ~15 minutes of hand clicks exposed 90–490 px tracker error that its own 77–81% "detection rate" concealed, and killed a smoothing pass that would have polished wrong arcs. Never tune a renderer against uncalibrated tracks.
2. **Human labels are the asset; everything else is support.** Fit curves *through* clicks (hard constraints), weight tracker detections as secondary evidence with per-phase bias correction, reject outliers against the label-constrained fit.
3. **Interpolated tracker points can be worse than no data.** Delivery-phase interpolations (pulled-to-anchor) measured 200–490 px off — dropping all 57 of them improved the fit. Measure before trusting any "filled" data.
4. **Only labelled frames are certified.** Every audit defect lived in IMAGE-ONLY (unlabelled) rows. Targeted top-ups are whack-a-mole: each round surfaces the next gap. Ground-truth entire phases in one sitting; the marginal click is cheap, the extra review round is not.
5. **Label conventions must be consistent and explicit.** Head centre-of-mass throughout; the tracker's hosel bias is absorbed by per-phase bias estimation, not by bending the convention.
6. **Labels are per-session.** Clicks do not transfer to new footage. v2's core economic problem is driving per-session labelling cost toward zero (detector bootstrapped from v1's ~200 labels; human corrects proposals instead of clicking from scratch).

## Decode determinism

7. **ffmpeg seek boundaries shift the decoded frame set.** Rounded vs exact timestamps changed which frames two swings decoded; labels acquired a silent +1-frame offset on two other swings. One sequential single-seek decode per window, count-indexed from window start, shared by labeller / fitter / retimer / renderer. Never per-frame seeks.
8. **ffprobe reports stored dimensions; decode autorotates.** 1920x1080 stored → 1080x1920 decoded. Reading the stored size yields garbage frames and zero detections. Trust decoded frames only.
9. **A/V offset is real and constant per recording, not universal** (~0.19 s audio-leads on one phone recording; ~0.12 s on a second recording from a different setup, checked frame by frame, issue #5). Audio impact clicks are excellent strike ground truth, but frame extraction at audio time lands in follow-through unless corrected, and a wrong constant shifts impact by 4-5 frames and breaks the ball launch gate. Measure the offset per clip; only fall back to a constant.

## Fitting and rendering

10. **Decouple space from time.** Spatial truth: arc-length-parameterised spline through labels (phase-split at the top cusp — the reversal must stay a cusp, never rounded into a loop). Timing: image-evidence DP (monotone Viterbi over arc × frame). Linear time→arc interpolation fails wherever speed varies — the club covers ~850 px of arc in the last 4 pre-impact frames after ~28 px/frame mid-downswing.
11. **Draw fitted curves sampled per frame, never raw observation polylines.** Gaps and jitter render as steps; three separately-reported "jagged clip" defects were one root cause.
12. **Verify numerics visually, every time.** A static object fits any smooth trajectory (one early tracker returned 53 "observations" of the same pixel). High detection percentages coexist with tens-of-px positional error. Frame strips with the overlay burned in are the only acceptance test that means anything.

## Physics gates (measured on this footage, not guessed)

13. Real struck balls rise ≥480 px; walking people produce convincing 250–300 px rising streaks. Min-rise gates exist for walkers.
14. **Launch-angle voting works because geometry is on your side:** every true observation lies near one ray from the tee (0.9–2.0° spread); club arcs sit ~30° away. Histogram the angle, take the peak.
15. **Tee-proximity seeding is the wrong instinct** — the ball is undetectable until it has travelled ~550 px (too fast, too blurred). Seeds near the tee return static mat clutter.
16. **Measure the tee per shot.** Hand placement wanders ±22 px; a hard-coded tee was 35–80 px wrong and single-handedly broke the angle vote. (A fixed rubber tee later in the session held ±1 px — capture-side fixes beat software.)
17. **Swing timing is faster than intuition:** backswing 0.33–0.52 s, top-to-impact 0.20–0.32 s. Naive ~1.6 s windows search mostly waggle and address.
18. **Motion blur helps detection.** At 1/250 s the ball is a ~150 px streak nothing else produces; at 1/2000 s it is a 13 px dot identical to thousands of range balls. Do not film in sports/action mode.

## Knowing when to stop

19. **Null results are results.** A 5-million-hypothesis forced-recall experiment recovered 0 of 25 abstained shots with 0 regressions: 33/58 tracked is this footage's genuine ceiling, not a threshold artefact. Better recall comes from the capture contract, not more search.
20. **Capture contract for future sessions** (value order): camera ON the ball-target line with offset measured (off-line parallax dominates curve signal ~6:1 and fakes a consistent slice); orange ball (floor is carpeted in yellow ones); 4K60; AE locked ~1/250 s; weighted tripod; alignment stick; 5 s static pre-roll; HDR off; two deliberate slices + two hooks as calibration.
21. **Curve sign is a geometry problem, not a detection problem.** Without an on-line camera or measured offset, azimuth drift from parallax is indistinguishable from real curvature.

## Generalisation (second outdoor corpus, 2026-08-17)

26. **The v1 ball tracker does not transfer to a second outdoor video unchanged.** On a sunny sky-dominant range session the tee estimator found 0/94 balls (harsh sun shades half the ball, and the golfer's shadow crosses the tee after impact, flooding the diff); a top-hat present-then-absent estimator fixed that (94/94 measured or filled). The launch vote then tracked 45/94, but roughly 10 of those are the **follow-through shaft against sky** (launch ≤ −10°, tracks end at hand height): a false-track mode the tree-line footage never exposed. Every gate value and ROI is per-footage; the tracker needs a second golden corpus, and the follow-through phase (which knows where the club is after impact) is the natural mask for this false mode.

## Orchestration (multi-agent build process)

22. **Partition file ownership when agents share a working tree**; forbid agents from git; the orchestrator commits between waves.
23. **A stopped agent leaves partial edits.** Diff before committing anything after a kill; park salvageable partial work as a patch, revert the tree.
24. **Make agents state failures plainly.** The most valuable agent reports were the honest ones ("strict acceptance test NOT passed", "0 recovered, recommend do not merge"). Prompts must demand per-item verification evidence and explicitly reward reporting what still looks off.
25. **Commit the baseline before fanning out.** Two uncommitted modules and scratchpad-only renderers nearly evaporated with a dead session.

## Porting (v2 build, 2026-08-18)

27. **"Port" means transcribe, not re-derive.** The first agent pass at the club chain was a plausible rewrite that passed its own unit tests and drew a zig-zag fan on the first real user labelling session (26–93 px RMS vs v1 with identical labels). The fix was mechanical: dump every intermediate from both pipelines side by side (`tools/club_parity.py`), transcribe v1 at the first diverging stage, repeat. Seven divergences later the RMS was 0.000 px. Every future port of a calibrated stage gets a side-by-side harness first and a numeric parity gate, and the acceptance case is a real user session, not a synthetic fixture.
