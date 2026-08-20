# v1 ball-chain transcription contract

This note is the executable contract for the faithful v2 ball port. It records
the production path in v1 `tracer/retrack.py` and the drawing path in
`scripts/render_v4.py`. It contains no footage, oracle data, or private paths.

## Orchestration

`scripts/retrack_batch.py` reads every accepted impact in session order, builds
one tee table for the complete session, then calls `retrack_combined` for each
impact with that table entry. The primary calibrated run selects the v1
bright-difference tee estimator. New footage defaults to the top-hat estimator.

`retrack_combined` requires a measured or session-filled tee when
`require_measured_tee=True`, tries the launch-angle vote first, and may try a
legacy beam result only when one is explicitly supplied. V2 has no beam tracker.
All 33 accepted rows in the primary oracle have `source="vote"`; therefore beam
fallback is not needed for primary parity.

The forced-recall experiment in `tracer/recall.py` is not part of this graph and
is not ported.

## Decode and time bases

`tracer/ingest.py::read_window` seeks backward to a keyframe with PyAV, decodes
sequential source frames, applies display rotation, and retains source PTS in
seconds. `gray=True` returns `(N,H,W)` grayscale; `gray=False` returns BGR
`(N,H,W,3)`. An `fps` argument is a maximum sampling rate, not an FFmpeg
resampling filter: a media-time clock selects the first decoded frame at or
after each sample instant, and the selected source PTS is preserved.

The launch window is `[impact-0.30, impact+1.10)` at a maximum 60 fps. The
descent window is independently decoded as `[impact-0.30, impact+2.20)` at 60
fps. The tee window is `[impact-0.85, impact+0.90)` at 30 fps. Impact frame
index is `searchsorted(PTS, impact, side="left")`, clipped to `[1,N-1]`.

Candidate `rel_frame` is count based: decoded candidate frame index minus the
count-indexed impact index. It is not derived from PTS. Output point time is
`impact_time + rel_frame / 60`, including descent points. Coordinates are full
resolution, display-oriented, stabilized image pixels `(u right, v down)`.

V2 keeps this PTS path separate from its count-grid club decoder.

## Stabilization

`tracer/stabilize.py::stabilize_frames` receives the grayscale decoded window.
Its reference is the uint8 median of every pre-impact frame. The default mask is
two display-relative regions: rows `0.34H:0.66H`, columns `0.56W:W`, plus rows
`0.49H:0.66H`, columns `0:0.16W`.

Frames are reduced so the maximum dimension is 480, high-pass filtered by
subtracting Gaussian sigma 3 around level 128, blurred 3x3, and phase-correlated
under an eroded 5x5 mask with a Hanning window. A transform is reused when its
small-image translation differs from the last accepted transform by less than
`0.08 * scale`. Translation only is accepted when response is at least 0.08,
full-resolution translation is at most 18 px, values are finite, and masked
residual improves by more than `max(0.05, 0.005 * before)`. Rejected frames use
identity. Warping is linear, inverse-map, `BORDER_REFLECT101`.

Output is the registered grayscale stack plus one transform/quality record per
decoded frame. Candidate coordinates remain in reference-frame pixels.

## Candidate observations

`tracer/candidates.py::extract_candidate_observations` runs with `LOOSE` from
`retrack.py`. Values live in `CandidateConfig`:

- background median: last 15 registered pre-impact frames, frozen at impact;
- temporal lag: 3 frames;
- minimum contrast: 14;
- robust noise multiplier: 5.0;
- connected-component area: 2..1600 px;
- analysis region: top 0.72 of frame;
- close kernel: 3 px ellipse;
- background level penalty: 0.08;
- background edge penalty: 0.25;
- maximum spatial penalty: 30;
- persistent edge floor/weight/max: 12 / 0.30 / 40;
- persistent edge dilation: 9 px ellipse.

The scalar evidence threshold is
`max(14, 5 * max(1, 1.4826*MAD))`, measured on a 4x subsample of concatenated
current-minus-background and current-minus-lag values inside the analysis mask.
The per-pixel threshold adds the background-level, background-gradient, and
persistent-preimpact-edge penalties.

Bright evidence is `max(current-background, current-lag)` and dark evidence is
the maximum of their negations. Each polarity is thresholded independently,
closed, and split with 8-connected components. The centroid is evidence
weighted. Each observation carries PTS, `u`, `v`, area, polarity, contrast,
local threshold, excess contrast, persistent penalty, principal-axis
orientation, elongation, and major/minor sigma. Retracking reduces this to
`(rel_frame,u,v,area)`.

## Tee estimation and session table

The ROI convention is `(v0,v1,u0,u1)` in decoded display pixels. Golden parity
uses the manifest ROI when supplied, otherwise the measured v1 ROI stored in
`Config.tee_v1_roi`. New footage defaults to `tee_method="tophat"`; when no ROI
is supplied, v2 derives a mat/feet ROI per impact and uses the coordinate-wise
session median. The derivation finds the largest lower-frame pre/post-impact
foreground component and expands around its horizontal centre and foot level.

### `tee_method="v1"`

`estimate_tee` median-stacks pre frames with PTS `< impact-0.10` and post frames
with PTS `> impact+0.30`. A pixel is gone when maximum BGR channel difference is
over 35 and grayscale pre-minus-post is over 20; the mask is closed 3x3.
Components must have area 55..600, width and height 8..30, aspect 0.55..1.8,
and fill at least 0.55. The centroid needs a 12 px border. Pre-ball contrast
against a 7 px-dilated local ring must be at least 25; the 7x7 centroid patch's
pre-frame standard deviation must be at most 18; mean pre/post drop must be at
least 18. Score is `contrast/100 + drop/100 - abs(area-210)/120 -
abs(1-width/height)`, with `distance_to_prior/150` subtracted when a prior is
provided.

### `tee_method="tophat"`

`tee_tophat.py` computes grayscale pre/post medians, Gaussian sigma 2, then a
41 px elliptical white top-hat. Pre top-hat threshold is 30. Components require
area 100..1500, width and height 10..45, aspect 0.5..2.2, and fill at least
0.45. Mean post top-hat over the component must be at most 0.4 of pre. The
centroid needs a 4 px border. At least 0.70 of pre-frame 7x7 patch means must be
within 18 of their median. Score is `mean_tophat/50 - abs(1-width/height)`, with
the same prior distance penalty.

### Two-pass session semantics

First measure every impact without a prior. The session prior is the
coordinate-wise median of successful first-pass measurements. Any measurement
over 220 px from the prior is re-measured with the prior; a failed remeasurement
retains the original. Remaining misses are filled from the mean of the nearest
one or two known impacts in time. `require_measured_tee` is evaluated only after
this batch process; the hard-coded `(640,1312)` is permitted solely when that
gate is explicitly disabled.

## Launch-angle vote

Only observations with `rel_frame >= 1` and `v < tee_v-60` enter the vote.
Angle is `degrees(atan2(u-tee_u, tee_v-v))`; only absolute angles at most 32
degrees remain. These constants live in `Config.ball_min_above_tee_px` and
`Config.ball_max_launch_angle_deg`.

For half-widths `(1.25,2.0,3.0,4.5)` and centres from -32 through +32 in 0.5
degree steps, select observations within the inclusive angular window. Windows
with fewer than 6 observations are skipped. Duplicate
`(round(half_width,2), selected-index-set)` windows are evaluated once. Within
each decoded frame choose the observation closest to the current centre, then
sort frames.

`_longest_rising` is an O(N^2) longest subsequence. An edge `j -> i` is allowed
when frames differ, the gap is at most 12, and `v[i] < v[j]`. Ties preserve the
first predecessor and `argmax` preserves the first longest chain. Fewer than 6
points abstains that window.

Every surviving window is passed through all physics gates. Accepted-window
score is `N/3 + rise/60 - start_gap/150 - 4*lateral/max(rise,1)`. Strictly
higher score replaces the winner, so ties keep the first half-width/centre.
Reported launch angle is the median point angle from the tee.

## Metrics and gates

`track_metrics` sorts by `rel_frame`. Per-frame speed is Euclidean point step
divided by `max(frame_gap,1)`. `rise=v_first-v_last`; vertical span and lateral
span are peak-to-peak. Early and late speed are the first/last third means.
Launch step is the median of the first up to three speeds. Global speed
violation fraction counts speeds above 1.7 times launch step. Speed-decay
correlation is Pearson correlation of transition index with `log(speed+0.5)`
when at least three transitions exist. A local violation is a speed above
`max(5, 2.0*median(previous up to three speeds))`.

Launch direction is a separate linear fit of `u(frame)` and `v(frame)` over the
first up to six points (at least three), normalized. With two points it is their
difference. Area metrics use positive supplied areas only for the median; zero
means area is unavailable.

`gate_track` stops at the first failure, in this exact order. Values live in
`Config`:

1. at least 6 points (`too_few_points`);
2. first frame at most 5 (`launch_too_late`);
3. positive net rise (`net_descends`);
4. rise at least 480 px (`rise_too_small`);
5. lateral span at most 0.60 times rise (`lateral_spread`);
6. at least 3 transitions and median speed at least 3 px/frame (`stationary`);
7. at least 8 transitions of 2 px/frame or more (`too_few_moving_steps`);
8. early speed strictly greater than late (`no_speed_decay`);
9. finite speed-decay correlation at most -0.25 (`incoherent_speed_decay`);
10. at most 3 local speed violations (`erratic_local_speed`);
11. launch step at least 30 px/frame (`launch_too_slow`);
12. global speed-violation fraction at most 0.10 (`erratic_speed`);
13. when positive areas exist, median area at most 90 (`blob_too_large`);
14. when area exists, maximum area at most 500 (`blob_size_inconsistent`);
15. the tee must not project ahead of the fitted start direction
    (`tee_ahead_of_start`);
16. perpendicular tee-to-launch-ray distance at most 140 px
    (`launch_ray_misses_tee`).

The legacy beam fallback uses a 200 px origin tolerance but otherwise the same
gates. V2 records this contract but does not implement beam tracking.

## Descent extension

Descent runs only after an ascent passes every gate. It starts from the last
ascent point and requires positive ascent area. Median area of the last up to
10 ascent points defines `tail_area`; descent candidate area is positive and at
most `max(24,4*tail_area)`.

Candidates must be later than the ascent end, no more than 8 px above it, no
more than 500 px below it, and inside `24 + 0.70*downward_distance` lateral
corridor. At least 12 candidates are required. Each frame retains at most 80
closest candidates.

The first-order DP permits gaps at most 5. From the ascent endpoint and between
candidates, vertical change is `[-1.5*gap,8*gap]`, absolute lateral change at
most `4*gap`, and Euclidean change at most `9*gap`. Node/edge reward is
`1 + 0.18*max(dv,0) - 0.03*abs(du)` minus
`0.03*abs(log((area_new+1)/(area_old+1)))` and `0.08*(gap-1)` on edges.

Endpoints are tried by descending score. The minimum-v point is the apex. A
continued-rise prefix is retained; subsequent rows form a monotone descending
subsequence, allowing 0.05 px tolerance. Acceptance requires at least 12
descent rows, at least 30 px drop, positive gaps no larger than 5, late mean
speed strictly greater than early, lateral span at most `24+0.70*drop`, and
median descent/ascent-tail area ratio in `[0.15,4.0]`. Output includes the
prefix and accepted descent, apex frame/point, counts, drop, lateral span, and
area ratio. A failed descent never changes ascent acceptance.

## Output and drawn fit

Accepted output preserves raw selected observations. `n_observed` is ascent
count; `n_total_observed`, `n_ascent`, and `n_descent` are explicit. It also
stores rise, lateral span, start gap, launch angle, tee, start sign, points, and
optional apex/descent metrics. Abstention returns no points and is rendered as
nothing.

`tracer/fit.py` and `tracer/features.py` are calibrated analysis utilities but
are not called by `retrack_combined` or the v4 renderer. They are retained as
ports, not inserted into the accepted observation chain.

`scripts/render_v4.py::fit_ball_track` sorts points by `t_s` and fits separate
FITPACK `UnivariateSpline` x(t)/y(t) pieces with degree `min(3,N-1)` and
`s=N*1px^2`. Without a usable apex it fits one piece. With an apex, each side
must have at least three points including the shared apex; the two splines are
offset so both evaluate exactly to the observed apex. Rendering samples the
fitted path at output-frame time. V2's compositor follows this same apex split.
