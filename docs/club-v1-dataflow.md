# v1 club-chain transcription contract

This note records the v1 implementation contract used by the v2 faithful port.
It intentionally contains no private footage, labels, oracle values, or paths.

## Orchestration and label merge

`club_finalize.py` validates every label document, normalises legacy decode
offsets, then merges documents by `(swing_id, rel_frame)`. Document order is
original, top-up, full-downswing; later documents replace earlier collisions.
It runs `club_refit.py`, then `club_retime.py`, then `render_v4.py`.

The v1 label document stores full-resolution autorotated decoded coordinates.
Each sample carries absolute `frame_time_s`, nominal whole-video
`frame_index = round(frame_time_s * fps)`, canonical window-local
`decode_frame_index`, tracker-relative `rel_frame`, and the five-way calibration
phase. The canonical image identity is the sequentially decoded window frame.
The full-downswing sampler selects every unlabelled frame from top through
impact-minus-one plus weak unlabelled backswing frames.

## Raw tracker

`tracer/clubtrack.py` decodes one fixed-rate 60 fps window ending immediately
before impact. It emits display/autorotated pixel coordinates and absolute
seconds. `rel_frame` is zero-based from the detected takeaway frame. The
backswing includes takeaway through top. The downswing contains top+1 through
the final decoded pre-impact frame, followed by an exact impact anchor at the
measured ball position.

Pose supplies the wrist midpoint. A wrist-anchored dark-ridge ray search scores
shaft directions, with shake-corrected motion support. A second-order monotone
DP over angle and angular velocity chooses the path. Strong image rows are
`detected`; weak rows are filled between trusted angle/radius anchors and marked
`interpolated`. The downswing is tracked in reverse from impact toward the top.

## Spatial refit

`club_refit.py` first computes one global coordinate-wise median
`clicked - detected` bias for each of takeaway, mid-backswing, top, downswing,
and delivery. Only detected tracker samples paired with labels contribute.

Each swing is split into two independent parametric FITPACK splines at one
shared top support. The top support is the raw point nearest `top_t`, shifted by
the global top bias, with weight `1e6`. The impact anchor has weight `1e6`.
Every human click has weight `25`. An impact-time click is ordered 1/240 s before
the exact impact anchor so both spatial points survive.

Tracker interpolations are always dropped. Label-collision frames and the raw
top observation are excluded from tracker candidates. Detected delivery points
have zero weight and are excluded. Remaining detected points are shifted by the
phase bias and weighted exactly as v1: takeaway 0.45, mid-backswing 0.65,
top 2.50, downswing 2.00.

One label-smoothing tolerance is selected globally across every phase of every
swing by leave-one-label-out RMSE from `(0, 0.75, 1.5, 2.5, 4, 6)` pixels.
Candidates with an all-label training maximum over 10 px are ineligible; ties
favour more smoothing. FITPACK's `s` is the sum of squared
`weight * tolerance` for every non-hard support, using the selected tolerance
for labels and 18 px for tracker points.

The label-only curve gates tracker candidates by residual at their timestamp.
Per-phase thresholds are `median + 4 * 1.4826 * MAD`, clipped to v1 floors and
caps: takeaway 80..140, mid-backswing 65..115, top 45..85, downswing 55..95.
The retained set is refit and re-gated up to eight times. The final spatial
curve is sampled at 8192 spline parameters and numerically parameterised by
pixel arc length.

Refit output is sampled only at the raw track's timestamps. It preserves raw
phase names, but every coordinate comes from the appropriate independent
spatial curve. Provenance records arc length plus accepted/rejected tracker
status. A synthetic pre-anchor record preserves an impact-time click when one
exists.

## Dense image-evidence retiming

`club_retime.py` turns each refit phase record sequence into a new PCHIP spatial
arc. The downswing sequence starts with the last backswing record, preserving
one shared cusp. Duplicate chord positions are removed; PCHIP x/y are densely
sampled and resampled at approximately 4 px arc bins.

The canonical decode window is `SwingWindow.for_swing(..., post_s=1/60)`.
Retiming starts at the first refit record, splits inclusively at the count-indexed
top frame, and ends inclusively at the impact frame. Backswing and downswing are
retimed independently; the duplicated downswing top row is omitted when the
two output lists are joined.

For each frame and arc state, the v1 emission combines:

- 2.2 times mean wrist-to-state shaft-ridge support;
- 1.30 times clubhead motion;
- 1.15 times local absolute head contrast gated by motion.

The row is normalised by its median and 90th-percentile spread, then clipped to
`[-2.5, 4]`. Human labels are soft pins at their canonical frame: subtract
`3.2 * (arc_distance / 14)^2`, with a further -80 beyond 28 px. Impact labels
do not pin retiming because the exact endpoint wins. First and last phase
states are hard-pinned to arc zero and arc end.

The same second-order monotone `dp_sweep` used by the tracker operates on
(previous step, arc state), with arc steps from zero through 320 px. Its exact
retime parameters are `accel_w=0.00020`, `start_w=8.0`,
`decel_w=0.00150`; per-frame acceleration weight is 0.00005 through the final
label and 0.00100 afterwards. The result is monotone by construction.

Output `rel_frame` is re-zeroed from the first curve frame, `t_s` comes from the
canonical decode count (except exact impact time), and provenance stores the
selected phase-local arc length and image score.

## Rendering and temporal clipping

`render_v4.py::fit_club_track` keeps backswing and downswing as independent
pieces, prepending the final backswing row to the downswing. Each trusted dense
curve is PCHIP-interpolated by chord parameter, densely reparameterised by pixel
arc length, and played using the monotone retimed `arc_length_px` knots. The
shared top row's downswing arc is reset to zero.

`FittedPath.samples_until(t)` returns an empty list before the first backswing
record. It draws only samples already reached at `t`, phase by phase, so the arc
exists only on `[takeaway_t, impact_t]` and never before takeaway. The two
independent spatial pieces share one coordinate but retain independent tangents,
which makes the top a cusp rather than a rounded loop.
