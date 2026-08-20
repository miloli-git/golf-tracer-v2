"""Central configuration for every tunable used by the M1 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    # Measured default from the original phone footage: audio leads video.
    av_offset_s: float = 0.19
    # tee-patch departure lags the strike by ~1.7 frames at 60 fps; calibrated on
    # 41 impacts of the primary golden session (departure median 0.162 s vs the
    # hand-verified 0.19 s constant)
    av_departure_lag_s: float = 0.028
    window_pre_s: float = 0.60
    window_post_s: float = 1.50
    backswing_pre_s: float = 1.90
    downswing_pre_s: float = 0.35
    ball_post_s: float = 1.50
    follow_through_post_s: float = 1.50
    # Hard wall-clock ceiling for one swing's tracking work. The worker process
    # is terminated on expiry so arbitrary native OpenCV/NumPy work cannot hold
    # the session loop indefinitely.
    track_swing_timeout_s: float = 600.0

    # Club phase fitting/retiming. Values are the measured v1 calibration defaults.
    club_label_weight: float = 25.0
    club_hard_weight: float = 1_000_000.0
    club_fit_tolerance_px: float = 18.0
    # learned-detector proposals are ~2 px on held-out frames; smooth them far
    # less than the 18 px raw-tracker tolerance or the fast post-impact frames
    # get rejected as outliers
    club_fit_tolerance_detector_px: float = 5.0
    club_label_smoothing_px: float = 2.5
    club_label_smoothing_candidates_px: tuple[float, ...] = (0.0, 0.75, 1.5, 2.5, 4.0, 6.0)
    club_max_training_label_residual_px: float = 10.0
    club_impact_label_lead_s: float = 1.0 / 240.0
    club_spline_samples: int = 8192
    club_label_tolerance_px: float = 14.0
    club_arc_bin_px: float = 4.0
    # Curves longer than this multiple of the decoded-frame diagonal are invalid
    # fit geometry. Guard before building the arc-sample x shaft-radius emissions.
    club_max_arc_frame_diagonals: float = 2.0
    club_max_arc_step_px: float = 320.0
    club_retime_accel_weight: float = 0.00020
    club_retime_start_weight: float = 8.0
    club_retime_decel_weight: float = 0.00150
    club_retime_frame_accel_weight: float = 0.00005
    club_retime_delivery_accel_weight: float = 0.00100
    club_tracker_weight_takeaway: float = 0.45
    club_tracker_weight_mid_backswing: float = 0.65
    club_tracker_weight_top: float = 2.50
    club_tracker_weight_downswing: float = 2.00
    # Follow-through is a new, separately calibrated phase. These defaults are
    # intentionally independent of the v1 backswing/downswing calibration.
    club_tracker_weight_followthrough: float = 1.00
    club_outlier_floor_takeaway_px: float = 80.0
    club_outlier_cap_takeaway_px: float = 140.0
    club_outlier_floor_mid_backswing_px: float = 65.0
    club_outlier_cap_mid_backswing_px: float = 115.0
    club_outlier_floor_top_px: float = 45.0
    club_outlier_cap_top_px: float = 85.0
    club_outlier_floor_downswing_px: float = 55.0
    club_outlier_cap_downswing_px: float = 95.0
    club_outlier_floor_followthrough_px: float = 65.0
    club_outlier_cap_followthrough_px: float = 120.0
    club_render_lead_s: float = 0.25
    club_tracker_bias_backswing: tuple[float, float] = (0.0, 0.0)
    club_tracker_bias_downswing: tuple[float, float] = (0.0, 0.0)
    club_tracker_bias_followthrough: tuple[float, float] = (0.0, 0.0)
    follow_finish_velocity_floor_px_s: float = 45.0
    follow_finish_stationary_frames: int = 8
    follow_finish_min_time_s: float = 0.35
    follow_exit_margin_px: float = 8.0
    follow_exit_missing_frames: int = 4
    follow_retime_initial_speed_weight: float = 0.02
    follow_retime_initial_speed_frames: int = 4
    # detector proposals time the follow-through retime as capped soft pins
    # (label pins use 3.2 * (d / tol)^2 uncapped; 0 disables)
    follow_detector_pin_weight: float = 3.2
    follow_detector_pin_cap: float = 12.0

    # PRD-09 detector. The ROI retains the full swing width while removing the
    # floor; inference is resized by the nano model, not by the phase trackers.
    detect_roi: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.90)
    detect_input_size_px: int = 416
    detect_box_size_px: float = 40.0
    detect_confidence: float = 0.05
    # Correction mode always shows the model's best candidate. This lower floor
    # never feeds autonomous fits or held-out detector gates.
    label_proposal_confidence: float = 0.001
    detect_acceptance_radius_px: float = 15.0
    detect_trajectory_min_within_pct: float = 70.0
    detect_trajectory_max_join_gap_px: float = 1.0

    # Ball gates, measured on the v1 outdoor corpus (PRD-05 / LESSONS 13-16).
    ball_fps: float = 60.0
    ball_pre_s: float = 0.30
    ball_post_s: float = 1.10
    ball_descent_post_s: float = 2.20
    ball_require_measured_tee: bool = True
    ball_tee_xy: tuple[float, float] = (640.0, 1312.0)
    ball_origin_tolerance_px: float = 140.0
    ball_min_rise_px: float = 480.0
    ball_min_inliers: int = 6
    ball_max_lateral_ratio: float = 0.60
    ball_min_above_tee_px: float = 60.0
    ball_max_launch_delay_frames: int = 5
    ball_max_launch_angle_deg: float = 32.0
    ball_bin_widths_deg: tuple[float, ...] = (1.25, 2.0, 3.0, 4.5)
    ball_centre_step_deg: float = 0.5
    ball_min_median_step_px: float = 3.0
    ball_min_launch_step_px: float = 30.0
    ball_speed_violation_factor: float = 1.70
    ball_max_speed_violation_frac: float = 0.10
    ball_max_median_area_px: float = 90.0
    ball_max_blob_area_px: float = 500.0
    ball_moving_step_px: float = 2.0
    ball_min_moving_steps: int = 8
    ball_min_speed_decay_correlation: float = -0.25
    ball_local_speed_violation_factor: float = 2.0
    ball_max_local_speed_violations: int = 3
    ball_max_frame_gap: int = 12
    # max times one rounded coordinate may appear in the accepted track;
    # repeats beyond this are static clutter (0 disables)
    ball_max_coord_repeats: int = 2
    # gappy/dwelling ball tracks render as an arc-length piece instead of the
    # loop-prone v1 time-spline; v1_style() disables for parity
    ball_render_arc_fit: bool = True
    ball_descent_max_gap: int = 5
    ball_descent_apex_slack_px: float = 8.0
    ball_descent_lateral_slack_px: float = 24.0
    ball_descent_max_lateral_ratio: float = 0.70
    ball_descent_max_step_px: float = 9.0
    ball_descent_max_lateral_step_px: float = 4.0
    ball_descent_max_vertical_step_px: float = 8.0
    ball_descent_area_factor: float = 4.0
    ball_descent_min_area_ratio: float = 0.15
    ball_descent_min_points: int = 12
    ball_descent_min_drop_px: float = 30.0
    ball_descent_max_drop_search_px: float = 500.0
    ball_descent_candidates_per_frame: int = 80
    # New footage uses top-hat with a derived ROI. Golden parity explicitly
    # selects ``v1`` and supplies the measured v1 corpus ROI.
    tee_method: str = "tophat"
    tee_pre_s: float = 0.85
    tee_post_s: float = 0.90
    tee_roi: tuple[int, int, int, int] | None = None
    tee_roi_above_feet_height_ratio: float = 0.18
    tee_roi_below_feet_height_ratio: float = 0.12
    tee_roi_left_of_golfer_width_ratio: float = 0.36
    tee_roi_right_of_golfer_width_ratio: float = 0.60
    tee_v1_roi: tuple[int, int, int, int] = (1230, 1430, 380, 860)
    tee_v1_area_px: tuple[int, int] = (55, 600)
    tee_v1_expected_area_px: float = 210.0
    tee_tophat_kernel_px: int = 41
    tee_tophat_min: float = 30.0
    tee_area_px: tuple[int, int] = (100, 1500)
    tee_side_px: tuple[int, int] = (10, 45)
    tee_static_fraction: float = 0.70
    tee_static_tolerance: float = 18.0
    ball_shaft_rule_enabled: bool = False
    ball_shaft_launch_max_deg: float = -10.0
    ball_hands_height_px: float | None = None
    ball_hands_height_ratio: float = 0.23

    impact_sample_rate: int = 16_000
    impact_fft_samples: int = 1_024
    impact_hop_samples: int = 256
    impact_band_low_hz: float = 2_000.0
    impact_band_high_hz: float = 8_000.0
    impact_floor_window: int = 40
    impact_floor_guard: int = 6
    impact_attack_threshold: float = 5.0
    impact_z_threshold: float = 8.0
    impact_min_onset: float = 0.00008
    impact_min_gap_s: float = 5.0
    # S1-measured audio refinement radius.  The v1 visual tee refinement used
    # 0.55 s, but that wider window can jump to a neighbouring range strike.
    impact_search_radius_s: float = 0.25

    # Measured visual-swing defaults ported from v1.  ROIs are normalized
    # display-space (x0, x1, y0, y1), so autorotated decoded geometry applies.
    motion_fps: float = 15.0
    motion_width: int = 192
    motion_roi: tuple[float, float, float, float] = (0.10, 0.85, 0.12, 0.90)
    # v1 started at 90 / 2.3.  The S2a unseeded sweep measured that 88 / 2.1
    # retains the same visual-first behaviour while recovering weak swings.
    motion_floor_percentile: float = 88.0
    motion_impulse_ratio: float = 2.1
    motion_background_window_s: float = 4.0
    motion_peak_window_s: float = 0.4
    motion_hwaccel: str = "auto"
    pose_window_s: float = 1.5
    pose_step_s: float = 0.2
    # v1 started at -0.02; -0.035 is the S2a measured count/recall setting.
    pose_wrist_shoulder_gap: float = -0.035
    pose_scale: float = 0.5
    pose_enabled: bool = True
    pose_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )

    calibration_frames: int = 5
    calibration_span_s: float = 0.50
    qa_every_frames: int = 15
    qa_columns: int = 5
    qa_thumb_width: int = 240

    marker_radius_px: int = 18
    marker_thickness_px: int = 4
    caption_margin_px: int = 28
    caption_scale: float = 0.75
    social_width: int = 1080
    social_height: int = 1920
    render_crf: int = 18
    render_fps_fallback: float = 60.0
    render_lead_s: float = 0.25
    render_post_s: float = 3.60
    render_alpha_tail: float = 0.10
    render_alpha_head: float = 0.90
    render_glow_alpha_scale: float = 0.40
    club_colour_bgr: tuple[int, int, int] = (60, 220, 255)
    club_width_px: int = 3
    club_glow_px: int = 5
    club_fade_length_s: float = 10.0
    ball_colour_bgr: tuple[int, int, int] = (0, 0, 255)
    ball_width_px: int = 4
    ball_glow_px: int = 6
    ball_fade_length_s: float = 3.6
    follow_colour_bgr: tuple[int, int, int] = (255, 180, 60)
    follow_width_px: int = 3
    follow_glow_px: int = 5
    follow_fade_length_s: float = 1.5

    @classmethod
    def v1_style(cls) -> "Config":
        """Return the v1 reel palette and stroke geometry for render parity."""
        return cls(
            club_colour_bgr=(60, 220, 255),
            club_width_px=2,
            club_glow_px=0,
            ball_colour_bgr=(0, 210, 255),
            ball_width_px=3,
            ball_glow_px=5,
            # v1 had no separate follow layer; its club style is equivalent.
            follow_colour_bgr=(60, 220, 255),
            follow_width_px=2,
            follow_glow_px=0,
            ball_render_arc_fit=False,
        )

    @classmethod
    def load(cls, yaml_path: str | Path | None = None) -> "Config":
        if yaml_path is None:
            return cls()
        path = Path(yaml_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("config YAML must contain a mapping")
        valid = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - valid)
        if unknown:
            raise ValueError(f"unknown config fields: {', '.join(unknown)}")
        return cls(**raw)

    def with_overrides(self, **values: Any) -> "Config":
        return replace(self, **{k: v for k, v in values.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
