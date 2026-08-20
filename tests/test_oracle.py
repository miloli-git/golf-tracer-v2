from __future__ import annotations

from golftracer.oracle import ball_tracks_from_data, club_tracks_from_data


def test_oracle_adapter_on_inline_samples() -> None:
    clubs = club_tracks_from_data({
        "swings": [{
            "t_impact": 1.25,
            "top_t": 1.1,
            "points": [
                {"rel_frame": 0, "x": 10, "y": 20, "t_s": 1.0, "source": "calibrated"},
                {"rel_frame": 1, "x": 12, "y": 18, "t_s": 1.1, "source": "calibrated"},
            ],
        }]
    })
    assert clubs[0][0] == 1.25
    assert clubs[0][1].phase == "club"
    assert clubs[0][1].observations[1].x == 12
    assert clubs[0][1].metadata["top_t"] == 1.1

    balls = ball_tracks_from_data([
        {"t": 1.25, "ok": True, "apex_frame": 4, "n_descent": 2, "points": [
            {"rel_frame": 1, "u": 50, "v": 60, "t_s": 1.266},
            {"rel_frame": 4, "u": 55, "v": 40, "t_s": 1.316},
        ]},
        {"t": 2.0, "ok": False, "points": []},
    ])
    assert balls[0][1] is not None
    assert balls[0][1].phase == "ball"
    assert balls[0][1].observations[0].frame_index == 1
    assert balls[1] == (2.0, None)
