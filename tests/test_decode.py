from __future__ import annotations

import hashlib
from pathlib import Path

from golftracer.decode import decode_window, probe


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.mp4"


def _hashes(frames):
    return [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]


def test_window_decode_is_deterministic() -> None:
    first, first_pts = decode_window(FIXTURE, 0.25, 1.5, gray=False)
    second, second_pts = decode_window(FIXTURE, 0.25, 1.5, gray=False)
    assert _hashes(first) == _hashes(second)
    assert first_pts.tolist() == second_pts.tolist()


def test_rounded_and_exact_start_have_same_first_frame() -> None:
    exact, _ = decode_window(FIXTURE, 1 / 3, 0.5)
    rounded, _ = decode_window(FIXTURE, round(1 / 3, 3), 0.5)
    assert _hashes(exact[:1]) == _hashes(rounded[:1])


def test_dimensions_are_the_decoded_display_dimensions() -> None:
    meta = probe(FIXTURE)
    frames, _ = decode_window(FIXTURE, 0.0, 0.1)
    assert (meta.width, meta.height) == (480, 854)
    assert frames.shape[1:3] == (meta.height, meta.width)

