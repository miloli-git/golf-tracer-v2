"""Generate the small deterministic audiovisual fixture used by the tests."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import wave

import numpy as np


DURATION_S = 3.0
SAMPLE_RATE = 16_000
CLICK_TIMES = (0.5, 1.5, 2.5)


def main() -> None:
    root = Path(__file__).resolve().parent
    audio_path = root / "synthetic-clicks.wav"
    output_path = root / "synthetic.mp4"
    samples = np.zeros(int(DURATION_S * SAMPLE_RATE), dtype=np.float64)
    click_length = int(0.008 * SAMPLE_RATE)
    envelope = np.hanning(click_length * 2)[:click_length]
    tone = np.sin(2 * math.pi * 4_000 * np.arange(click_length) / SAMPLE_RATE)
    click = 0.95 * envelope * tone
    for timestamp in CLICK_TIMES:
        start = int(timestamp * SAMPLE_RATE)
        samples[start : start + click_length] += click
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(audio_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        (
            "color=c=0x18324a:s=480x854:r=30:d=3,"
            "drawbox=x=80:y=300:w=300:h=300:color=white:t=fill:"
            "enable='between(t,0.43,0.50)+between(t,1.43,1.50)+between(t,2.43,2.50)'"
        ),
        "-i", str(audio_path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-shortest",
        "-map_metadata", "-1", "-metadata", "creation_time=", "-y", str(output_path),
    ]
    subprocess.run(command, check=True)
    audio_path.unlink()
    if output_path.stat().st_size >= 2_000_000:
        raise RuntimeError("fixture exceeds the 2 MB limit")


if __name__ == "__main__":
    main()
