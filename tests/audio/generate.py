"""Utility script to (re)generate the deterministic test WAV files.

Run once to populate the ``tests/audio/`` directory:

    python tests/audio/generate.py

The files are also generated automatically by the pytest ``test_audio_*``
fixtures in ``test_transcription.py``, so you rarely need to run this by hand.
"""
from __future__ import annotations

import math
import pathlib
import struct
import wave

_HERE = pathlib.Path(__file__).parent


def _write_wav(
    path: pathlib.Path,
    duration_s: float,
    sample_rate: int = 16_000,
    frequency: float = 440.0,
    amplitude: float = 0.3,
) -> None:
    """Write a mono 16-bit PCM WAV with a pure sine tone."""
    n_samples = int(duration_s * sample_rate)
    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(n_samples)
    ]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


if __name__ == "__main__":
    _HERE.mkdir(parents=True, exist_ok=True)
    _write_wav(_HERE / "test_1s.wav", duration_s=1.0)
    _write_wav(_HERE / "test_30s.wav", duration_s=30.0)
    print("Generated test WAV files in", _HERE)
