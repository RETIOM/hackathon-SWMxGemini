from __future__ import annotations

import pathlib
import sys

import numpy as np

# Make src/ importable for tests when pytest runs from repository root.
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingestor.models import MediaChunk
from synchronizer import Synchronizer


def test_decode_chunk_audio_applies_ducking_masks() -> None:
    sync = Synchronizer()

    # 8 stereo samples, constant amplitude.
    audio = np.full((8, 2), 0.8, dtype=np.float32)
    chunk = MediaChunk(
        start_time=0.0,
        end_time=2.0,
        raw_video_frames=[np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)],
        compressed_frames=[b"jpg", b"jpg"],
        raw_audio_bytes=audio.reshape(-1).tobytes(),
        audio_sample_rate=4,
        audio_channels=2,
    )

    mixed = sync._decode_chunk_audio(
        chunk=chunk,
        vap_masks=[0.5, 1.0],
        sample_rate=4,
        channels=2,
        duration_s=2.0,
    )

    assert mixed is not None
    raw = mixed.raw_data or b""
    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    # First half should be attenuated relative to second half.
    first_half_peak = np.max(np.abs(pcm[:4]))
    second_half_peak = np.max(np.abs(pcm[4:]))
    assert first_half_peak < second_half_peak
