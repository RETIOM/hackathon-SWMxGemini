from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class MediaChunk:
    """
    Represents exactly 1 chunk (e.g. 5.0 seconds) of audio/video from the Ingestor queue.
    Carries heavily optimized byte strings natively in memory.
    """

    start_time: float
    end_time: float

    raw_video_frames: List[np.ndarray]

    compressed_frames: List[bytes]

    raw_audio_bytes: Optional[bytes] = None
    audio_sample_rate: int = 48000
    audio_channels: int = 2
