from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class MediaChunk:
    """
    Represents exactly 1 chunk (e.g. 5.0 seconds) of audio/video from the Ingestor queue.
    Carries heavily optimized byte strings natively in memory.
    """
    start_time: float      # Window start in seconds
    end_time: float        # Window end in seconds
    
    # Store complete uncompressed frames representing ALL frames in this window natively.
    # Essential for zero-delay operations across pure memory (no disk I/O).
    raw_video_frames: List[np.ndarray] 
    
    # ALL tightly compressed JPEGs intercepted for Gemini visual framing logic
    compressed_frames: List[bytes]
    
    # Appended raw audio track spanning this 5-second chunk block
    raw_audio_bytes: Optional[bytes] = None
    audio_sample_rate: int = 48000   # Source sample rate from PyAV
    audio_channels: int = 2          # Number of audio channels
