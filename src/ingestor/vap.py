import torch
import torchaudio
import numpy as np
from typing import List
from ingestor.models import MediaChunk


class VAP:
    """
    Voice Activity Predictor (VAP) for Real-Time Audiodescription.

    Pipeline-compatible: accepts MediaChunk objects from the Ingestor queue.
    Produces a per-second ducking mask list that the Synchronizer consumes.

    Handles the PyAV audio format (float32 planar, typically 48kHz stereo)
    and resamples to 16kHz mono for Silero VAD.
    """

    def __init__(
        self,
        strategy: str = "silero",
        ducking_level: float = 0.2,
        rms_threshold: float = 0.05,
        window_duration: float = 1.0,
    ):
        """
        :param strategy: "silero" (optimal) or "rms" (fast fallback).
        :param ducking_level: Volume multiplier when voice is detected (e.g. 0.2).
        :param rms_threshold: Threshold for RMS energy if fallback strategy is used.
        :param window_duration: Size of each analysis sub-window in seconds.
        """
        self.strategy = strategy
        self.ducking_level = ducking_level
        self.rms_threshold = rms_threshold
        self.window_duration = window_duration
        self.vad_sample_rate = 16000
        self._resamplers = {}

        if self.strategy == "silero":
            try:
                self.model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                self.model.eval()
            except Exception as e:
                print(
                    f"Warning: Failed to load Silero VAD ({e}), falling back to 'rms'."
                )
                self.strategy = "rms"

    def _get_resampler(self, source_rate: int):
        """Get or create a cached resampler for converting source_rate -> 16kHz."""
        if source_rate not in self._resamplers:
            self._resamplers[source_rate] = torchaudio.transforms.Resample(
                orig_freq=source_rate, new_freq=self.vad_sample_rate
            )
        return self._resamplers[source_rate]

    def _prepare_audio(self, chunk: MediaChunk) -> torch.Tensor:
        """
        Convert raw PyAV audio bytes into a 16kHz mono float32 tensor.

        PyAV outputs float32 planar (fltp) format. When to_ndarray() is called
        on a stereo frame, the shape is (channels, samples). These are then
        serialized to bytes and concatenated across frames in the ingestor.

        Since frames are concatenated as raw bytes, the resulting buffer is
        a flat float32 array with interleaved channel data per-frame.
        We reshape, average to mono, and resample to 16kHz.
        """
        audio_f32 = np.frombuffer(chunk.raw_audio_bytes, dtype=np.float32)

        channels = chunk.audio_channels
        source_rate = chunk.audio_sample_rate

        if channels > 1 and len(audio_f32) % channels == 0:
            audio_f32 = audio_f32.reshape(-1, channels).mean(axis=1)

        audio_tensor = torch.from_numpy(audio_f32.copy()).clamp(-1.0, 1.0)

        if source_rate != self.vad_sample_rate:
            resampler = self._get_resampler(source_rate)
            audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)

        return audio_tensor

    def process_chunk(self, chunk: MediaChunk) -> List[float]:
        """
        Pipeline entry point. Takes a MediaChunk from the Ingestor queue
        and returns a list of per-frame ducking masks (one float per video frame).

        Internally runs VAD on 1-second audio windows, then maps each video
        frame to the mask of the second it falls in.

        :param chunk: A MediaChunk from ChunkingIngestor.
        :return: List of float masks, one per video frame in the chunk.
        """
        num_frames = len(chunk.raw_video_frames)
        duration = chunk.end_time - chunk.start_time

        if chunk.raw_audio_bytes is None or len(chunk.raw_audio_bytes) == 0:
            return [1.0] * num_frames

        audio_tensor = self._prepare_audio(chunk)

        if len(audio_tensor) == 0:
            return [1.0] * num_frames

        samples_per_window = int(self.vad_sample_rate * self.window_duration)
        second_masks = []

        for i in range(0, len(audio_tensor), samples_per_window):
            window = audio_tensor[i : i + samples_per_window]
            if len(window) < 256:
                second_masks.append(1.0)
                continue
            second_masks.append(self._analyze_window(window))

        if not second_masks:
            return [1.0] * num_frames

        frame_masks = []
        for frame_idx in range(num_frames):
            frame_time = (frame_idx / num_frames) * duration
            window_idx = min(
                int(frame_time / self.window_duration), len(second_masks) - 1
            )
            frame_masks.append(second_masks[window_idx])

        return frame_masks

    def _analyze_window(self, audio_tensor: torch.Tensor) -> float:
        """
        Run VAD on a single 1-second window of 16kHz mono audio.
        Returns ducking_level if speech detected, 1.0 otherwise.
        """
        if self.strategy == "silero":
            with torch.no_grad():
                chunk_len = 512
                max_prob = 0.0
                for i in range(0, len(audio_tensor), chunk_len):
                    chunk = audio_tensor[i : i + chunk_len]
                    if len(chunk) < chunk_len:
                        break
                    prob = self.model(chunk, self.vad_sample_rate).item()
                    if prob > max_prob:
                        max_prob = prob

            return self.ducking_level if max_prob > 0.5 else 1.0

        elif self.strategy == "rms":
            rms = torch.sqrt(torch.mean(audio_tensor**2)).item()
            return self.ducking_level if rms > self.rms_threshold else 1.0

        return 1.0
