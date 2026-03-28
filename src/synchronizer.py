from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Iterable, cast

import av
import numpy as np
from pydub import AudioSegment

from ingestor.models import MediaChunk
from narrator.narrator import NarratorResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Synchronizer:
    """Combines original chunk media, VAP masks, and narrator audio into MP4 bytes."""

    video_codec: str = "libx264"
    audio_codec: str = "aac"
    movflags: str = "frag_keyframe+empty_moov+default_base_moof"

    def process(
        self,
        chunk: MediaChunk,
        narrator_result: NarratorResult,
        vap_masks: list[float],
    ) -> bytes:
        """Build one self-contained MP4 segment for a media chunk."""
        if not chunk.raw_video_frames:
            return b""

        duration_s = max(chunk.end_time - chunk.start_time, 0.0)
        fps = self._infer_fps(frame_count=len(chunk.raw_video_frames), duration_s=duration_s)
        mixed = self._mix_audio(chunk, narrator_result, vap_masks, duration_s)
        return self._mux_segment(chunk, mixed, fps)

    def concat_segments(self, segments: Iterable[bytes]) -> bytes:
        """Decode/re-encode chunk segments into one continuous MP4 payload."""
        segment_list = [s for s in segments if s]
        if not segment_list:
            return b""

        output_buffer = io.BytesIO()
        out_container = av.open(
            output_buffer,
            mode="w",
            format="mp4",
            container_options={"movflags": self.movflags},
        )

        out_video_stream: Any = None
        out_audio_stream: Any = None

        try:
            for segment in segment_list:
                in_container = av.open(io.BytesIO(segment), mode="r", format="mp4")
                try:
                    if out_video_stream is None:
                        video_streams = [s for s in in_container.streams if s.type == "video"]
                        if video_streams:
                            first_v = video_streams[0]
                            first_v_cc = cast(Any, first_v.codec_context)
                            out_video_stream = out_container.add_stream(self.video_codec, rate=int(first_v.average_rate or 30))
                            out_video_stream.width = first_v_cc.width
                            out_video_stream.height = first_v_cc.height
                            out_video_stream.pix_fmt = "yuv420p"

                    if out_audio_stream is None:
                        audio_streams = [s for s in in_container.streams if s.type == "audio"]
                        if audio_streams:
                            first_a = cast(Any, audio_streams[0])
                            out_audio_stream = out_container.add_stream(self.audio_codec, rate=first_a.rate or 48000)
                            if first_a.channels:
                                out_audio_stream.layout = "stereo" if first_a.channels >= 2 else "mono"

                    for frame in in_container.decode(video=0):
                        frame.pts = None
                        if out_video_stream is not None:
                            for packet in out_video_stream.encode(frame):
                                out_container.mux(packet)

                    if out_audio_stream is not None:
                        for frame in in_container.decode(audio=0):
                            frame.pts = None
                            for packet in out_audio_stream.encode(frame):
                                out_container.mux(packet)
                finally:
                    in_container.close()

            if out_video_stream is not None:
                for packet in out_video_stream.encode():
                    out_container.mux(packet)
            if out_audio_stream is not None:
                for packet in out_audio_stream.encode():
                    out_container.mux(packet)
        finally:
            out_container.close()

        return output_buffer.getvalue()

    def _infer_fps(self, frame_count: int, duration_s: float) -> int:
        if frame_count <= 1:
            return 1
        if duration_s <= 0:
            return 30
        return max(1, int(round(frame_count / duration_s)))

    def _mix_audio(
        self,
        chunk: MediaChunk,
        narrator_result: NarratorResult,
        vap_masks: list[float],
        duration_s: float,
    ) -> AudioSegment | None:
        target_duration_ms = int(max(duration_s, 0.0) * 1000)
        if target_duration_ms <= 0:
            target_duration_ms = 1

        sample_rate = max(1, int(chunk.audio_sample_rate or 48000))
        channels = max(1, int(chunk.audio_channels or 1))

        base = self._decode_chunk_audio(chunk, vap_masks, sample_rate, channels, duration_s)
        if base is None:
            base = AudioSegment.silent(duration=target_duration_ms, frame_rate=sample_rate).set_channels(channels)

        narration = self._decode_narration_audio(narrator_result, sample_rate, channels)
        if narration is not None:
            # Pydub overlay truncates the overlaid track to the length of the base track.
            # We must pad the base track with silence if the TTS is longer than the base!
            if len(narration) > len(base):
                silence = AudioSegment.silent(duration=len(narration) - len(base), frame_rate=sample_rate).set_channels(channels)
                base = base + silence
                
            base = base.overlay(narration)

        return self._fit_duration(base, target_duration_ms)

    def _decode_chunk_audio(
        self,
        chunk: MediaChunk,
        vap_masks: list[float],
        sample_rate: int,
        channels: int,
        duration_s: float,
    ) -> AudioSegment | None:
        if not chunk.raw_audio_bytes:
            return None

        audio_f32 = np.frombuffer(chunk.raw_audio_bytes, dtype=np.float32).copy()
        if audio_f32.size == 0:
            return None

        usable = (audio_f32.size // channels) * channels
        if usable == 0:
            return None
        audio_f32 = audio_f32[:usable].reshape(-1, channels)

        frame_count = len(chunk.raw_video_frames)
        if frame_count > 0 and duration_s > 0:
            fps = frame_count / duration_s
            samples_per_frame = sample_rate / max(fps, 1e-6)
            max_masks = min(len(vap_masks), frame_count)
            for frame_idx in range(max_masks):
                mask = float(vap_masks[frame_idx])
                start = int(frame_idx * samples_per_frame)
                end = int((frame_idx + 1) * samples_per_frame)
                start = max(0, min(start, audio_f32.shape[0]))
                end = max(start, min(end, audio_f32.shape[0]))
                if start < end:
                    audio_f32[start:end, :] *= mask

        pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        return AudioSegment(
            data=pcm16.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=channels,
        )

    def _decode_narration_audio(
        self,
        narrator_result: NarratorResult,
        sample_rate: int,
        channels: int,
    ) -> AudioSegment | None:
        if not narrator_result.audio_bytes:
            return None
        try:
            narration = AudioSegment.from_file(io.BytesIO(narrator_result.audio_bytes), format="mp3")
        except Exception:
            logger.warning("Failed to decode narrator MP3 bytes, dropping narration track", exc_info=True)
            return None
        return narration.set_frame_rate(sample_rate).set_channels(channels)

    def _fit_duration(self, audio: AudioSegment, target_duration_ms: int) -> AudioSegment:
        # If audio is longer, DO NOT truncate. Return as-is so video can be padded to match.
        return audio

    def _mux_segment(self, chunk: MediaChunk, mixed_audio: AudioSegment | None, fps: int) -> bytes:
        output_buffer = io.BytesIO()
        container = av.open(
            output_buffer,
            mode="w",
            format="mp4",
            container_options={"movflags": self.movflags},
        )

        v0 = chunk.raw_video_frames[0]
        height, width = int(v0.shape[0]), int(v0.shape[1])
        video_stream: Any = container.add_stream(self.video_codec, rate=fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"

        audio_stream: Any = None
        wav_container = None
        if mixed_audio is not None and len(mixed_audio) > 0:
            wav_buffer = io.BytesIO()
            mixed_audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            wav_container = av.open(wav_buffer, mode="r", format="wav")
            in_audio_stream = cast(Any, next((s for s in wav_container.streams if s.type == "audio"), None))
            if in_audio_stream is not None:
                audio_stream = container.add_stream(self.audio_codec, rate=in_audio_stream.rate or mixed_audio.frame_rate)
                if in_audio_stream.channels:
                    audio_stream.layout = "stereo" if in_audio_stream.channels >= 2 else "mono"

        frames_to_write = chunk.raw_video_frames
        if mixed_audio is not None and len(mixed_audio) > 0:
            target_duration_s = len(mixed_audio) / 1000.0
            required_frames = int(round(target_duration_s * fps))
            if len(frames_to_write) < required_frames:
                last_frame = frames_to_write[-1]
                frames_to_write = frames_to_write + [last_frame] * (required_frames - len(frames_to_write))

        all_packets = []
        try:
            for frame_ndarray in frames_to_write:
                vf = av.VideoFrame.from_ndarray(frame_ndarray, format="rgb24")
                for packet in video_stream.encode(vf):
                    all_packets.append(packet)

            for packet in video_stream.encode():
                all_packets.append(packet)

            if audio_stream is not None and wav_container is not None:
                for af in wav_container.decode(audio=0):
                    af.pts = None
                    for packet in audio_stream.encode(af):
                        all_packets.append(packet)
                for packet in audio_stream.encode():
                    all_packets.append(packet)

            # Sort packets by DTS (or PTS if DTS is missing) to correctly interleave the fragment
            def get_timestamp(pkt):
                if pkt.dts is not None:
                    return pkt.dts * float(getattr(pkt.stream.time_base, "numerator", 1)) / float(getattr(pkt.stream.time_base, "denominator", 1))
                if pkt.pts is not None:
                    return pkt.pts * float(getattr(pkt.stream.time_base, "numerator", 1)) / float(getattr(pkt.stream.time_base, "denominator", 1))
                return 0

            all_packets.sort(key=get_timestamp)
            
            for packet in all_packets:
                container.mux(packet)
        finally:
            if wav_container is not None:
                wav_container.close()
            container.close()

        return output_buffer.getvalue()
