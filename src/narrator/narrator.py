"""StreamingAINarrator — Real-time video frame narration via Gemini + TTS.

Processes sequential video frames through a three-phase pipeline:
  1. Vision AI generates a short text description (Gemini)
  2. Text-to-Speech converts it to audio (Google Cloud TTS)
  3. Audio is mechanically fitted to a fixed duration window (Pydub)

Integrates with :class:`~ingestor.models.MediaChunk` from the
:mod:`ingestor` package for end-to-end video-to-narration pipelines.
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass, field
import sys
import pathlib

# Ensure the ingestor package is importable (sibling package under src/)
_src_dir = str(pathlib.Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from google import genai
from google.cloud import texttospeech_v1
from google.genai import types as genai_types
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

from typing import TypeAlias

from ingestor.models import MediaChunk

JpegFrames: TypeAlias = list[bytes]
"""A list of compressed JPEG byte strings representing sequential video frames."""

DEFAULT_SAMPLE_COUNT: int = 5
"""Number of frames to sample from a MediaChunk for vision processing."""


@dataclass(frozen=True, slots=True)
class NarratorResult:
    """Output of a single narration cycle."""

    audio_bytes: bytes
    """MP3 audio fitted to the target duration."""

    text: str
    """The generated description, or empty string on failure."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CONSECUTIVE_FAILURES: int = 3
VISION_TIMEOUT: float = 10.0  # seconds
TTS_TIMEOUT: float = 4.0  # seconds

DEFAULT_TARGET_DURATION_MS: int = 10000
DURATION_TOLERANCE_MS: int = 50

TTS_VOICE_NAME: str = "en-US-Neural2-J"
TTS_LANGUAGE_CODE: str = "en-US"
TTS_SPEAKING_RATE: float = 1.15

GEMINI_MODEL: str = "gemini-2.5-flash"

VISION_PROMPT_TEMPLATE: str = (
    "{context_clause}"
    "Describe the continuous action in these sequential frames. "
    "CRITICAL: Write AT MOST 30 words."
    "Focus only on physical movement."
)

# ---------------------------------------------------------------------------
# StreamingAINarrator
# ---------------------------------------------------------------------------


class StreamingAINarrator:
    """Processes video frame chunks into time-synchronised narration audio.

    Each call to :meth:`process_chunk` accepts a batch of JPEG frames,
    generates a short text description via Gemini, converts it to speech,
    and returns exactly ``target_duration_ms`` milliseconds of MP3 audio.

    Args:
        project_id: GCP project ID for Vertex AI and TTS.
        location: GCP region (e.g. ``"us-central1"``).
        target_duration_ms: Required output duration in milliseconds.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        target_duration_ms: int = DEFAULT_TARGET_DURATION_MS,
    ) -> None:
        self._target_duration_ms = target_duration_ms

        # State
        self._previous_context: str = ""
        self._consecutive_failures: int = 0

        # Gemini client (google-genai SDK)
        self._genai_client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        # TTS client
        self._tts_client = texttospeech_v1.TextToSpeechAsyncClient()

        logger.info(
            "StreamingAINarrator initialised (model=%s, voice=%s, target=%dms)",
            GEMINI_MODEL,
            TTS_VOICE_NAME,
            self._target_duration_ms,
        )

    async def warmup(self) -> None:
        """Pre-establish API connections to avoid cold-start timeouts.

        Sends a minimal request to both Gemini and TTS so that the first
        real :meth:`process_chunk` call doesn't pay the connection setup cost.
        """
        logger.info("Warming up API connections...")

        # Warm up Gemini
        try:
            await self._genai_client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents="Say OK.",
            )
            logger.info("Gemini warmup complete.")
        except Exception as e:
            logger.warning("Gemini warmup failed (non-fatal): %s", e)

        # Warm up TTS
        try:
            await self._tts_client.synthesize_speech(
                input=texttospeech_v1.SynthesisInput(text="OK"),
                voice=texttospeech_v1.VoiceSelectionParams(
                    language_code=TTS_LANGUAGE_CODE,
                    name=TTS_VOICE_NAME,
                ),
                audio_config=texttospeech_v1.AudioConfig(
                    audio_encoding=texttospeech_v1.AudioEncoding.MP3,
                ),
            )
            logger.info("TTS warmup complete.")
        except Exception as e:
            logger.warning("TTS warmup failed (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Phase 1 — Vision AI
    # ------------------------------------------------------------------

    async def _generate_description(self, frames: JpegFrames) -> str:
        """Send frames to Gemini and return a short action description.

        Args:
            frames: Compressed JPEG byte strings (typically 5 frames).

        Returns:
            A single-sentence description of the physical action.

        Raises:
            ValueError: If the model response is blocked by safety filters.
            google.api_core.exceptions.GoogleAPICallError: On API failure.
        """
        image_parts = [
            genai_types.Part.from_bytes(data=frame, mime_type="image/jpeg")
            for frame in frames
        ]

        context_clause = (
            f"Previous context: \"{self._previous_context}\". "
            if self._previous_context
            else ""
        )
        prompt_text = VISION_PROMPT_TEMPLATE.format(context_clause=context_clause)

        contents: list[genai_types.Part | str] = [*image_parts, prompt_text]

        response = await self._genai_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
        )

        # Free frame references as early as possible.
        del frames, image_parts

        # Safety filter check
        if not response.text:
            reason = (
                response.candidates[0].finish_reason
                if response.candidates
                else "unknown"
            )
            raise ValueError(f"Gemini returned no text (finish_reason={reason})")

        text = response.text.strip()
        logger.debug("Vision description: %s", text)
        return text

    # ------------------------------------------------------------------
    # Phase 2 — Text-to-Speech
    # ------------------------------------------------------------------

    async def _generate_tts(self, text: str) -> bytes:
        """Convert *text* to MP3 speech audio.

        Args:
            text: Short sentence to synthesise.

        Returns:
            Raw MP3 audio bytes.
        """
        synthesis_input = texttospeech_v1.SynthesisInput(text=text)

        voice = texttospeech_v1.VoiceSelectionParams(
            language_code=TTS_LANGUAGE_CODE,
            name=TTS_VOICE_NAME,
        )

        audio_config = texttospeech_v1.AudioConfig(
            audio_encoding=texttospeech_v1.AudioEncoding.MP3,
            speaking_rate=TTS_SPEAKING_RATE,
        )

        response = await self._tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        logger.debug("TTS returned %d bytes", len(response.audio_content))
        return response.audio_content

    # ------------------------------------------------------------------
    # Phase 3 — Synchronisation Engine
    # ------------------------------------------------------------------

    def _sync_audio_duration(self, audio_bytes: bytes) -> bytes:
        """Fit *audio_bytes* to exactly ``target_duration_ms``.

        - **Underflow:** Silence is appended.
        - **Overflow:** Audio is sped up via :pymethod:`pydub.AudioSegment.speedup`.
        - **Within tolerance (±50 ms):** Returned as-is.

        Args:
            audio_bytes: Raw MP3 audio from TTS.

        Returns:
            MP3 bytes with duration == ``target_duration_ms`` (±tolerance).
        """
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        duration = len(audio)
        target = self._target_duration_ms

        if abs(duration - target) <= DURATION_TOLERANCE_MS:
            logger.debug("Audio duration %dms within tolerance, no adjustment", duration)
            return audio_bytes

        if duration < target:
            silence = AudioSegment.silent(
                duration=target - duration,
                frame_rate=audio.frame_rate,
            )
            audio = audio + silence
            logger.debug("Padded audio from %dms to %dms", duration, len(audio))
        else:
            ratio = duration / target
            audio = audio.speedup(playback_speed=ratio)
            logger.debug(
                "Sped up audio from %dms (ratio=%.2f) to %dms",
                duration,
                ratio,
                len(audio),
            )

        buffer = io.BytesIO()
        audio.export(buffer, format="mp3")
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Silent Fallback
    # ------------------------------------------------------------------

    def _generate_silent_audio(self) -> bytes:
        """Return exactly ``target_duration_ms`` of silence as MP3.

        Uses 24 kHz mono to match typical TTS output characteristics.
        """
        silence = AudioSegment.silent(
            duration=self._target_duration_ms,
            frame_rate=24000,
        )
        buffer = io.BytesIO()
        silence.export(buffer, format="mp3")
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Frame Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_frames(
        frames: JpegFrames, n: int = DEFAULT_SAMPLE_COUNT
    ) -> JpegFrames:
        """Select *n* evenly-spaced frames from *frames*.

        If there are fewer than *n* frames available, all are returned.
        """
        total = len(frames)
        if total <= n:
            return frames
        step = total / n
        return [frames[int(i * step)] for i in range(n)]

    # ------------------------------------------------------------------
    # Main Orchestrator
    # ------------------------------------------------------------------

    async def process_chunk(self, frames: JpegFrames) -> NarratorResult:
        """Process a batch of JPEG frames into narration audio.

        This is the low-level entry-point that accepts raw JPEG byte
        lists.  Prefer :meth:`process_media_chunk` when working with
        the ingestor pipeline.

        Args:
            frames: List of compressed JPEG byte strings.

        Returns:
            A :class:`NarratorResult` with fitted audio and description text.
        """
        try:
            # Phase 1 — Vision
            text = await asyncio.wait_for(
                self._generate_description(frames),
                timeout=VISION_TIMEOUT,
            )

            # State update on successful vision
            self._previous_context = text
            self._consecutive_failures = 0

            # Phase 2 — TTS
            raw_audio = await asyncio.wait_for(
                self._generate_tts(text),
                timeout=TTS_TIMEOUT,
            )

            # Phase 3 — Sync
            fitted_audio = self._sync_audio_duration(raw_audio)

            logger.info("Chunk processed: '%s'", text)
            return NarratorResult(audio_bytes=fitted_audio, text=text)

        except (asyncio.TimeoutError, Exception) as exc:
            self._consecutive_failures += 1
            logger.warning(
                "Chunk failed (consecutive=%d): %s",
                self._consecutive_failures,
                exc,
                exc_info=True,
            )

            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Resetting context after %d consecutive failures",
                    self._consecutive_failures,
                )
                self._previous_context = ""

            return NarratorResult(
                audio_bytes=self._generate_silent_audio(),
                text="",
            )

    async def process_media_chunk(self, chunk: MediaChunk) -> NarratorResult:
        """Process a :class:`MediaChunk` from the ingestor.

        Samples a small number of evenly-spaced JPEG frames from
        ``chunk.compressed_frames`` and delegates to :meth:`process_chunk`.

        Args:
            chunk: A ``MediaChunk`` produced by :class:`ChunkingIngestor`.

        Returns:
            A :class:`NarratorResult` with fitted audio and description text.
        """
        sampled = self._sample_frames(chunk.compressed_frames)
        logger.info(
            "Processing media chunk [%.1fs–%.1fs] (%d frames, sampled %d)",
            chunk.start_time,
            chunk.end_time,
            len(chunk.compressed_frames),
            len(sampled),
        )
        return await self.process_chunk(sampled)
