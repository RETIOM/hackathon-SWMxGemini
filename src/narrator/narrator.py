"""StreamingAINarrator — Real-time video frame narration via Gemini + TTS.

Processes sequential video frames through a three-phase pipeline:
  1. Vision AI generates a short text description (Gemini)
  2. Text-to-Speech converts it to audio (Google Cloud TTS)
  3. Audio is mechanically fitted to a fixed duration window (Pydub)
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass, field

from google import genai
from google.cloud import texttospeech_v1
from google.genai import types as genai_types
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

from typing import TypeAlias

JpegFrames: TypeAlias = list[bytes]
"""A list of compressed JPEG byte strings representing sequential video frames."""


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
VISION_TIMEOUT: float = 6.0  # seconds
TTS_TIMEOUT: float = 4.0  # seconds

DEFAULT_TARGET_DURATION_MS: int = 5000
DURATION_TOLERANCE_MS: int = 50

TTS_VOICE_NAME: str = "en-US-Neural2-J"
TTS_LANGUAGE_CODE: str = "en-US"
TTS_SPEAKING_RATE: float = 1.15

GEMINI_MODEL: str = "gemini-2.5-flash"

VISION_PROMPT_TEMPLATE: str = (
    "{context_clause}"
    "Describe the continuous action in these sequential frames. "
    "CRITICAL: Write ONE sentence. Maximum 10 words. "
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
    # Main Orchestrator
    # ------------------------------------------------------------------

    async def process_chunk(self, frames: JpegFrames) -> NarratorResult:
        """Process a batch of JPEG frames into narration audio.

        This is the public entry-point called by the upstream ingestor.
        It enforces strict timeouts on each API phase and falls back to
        silence on any failure.

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
