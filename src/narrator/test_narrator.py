"""Tests for StreamingAINarrator.

Tests the sync engine and silent fallback with real pydub operations,
and the orchestrator failure paths with mocked API clients.
"""

from __future__ import annotations

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from narrator.narrator import (
    DEFAULT_TARGET_DURATION_MS,
    DURATION_TOLERANCE_MS,
    MAX_CONSECUTIVE_FAILURES,
    NarratorResult,
    StreamingAINarrator,
)


def _make_audio_bytes(duration_ms: int, freq: int = 440) -> bytes:
    """Generate MP3 bytes of a sine wave with a given duration."""
    tone = Sine(freq).to_audio_segment(duration=duration_ms)
    buf = io.BytesIO()
    tone.export(buf, format="mp3")
    return buf.getvalue()


def _audio_duration_ms(mp3_bytes: bytes) -> int:
    """Return the duration in ms of MP3 bytes."""
    return len(AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3"))


def _make_narrator() -> StreamingAINarrator:
    """Create a narrator instance with mocked clients."""
    with (
        patch("narrator.narrator.genai.Client"),
        patch("narrator.narrator.texttospeech_v1.TextToSpeechAsyncClient"),
    ):
        return StreamingAINarrator(project_id="test-project")


class TestSyncAudioDuration:
    """Tests for _sync_audio_duration (pure pydub logic, no mocking)."""

    def test_underflow_pads_to_target(self) -> None:
        narrator = _make_narrator()
        short_audio = _make_audio_bytes(3000)

        result = narrator._sync_audio_duration(short_audio)
        duration = _audio_duration_ms(result)

        assert abs(duration - DEFAULT_TARGET_DURATION_MS) <= DURATION_TOLERANCE_MS

    def test_within_tolerance_returns_unchanged(self) -> None:
        narrator = _make_narrator()
        ok_audio = _make_audio_bytes(4970)

        result = narrator._sync_audio_duration(ok_audio)

        assert result == ok_audio

    def test_overflow_speeds_up(self) -> None:
        narrator = _make_narrator()
        long_audio = _make_audio_bytes(7000)

        result = narrator._sync_audio_duration(long_audio)
        duration = _audio_duration_ms(result)

        assert duration < 7000
        assert duration < DEFAULT_TARGET_DURATION_MS + 500


class TestSilentFallback:
    """Tests for _generate_silent_audio."""

    def test_produces_exact_duration(self) -> None:
        narrator = _make_narrator()

        result = narrator._generate_silent_audio()
        duration = _audio_duration_ms(result)

        assert abs(duration - DEFAULT_TARGET_DURATION_MS) <= DURATION_TOLERANCE_MS

    def test_returns_valid_mp3(self) -> None:
        narrator = _make_narrator()

        result = narrator._generate_silent_audio()

        audio = AudioSegment.from_file(io.BytesIO(result), format="mp3")
        assert audio.channels >= 1


class TestProcessChunk:
    """Tests for process_chunk failure paths."""

    def test_timeout_returns_silent_audio(self) -> None:
        narrator = _make_narrator()
        narrator._generate_description = AsyncMock(side_effect=asyncio.TimeoutError())

        result = asyncio.run(narrator.process_chunk([b"fake_jpeg"]))

        assert isinstance(result, NarratorResult)
        assert result.text == ""
        assert len(result.audio_bytes) > 0
        assert narrator._consecutive_failures == 1

    def test_context_resets_after_max_failures(self) -> None:
        narrator = _make_narrator()
        narrator._previous_context = "some old context"
        narrator._generate_description = AsyncMock(side_effect=asyncio.TimeoutError())

        for _ in range(MAX_CONSECUTIVE_FAILURES):
            asyncio.run(narrator.process_chunk([b"fake_jpeg"]))

        assert narrator._consecutive_failures == MAX_CONSECUTIVE_FAILURES
        assert narrator._previous_context == ""

    def test_success_resets_failure_counter(self) -> None:
        narrator = _make_narrator()
        narrator._consecutive_failures = 2
        narrator._generate_description = AsyncMock(return_value="A person walks")
        narrator._generate_tts = AsyncMock(return_value=_make_audio_bytes(4500))

        result = asyncio.run(narrator.process_chunk([b"fake_jpeg"]))

        assert result.text == "A person walks"
        assert narrator._consecutive_failures == 0
        assert narrator._previous_context == "A person walks"

    def test_tts_timeout_returns_silent(self) -> None:
        narrator = _make_narrator()
        narrator._generate_description = AsyncMock(return_value="Walking fast")
        narrator._generate_tts = AsyncMock(side_effect=asyncio.TimeoutError())

        result = asyncio.run(narrator.process_chunk([b"fake_jpeg"]))

        assert result.text == ""
        assert narrator._consecutive_failures == 1
